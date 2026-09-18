# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Collective latency on a selected CMP group, with graph and eager correctness."""

import argparse
import datetime
import json
import os
import socket
import time
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world, port, args):
    torch.set_num_threads(1)
    device_rank = (
        args.visible_gpus.split(",").index(args.gpus.split(",")[rank])
        if args.visible_gpus
        else rank
    )
    torch.accelerator.set_device_index(device_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        timeout=datetime.timedelta(seconds=180),
        device_id=torch.device("cuda", device_rank),
    )
    custom = None
    if args.backend == "custom":
        from vllm.config import VllmConfig, set_current_vllm_config
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.platforms import current_platform
        from vllm.platforms.interface import set_assigned_physical_gpu_ids

        if args.visible_gpus:
            set_assigned_physical_gpu_ids(
                [
                    current_platform.device_control_id_to_physical_device_id(gpu)
                    for gpu in args.gpus.split(",")
                ]
            )

        cpu_group = dist.new_group(backend="gloo")
        # This explicit benchmark-only override tests the IPC kernels on a
        # PCIe mesh. The regular P2P read/write check still runs on every rank.
        topology = (
            patch.object(current_platform, "is_fully_connected", return_value=True)
            if args.allow_pcie_mesh
            else nullcontext()
        )
        config_context = (
            set_current_vllm_config(
                VllmConfig(
                    additional_config={
                        "cmp_pcie_allreduce_max_bytes": args.pcie_max_bytes,
                        "cmp_pcie_allreduce_devices": args.gpus.split(","),
                    }
                )
            )
            if args.pcie_max_bytes
            else nullcontext()
        )
        with topology, config_context:
            custom = CustomAllreduce(cpu_group, device_rank)
        if custom.disabled:
            raise RuntimeError("Custom collectives are unavailable for this group")

    def reduce(x):
        result = custom.custom_all_reduce(x) if custom is not None else None
        if result is not None:
            return result
        dist.all_reduce(x)
        return x

    def capture():
        return custom.capture() if custom is not None else nullcontext()

    records = []
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for rows in args.rows:
                generator = torch.Generator(device="cuda").manual_seed(170)
                pattern = torch.randint(
                    0, 17, (rows, 5120), device="cuda", generator=generator
                ).to(dtype)
                x = pattern + rank
                expected = pattern * world + world * (world - 1) / 2
                result = reduce(x)
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                x.zero_()
                for _ in range(5):
                    reduce(x)
                torch.accelerator.synchronize()
                one = torch.cuda.CUDAGraph()
                with capture(), torch.cuda.graph(one):
                    result = reduce(x)
                x.copy_(pattern + rank)
                one.replay()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                del one
                x.zero_()
                graph = torch.cuda.CUDAGraph()
                with capture(), torch.cuda.graph(graph):
                    for _ in range(args.calls):
                        result = reduce(x)
                for _ in range(3):
                    graph.replay()
                torch.accelerator.synchronize()
                dist.barrier()
                samples = []
                begin = time.perf_counter()
                for _ in range(args.samples):
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) * 1000 / args.calls)
                wall = (time.perf_counter() - begin) * 1e6 / (args.calls * args.samples)
                torch.testing.assert_close(result, torch.zeros_like(x), rtol=0, atol=0)
                samples.sort()
                row = {
                    "rows": rows,
                    "dtype": str(dtype),
                    "bytes": x.nbytes,
                    "p50_us": samples[len(samples) // 2],
                    "p95_us": samples[min(len(samples) - 1, int(len(samples) * 0.95))],
                    "wall_us_per_call": wall,
                    "rank": rank,
                    "correct_eager_and_graph": True,
                    "used_custom": custom is not None and custom.should_custom_ar(x),
                }
                gathered = [None] * world
                dist.all_gather_object(gathered, row)
                if rank == 0:
                    record = {
                        "rows": rows,
                        "dtype": str(dtype),
                        "bytes": x.nbytes,
                        "slowest_rank_p50_us": max(r["p50_us"] for r in gathered),
                        "ranks": gathered,
                    }
                    records.append(record)
                    print(json.dumps(record), flush=True)
                del graph
        if rank == 0:
            result = {
                "gpus": args.gpus,
                "visible_gpus": args.visible_gpus or args.gpus,
                "backend": args.backend,
                "allow_pcie_mesh": args.allow_pcie_mesh,
                "pcie_max_bytes": args.pcie_max_bytes,
                "custom_allreduce_algorithm": os.environ.get(
                    "VLLM_CUSTOM_ALLREDUCE_ALGO", "auto"
                ),
                "torch": torch.__version__,
                "calls_per_graph": args.calls,
                "samples": args.samples,
                "nccl_version": torch.cuda.nccl.version(),
                "nccl_env": {
                    k: v for k, v in os.environ.items() if k.startswith("NCCL_")
                },
                "device": str(torch.cuda.get_device_properties(device_rank)),
                "results": records,
            }
            Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    finally:
        torch.accelerator.synchronize()
        if custom is not None:
            custom.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--visible-gpus")
    parser.add_argument("--backend", choices=["nccl", "custom"], default="nccl")
    parser.add_argument("--allow-pcie-mesh", action="store_true")
    parser.add_argument("--pcie-max-bytes", type=int, default=0)
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 32, 128, 1024])
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus or args.gpus
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(
        worker,
        args=(len(args.gpus.split(",")), port, args),
        nprocs=len(args.gpus.split(",")),
        join=True,
    )
