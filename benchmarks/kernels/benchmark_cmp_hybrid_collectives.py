# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare flat and two-stage reductions for the CMP TP2/EP4 layout."""

import argparse
import json
import socket
import statistics
from pathlib import Path

import torch
import torch.multiprocessing as mp


def worker(rank, port, output):
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_ep_group,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.parallel_state import init_model_parallel_group

    torch.set_num_threads(1)
    physical = [0, 1, 4, 5, 6, 7, 2, 3][rank]
    torch.accelerator.set_device_index(physical)
    device = torch.device("cuda", physical)
    config = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=8, enable_expert_parallel=True
        ),
        additional_config={"cmp_tp2_ep4": True},
    )
    with set_current_vllm_config(config):
        init_distributed_environment(
            world_size=8,
            rank=rank,
            local_rank=physical,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(tensor_model_parallel_size=8)
        pair, full = get_tp_group(), get_ep_group()
        replica = init_model_parallel_group(
            [[0, 2, 4, 6], [1, 3, 5, 7]],
            physical,
            "nccl",
            group_name="cmp_replica",
        )

        def split_elements(x):
            shard = pair.reduce_scatter(x.flatten(), dim=0)
            reduced = replica.all_reduce(shard)
            return pair.all_gather(reduced, dim=0).view_as(x)

        methods = {
            "flat": full.all_reduce,
            "pair_first": lambda x: replica.all_reduce(pair.all_reduce(x)),
            "replica_first": lambda x: pair.all_reduce(replica.all_reduce(x)),
            "split_elements": split_elements,
        }
        results = []
        for rows in (1, 6, 32, 256, 1020):
            pattern = (
                torch.arange(rows * 5120, device=device).reshape(rows, 5120) % 31
            ).float()
            data = pattern + rank
            expected = pattern * 8 + 28
            for name, reduce in methods.items():
                for _ in range(3):
                    torch.testing.assert_close(reduce(data), expected, rtol=0, atol=0)
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with (
                    pair.graph_capture() as capture,
                    torch.cuda.graph(graph, stream=capture.stream),
                ):
                    result = reduce(data)
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(result, expected, rtol=0, atol=0)
                data.add_(0.5)
                graph.replay()
                torch.accelerator.synchronize()
                torch.testing.assert_close(result, expected + 4, rtol=0, atol=0)
                data.sub_(0.5)
                times = []
                for _ in range(10):
                    torch.distributed.barrier(device_ids=[physical])
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(32):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    times.append(start.elapsed_time(end) * 1000 / 32)
                record = {
                    "rows": rows,
                    "method": name,
                    "median_us": statistics.median(times),
                    "samples_us": times,
                }
                results.append(record)
                if rank == 0:
                    print(json.dumps(record), flush=True)
                graph.reset()
        output.mkdir(exist_ok=True, parents=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(results, indent=2) + "\n")
        torch.accelerator.synchronize()
        torch.distributed.barrier(device_ids=[physical])
        replica.destroy()
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args.output), nprocs=8, join=True)
