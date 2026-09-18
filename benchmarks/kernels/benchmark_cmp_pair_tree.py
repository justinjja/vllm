# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference checks and CUDA-graph timing for the CMP four-rank pair tree.

Compile cmp_pair_tree_reduce.cu as a standalone SM80 shared library and pass
its path with --library. Both selected PIX pairs must support CUDA IPC, and
their leaders must support peer writes. This experiment does not change the
serving communicator.
"""

import argparse
import ctypes
import json
import os
import socket
import statistics
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class PairTree:
    def __init__(self, rank, library, capacity=48 * 6144, group=None):
        from vllm import _custom_ops as ops

        self.ops = ops
        self.rank = rank
        group = group if group is not None else dist.group.WORLD
        members = dist.get_process_group_ranks(group)
        assert len(members) == 4 and dist.get_rank(group) == rank
        self.capacity = capacity
        self.lib = ctypes.CDLL(str(Path(library).resolve()))
        self.lib.pair_tree_zero.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.pair_tree_launch.argtypes = (
            [ctypes.c_void_p] * 11 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        )
        self.pointers = {}
        self.owned = []
        for leader in (0, 2):
            if rank == leader:
                pointer, handle = ops.allocate_shared_buffer_and_handle(capacity * 12)
                assert self.lib.pair_tree_zero(pointer, capacity * 12) == 0
                torch.accelerator.synchronize()
                self.owned.append(pointer)
            else:
                pointer, handle = None, None
            handles = [handle]
            dist.broadcast_object_list(handles, src=members[leader], group=group)
            if rank != leader and (rank % 2 == 0 or rank // 2 == leader // 2):
                pointer = ops.open_mem_handle(handles[0])
            self.pointers[leader] = pointer
        signal, handle = ops.allocate_shared_buffer_and_handle(32 * 192 * 4)
        assert self.lib.pair_tree_zero(signal, 32 * 192 * 4) == 0
        torch.accelerator.synchronize()
        self.owned.append(signal)
        handles = [None] * 4
        dist.all_gather_object(handles, handle, group=group)
        self.signals = {rank: signal, rank ^ 1: ops.open_mem_handle(handles[rank ^ 1])}
        if rank % 2 == 0:
            self.signals[rank ^ 2] = ops.open_mem_handle(handles[rank ^ 2])
        result, handle = ops.allocate_shared_buffer_and_handle(capacity * 2)
        self.owned.append(result)
        handles = [None] * 4
        dist.all_gather_object(handles, handle, group=group)
        self.results = {rank: result, rank ^ 1: ops.open_mem_handle(handles[rank ^ 1])}
        self.counters = torch.zeros(32, device="cuda", dtype=torch.int32)
        self.errors = torch.zeros_like(self.counters)
        dist.barrier(group=group)

    def run(self, value, output, blocks):
        code = self.lib.pair_tree_launch(
            value.data_ptr(),
            output.data_ptr(),
            self.pointers[(self.rank // 2) * 2],
            self.pointers[2 - (self.rank // 2) * 2] if self.rank % 2 == 0 else None,
            self.signals[self.rank],
            self.signals[self.rank ^ 1],
            self.signals[self.rank ^ 2] if self.rank % 2 == 0 else None,
            self.results[self.rank],
            self.results[self.rank ^ 1],
            self.counters.data_ptr(),
            self.errors.data_ptr(),
            value.numel(),
            self.capacity,
            self.rank % 2,
            blocks,
            torch.cuda.current_stream().cuda_stream,
        )
        assert code == 0, code
        return output

    def close(self):
        for pointer in self.owned:
            self.ops.free_shared_buffer(pointer)


def worker(rank, port, args):
    import faulthandler

    faulthandler.dump_traceback_later(120, repeat=True)
    torch.set_num_threads(1)
    device_index = [int(value) for value in args.gpus.split(",")][rank]
    torch.accelerator.set_device_index(device_index)
    torch.manual_seed(1289 + rank)
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=180),
    )
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    nccl = PyNcclCommunicator(dist.group.WORLD, device=device_index)
    candidate = PairTree(rank, args.library)
    records = []
    for rows in (1, 3, 6, 12, 24, 48):
        inputs = [
            torch.randn(rows, 6144, device="cuda", dtype=torch.bfloat16)
            for _ in range(16)
        ]
        outputs = [torch.empty_like(value) for value in inputs]

        def reference(inputs=inputs):
            expected = []
            for value in inputs:
                local = value.float().cpu()
                peers = [torch.empty_like(local) for _ in range(4)]
                dist.all_gather(peers, local)
                expected.append(
                    ((peers[0] + peers[1]) + (peers[2] + peers[3]))
                    .to(torch.bfloat16)
                    .cuda()
                )
            return expected

        expected = reference()
        for blocks in (2, 4, 8, 16):

            def run(skew=False, inputs=inputs, outputs=outputs, blocks=blocks):
                for index, (value, output) in enumerate(zip(inputs, outputs)):
                    if skew and index % 4 == rank:
                        torch.cuda._sleep(30000)
                    candidate.run(value, output, blocks)

            def check(targets, outputs=outputs):
                assert not candidate.errors.any(), candidate.errors
                for output, target in zip(outputs, targets):
                    torch.testing.assert_close(output, target, rtol=0, atol=0)

            run(True)
            torch.accelerator.synchronize()
            check(expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()
            torch.accelerator.synchronize()
            check(expected)
            # Vary every rank's inputs between replays to expose stale buffers.
            for value in inputs:
                value.mul_(-1).add_(rank * 0.0625)
            expected = reference()
            for _ in range(3):
                graph.replay()
            torch.accelerator.synchronize()
            check(expected)
            samples = []
            for _ in range(7):
                dist.barrier()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / len(inputs))
            check(expected)
            gathered = [None] * 4
            dist.all_gather_object(gathered, samples)
            if rank == 0:
                row = {
                    "method": "pair_tree_fp32",
                    "rows": rows,
                    "blocks": blocks,
                    "max_rank_median_us": max(statistics.median(x) for x in gathered),
                    "rank_samples_us": gathered,
                    "all_ranks_exact_reference": True,
                }
                records.append(row)
                print(
                    {k: v for k, v in row.items() if k != "rank_samples_us"}, flush=True
                )
                Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
            graph.reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for value, output in zip(inputs, outputs):
                nccl.all_reduce(value, output)
        graph.replay()
        torch.accelerator.synchronize()
        for output, target in zip(outputs, expected):
            torch.testing.assert_close(
                output.float(), target.float(), rtol=0.02, atol=0.03
            )
        samples = []
        for _ in range(7):
            dist.barrier()
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / len(inputs))
        gathered = [None] * 4
        dist.all_gather_object(gathered, samples)
        if rank == 0:
            row = {
                "method": "nccl",
                "rows": rows,
                "max_rank_median_us": max(statistics.median(x) for x in gathered),
                "rank_samples_us": gathered,
            }
            records.append(row)
            print({k: v for k, v in row.items() if k != "rank_samples_us"}, flush=True)
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
        graph.reset()
    dist.barrier()
    candidate.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    selected = [int(value) for value in args.gpus.split(",")]
    if len(selected) != 4 or len(set(selected)) != 4:
        parser.error("--gpus must select four distinct visible device ordinals")
    args.library = args.library.resolve()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, args), nprocs=4, join=True)
