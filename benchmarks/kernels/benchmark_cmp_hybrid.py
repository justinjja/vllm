# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TP2/EP4 native expert and collective checks against dequantized Torch GEMMs."""

import argparse
import json
import os
import socket
import statistics
from pathlib import Path

import torch
import torch.multiprocessing as mp


def worker(rank, port, args):
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_ep_group,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        initialize_model_parallel,
    )
    from vllm.distributed.cmp_hybrid import (
        combine_replicated_experts,
        hybrid_expert_all_reduce,
    )
    from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
    from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        determine_expert_map,
    )
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_workspace_new,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        rand_marlin_weight_mxfp4_like,
    )
    from vllm.models.deepseek_v41.nvidia.engram import ParallelEngramEmbedding
    from vllm.scalar_type import scalar_types

    torch.set_num_threads(1)
    physical = [0, 1, 4, 5, 6, 7, 2, 3][rank]
    torch.accelerator.set_device_index(physical)
    device = torch.device("cuda", physical)
    configuration = VllmConfig(
        parallel_config=ParallelConfig(
            tensor_parallel_size=8, enable_expert_parallel=True
        ),
        additional_config={
            "cmp_tp2_ep4": True,
            "cmp_hybrid_pair_reduce": args.pair_reduce,
        },
    )
    reports = []
    with set_current_vllm_config(configuration):
        init_distributed_environment(
            world_size=8,
            rank=rank,
            local_rank=physical,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(tensor_model_parallel_size=8)
        pair = get_tp_group()
        world = get_ep_group()
        assert pair.world_size == 2 and world.world_size == 8
        assert pair.ranks == [rank // 2 * 2, rank // 2 * 2 + 1]
        moe_parallel = FusedMoEParallelConfig.make(
            tp_size_=2,
            dp_size_=1,
            pcp_size_=1,
            sp_size_=1,
            vllm_parallel_config=configuration.parallel_config,
        )
        assert (moe_parallel.tp_rank, moe_parallel.ep_rank) == (rank % 2, rank // 2)
        local_count, mapping, _ = determine_expert_map(4, rank // 2, 384)
        assert local_count == 96
        mapping = mapping.to(device)
        with torch.device(device):
            embedding = ParallelEngramEmbedding(
                24 * 17, 64, (17,) * 24, cpu_offload=True
            )
        assert embedding.tp_size == 8
        assert embedding.part_num_embeddings == 3 * 17
        assert embedding.head_start == rank * 3
        table = (torch.arange(24 * 17 * 64).reshape(-1, 64) % 13).to(
            torch.float8_e4m3fn
        )
        scales = torch.full((24 * 17, 2), 127, dtype=torch.uint8)
        embedding.weight.weight_loader(embedding.weight, table)
        embedding.weight_scale_inv.weight_loader(
            embedding.weight_scale_inv, scales.view(torch.float8_e8m0fnu)
        )
        hashes = (torch.arange(24)[None, :] * 17 + torch.arange(3)[:, None]).int()
        expected_embedding = table.float()[hashes.long()].to(torch.bfloat16)
        torch.testing.assert_close(
            embedding(hashes.to(device)).cpu(), expected_embedding, rtol=0, atol=0
        )
        del embedding
        refs_up, refs_down, packed_up, packed_down, scales_up, scales_down = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for pattern in range(7):
            torch.manual_seed(170 + pattern + 10 * (rank % 2))
            ref, packed, scale = rand_marlin_weight_mxfp4_like(
                torch.empty(2304, 5120, device=device, dtype=torch.bfloat16), 32
            )
            refs_up.append(ref)
            packed_up.append(packed)
            scales_up.append(scale)
            ref, packed, scale = rand_marlin_weight_mxfp4_like(
                torch.empty(5120, 1152, device=device, dtype=torch.bfloat16), 32
            )
            refs_down.append(ref)
            packed_down.append(packed)
            scales_down.append(scale)
        expert_ids = torch.arange(rank // 2 * 96, (rank // 2 + 1) * 96, device=device)
        patterns = expert_ids % 7
        w1 = torch.stack(packed_up)[patterns].contiguous()
        w2 = torch.stack(packed_down)[patterns].contiguous()
        s1 = (
            torch.stack(scales_up)
            .view(torch.uint8)[patterns]
            .contiguous()
            .view(torch.float8_e8m0fnu)
        )
        s2 = (
            torch.stack(scales_down)
            .view(torch.uint8)[patterns]
            .contiguous()
            .view(torch.float8_e8m0fnu)
        )
        workspace = marlin_make_workspace_new(device, 4)

        for rows in args.rows:
            for routing in ("spread", "pair0_only"):
                torch.manual_seed(2026 + rows)
                x = torch.randn(rows, 5120, device=device, dtype=torch.bfloat16)
                if routing == "spread":
                    ids = (
                        torch.randn(rows, 384, device=device).topk(6, -1).indices.int()
                    )
                else:
                    ids = (
                        torch.arange(6, device=device)
                        .expand(rows, -1)
                        .contiguous()
                        .int()
                    )
                weights = torch.randn(rows, 6, device=device).softmax(-1)
                # Distinct TP partials, replicated across the four expert owners.
                shared = x * (0.03125 * (rank % 2 + 1))

                def reference(x=x, ids=ids, weights=weights, shared=shared):
                    result = torch.zeros_like(x, dtype=torch.float32)
                    for pattern in range(7):
                        projected = (
                            (x.float() @ refs_up[pattern].float()).to(x.dtype).float()
                        )
                        gate, linear = projected.chunk(2, -1)
                        activated = (
                            torch.nn.functional.silu(gate.clamp(max=10))
                            * linear.clamp(-10, 10)
                        ).to(x.dtype)
                        value = (
                            (activated.float() @ refs_down[pattern].float())
                            .to(x.dtype)
                            .float()
                        )
                        for route in range(6):
                            own = (
                                (ids[:, route] >= rank // 2 * 96)
                                & (ids[:, route] < (rank // 2 + 1) * 96)
                                & (ids[:, route] % 7 == pattern)
                            )
                            result += (
                                (value * (weights[:, route] * own)[:, None])
                                .to(x.dtype)
                                .float()
                            )
                    local_reference = result.to(x.dtype)
                    return local_reference, world.all_reduce(
                        combine_replicated_experts(local_reference, shared, 4)
                    )

                def run(x=x, ids=ids, weights=weights, shared=shared):
                    routed = fused_marlin_moe(
                        x,
                        w1,
                        w2,
                        None,
                        None,
                        s1,
                        s2,
                        weights,
                        ids,
                        scalar_types.float4_e2m1f.id,
                        global_num_experts=384,
                        expert_map=mapping,
                        workspace=workspace,
                        activation_config=ApplyMoEActivationConfig(clamp_limit=10),
                    )
                    output = hybrid_expert_all_reduce(
                        combine_replicated_experts(routed, shared, 4), args.pair_reduce
                    )
                    return routed, output

                def check(actual, expected):
                    assert torch.isfinite(actual).all()
                    error = (actual - expected).abs()
                    relative = (
                        error.mean() / expected.abs().mean().clamp_min(1e-9)
                    ).item()
                    assert relative < 0.015, relative
                    assert error.max() < expected.abs().max() * 0.05 + 1e-7
                    return relative

                local_expected, expected = reference()
                routed, actual = run()
                local_error = check(routed.float(), local_expected.float())
                eager_error = check(actual, expected)
                if routing == "pair0_only" and rank // 2 != 0:
                    assert torch.count_nonzero(routed) == 0
                for _ in range(3):
                    run()
                torch.accelerator.synchronize()
                graph = torch.cuda.CUDAGraph()
                with (
                    graph_capture(device=device) as capture,
                    torch.cuda.graph(graph, stream=capture.stream),
                ):
                    routed_graph, actual_graph = run()
                graph.replay()
                torch.accelerator.synchronize()
                graph_error = check(actual_graph, expected)
                x.mul_(0.75)
                shared.copy_(x * (0.03125 * (rank % 2 + 1)))
                local_expected, expected = reference()
                graph.replay()
                torch.accelerator.synchronize()
                mutable_local_error = check(
                    routed_graph.float(), local_expected.float()
                )
                mutable_error = check(actual_graph, expected)
                if routing == "pair0_only" and rank // 2 != 0:
                    assert torch.count_nonzero(routed_graph) == 0
                timings = []
                for _ in range(5):
                    torch.distributed.barrier(device_ids=[physical])
                    begin, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    begin.record()
                    for _ in range(20):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    timings.append(begin.elapsed_time(end) * 1000 / 20)
                record = {
                    "rows": rows,
                    "routing": routing,
                    "local_route_count": int(
                        ((ids >= rank // 2 * 96) & (ids < (rank // 2 + 1) * 96))
                        .sum()
                        .item()
                    ),
                    "local_unique_experts": int(
                        ids[(ids >= rank // 2 * 96) & (ids < (rank // 2 + 1) * 96)]
                        .unique()
                        .numel()
                    ),
                    "eager_relative_error": eager_error,
                    "routed_relative_error": local_error,
                    "mutable_routed_relative_error": mutable_local_error,
                    "graph_relative_error": graph_error,
                    "mutable_relative_error": mutable_error,
                    "median_us": statistics.median(timings),
                }
                reports.append(record)
                if rank == 0:
                    print(json.dumps(record), flush=True)
                torch.accelerator.synchronize()
                graph.reset()
                if args.tune_gemm and routing == "spread":
                    from cmp_marlin_tuning import tune_projections

                    record["projection_tuning"] = tune_projections(
                        run, (ids >= rank // 2 * 96) & (ids < (rank // 2 + 1) * 96)
                    )
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "physical_gpu": physical,
                    "expert_parallel": 4,
                    "tensor_parallel": 2,
                    "small_pair_reduce": args.pair_reduce,
                    "local_experts": 96,
                    "engram_heads_sharded_over_eight": True,
                    "cases": reports,
                },
                indent=2,
            )
            + "\n"
        )
        torch.accelerator.synchronize()
        torch.distributed.barrier(device_ids=[physical])
        destroy_model_parallel()
        destroy_distributed_environment()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", nargs="+", type=int, default=[1, 6, 32, 256])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tune-gemm", action="store_true")
    parser.add_argument("--pair-reduce", action="store_true")
    arguments = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # Keep CUDA's physical enumeration intact; each worker chooses its own GPU.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(worker, args=(port, arguments), nprocs=8, join=True)
