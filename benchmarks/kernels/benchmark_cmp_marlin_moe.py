# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed MXFP4 MoE launch tuning with the V4.1 expert dimensions."""

import argparse
import json
import random
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import ApplyMoEActivationConfig
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import _fused_marlin_moe
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
)
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton


def weights(experts, n, k):
    # Distinct patterns catch routing errors without retaining all dequantized
    # experts. Every packed expert has its own physical allocation in the bank.
    references, packed, scales = [], [], []
    shape = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
    for _ in range(8):
        ref, quant, scale = rand_marlin_weight_mxfp4_like(shape, 32)
        references.append(ref)
        packed.append(quant)
        scales.append(scale)
    ids = torch.arange(experts, device="cuda") % len(packed)
    bank = torch.stack(packed)[ids].contiguous()
    scale_bank = torch.stack(scales).view(torch.uint8)[ids].contiguous()
    return references, bank, scale_bank.view(torch.float8_e8m0fnu)


def reference(x, w1, w2, ids, router):
    result = torch.zeros_like(x, dtype=torch.float32)
    # Evaluate the same quantized weights with independent torch GEMMs.
    for pattern, (up, down) in enumerate(zip(w1, w2)):
        projected = (x @ up).float()
        gate, linear = projected.chunk(2, -1)
        activated = (
            torch.nn.functional.silu(gate.clamp(max=10)) * linear.clamp(-10, 10)
        ).to(x.dtype)
        value = (activated @ down).float()
        for route in range(ids.shape[1]):
            factor = router[:, route] * (ids[:, route] % len(w1) == pattern)
            result += (value * factor[:, None]).to(x.dtype).float()
    return result.to(x.dtype)


def run_moe(
    x, w1, w2, s1, s2, router, ids, workspace, cache13, cache2, output, block_m
):
    rows, k = x.shape
    experts, topk = w1.shape[0], ids.shape[1]
    sorted_ids, expert_ids, padded = moe_align_block_size(
        ids, block_m, experts, ignore_invalid_experts=True
    )
    result = _fused_marlin_moe(
        x,
        w1,
        w2,
        None,
        None,
        s1,
        s2,
        router,
        topk,
        scalar_types.float4_e2m1f,
        False,
        None,
        block_m,
        sorted_ids,
        expert_ids,
        padded,
        topk_ids=ids,
        workspace=workspace,
        intermediate_cache13=cache13,
        intermediate_cache2=cache2,
        activation_config=ApplyMoEActivationConfig(clamp_limit=10),
    ).view(rows, topk, k)
    ops.moe_sum(result, output)


def main(args):
    torch.manual_seed(170)
    torch.set_num_threads(1)
    records = []
    for tp in args.tp:
        experts, k, n, topk = 384, 5120, 2304 // tp, 6
        r1, w1, s1 = weights(experts, 2 * n, k)
        r2, w2, s2 = weights(experts, k, n)
        workspace = marlin_make_workspace_new(torch.device("cuda"), 4)
        for rows in args.rows:
            x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
            score = torch.randn(rows, experts, device="cuda")
            _, ids = score.topk(topk, -1)
            ids = ids.to(torch.int32)
            router = torch.randn(rows, topk, device="cuda").softmax(-1)
            expected = reference(x, r1, r2, ids, router)
            cache13 = torch.empty(
                rows * topk * max(2 * n, k), device="cuda", dtype=x.dtype
            )
            cache2 = torch.empty(rows * topk, n, device="cuda", dtype=x.dtype)
            output = torch.empty_like(x)
            variants, calls = [], []
            for block_m in (8, 16, 32, 48, 64):
                run = partial(
                    run_moe,
                    x,
                    w1,
                    w2,
                    s1,
                    s2,
                    router,
                    ids,
                    workspace,
                    cache13,
                    cache2,
                    output,
                    block_m,
                )

                run()
                error = (
                    (output.float() - expected.float()).abs().mean()
                    / expected.float().abs().mean()
                ).item()
                assert torch.isfinite(output).all() and error < 0.01, error
                triton.testing.do_bench_cudagraph(run, rep=100)
                variants.append({"block_m": block_m, "relative_mean_abs_error": error})
                calls.append(run)
            samples = [[] for _ in calls]
            rng = random.Random(170)
            for _ in range(3):
                order = list(range(len(calls)))
                rng.shuffle(order)
                for i in order:
                    samples[i].append(
                        triton.testing.do_bench_cudagraph(calls[i], rep=100) * 1000
                    )
            for variant, times in zip(variants, samples):
                variant.update(us=statistics.median(times), samples_us=times)
            default = next(
                (b for b in (8, 16, 32, 48, 64) if rows * topk / experts / b < 0.9), 64
            )
            baseline = next(v for v in variants if v["block_m"] == default)
            best = min(variants, key=lambda v: v["us"])
            record = {
                "tp": tp,
                "rows": rows,
                "experts": experts,
                "topk": topk,
                "weight_bytes": sum(t.nbytes for t in (w1, w2, s1, s2)),
                "baseline": baseline,
                "best": best,
                "speedup": baseline["us"] / best["us"],
                "trials": variants,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
        del r1, r2, w1, w2, s1, s2, workspace, calls, run
        torch.accelerator.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 4, 12, 32, 128, 512])
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
