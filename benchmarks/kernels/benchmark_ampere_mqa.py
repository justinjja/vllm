# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sweep query/KV tile sizes for CMP decode, speculation, and prefill."""

import argparse
import json
import statistics
from pathlib import Path

import torch

from vllm.model_executor.kernels.attention.dsa.ampere_mqa import (
    mqa_logits,
    paged_mqa_logits,
)
from vllm.triton_utils import triton


def benchmark_predecode(rows, heads, n):
    """Include conversion cost and bound scores to the serving workspace size."""
    q = torch.randn(rows, heads, 128, device="cuda", dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    k = torch.randn(n, 128, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    scales = torch.rand(n, device="cuda")
    weights = torch.randn(rows, heads, device="cuda")
    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
    ends = torch.arange(n - rows + 1, n + 1, device="cuda", dtype=torch.int32)
    args = ((q, None), (k, scales), weights, starts, ends)
    reference = mqa_logits(*args)
    output = torch.empty_like(reference)
    workspace = (
        torch.empty_like(q, dtype=torch.float16),
        torch.empty_like(k, dtype=torch.float16),
    )

    def run(predecode):
        return mqa_logits(
            *args, out=output, decode_workspace=workspace if predecode else None
        )

    run(True)
    torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-4)
    unequal = (output != reference).sum().item()
    max_abs = torch.nan_to_num((output - reference).abs(), nan=0.0).max().item()
    ref_topk = reference.topk(2048, dim=1).indices.sort(dim=1).values
    actual_topk = output.topk(2048, dim=1).indices.sort(dim=1).values
    changed_rows = (ref_topk != actual_topk).any(dim=1).sum().item()
    samples = [[], []]
    for choice in (False, True):
        triton.testing.do_bench_cudagraph(lambda choice=choice: run(choice), rep=200)
    for repeat in range(4):
        for choice in (0, 1) if repeat % 2 == 0 else (1, 0):
            samples[choice].append(
                triton.testing.do_bench_cudagraph(
                    lambda choice=choice: run(choice), rep=240
                )
            )
    baseline, predecoded = [statistics.median(values) for values in samples]
    result = {
        "rows": rows,
        "heads": heads,
        "kv_len": n,
        "baseline_ms": baseline,
        "predecoded_ms": predecoded,
        "speedup": baseline / predecoded,
        "samples_ms": samples,
        "conversion_included": True,
        "extra_workspace_bytes": sum(x.numel() * x.element_size() for x in workspace),
        "unequal_logits": unequal,
        "max_abs_difference": max_abs,
        "topk_rows_with_changed_membership": changed_rows,
    }
    print(json.dumps(result), flush=True)
    return result


def benchmark(mode, batch, next_n, heads, n, fp16=False, verify_dispatch=False):
    m = batch * next_n
    q = torch.randn(m, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    weights = torch.randn(m, heads, device="cuda")
    if mode == "dense":
        k = torch.randn(n, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(n, device="cuda")
        starts = torch.zeros(m, device="cuda", dtype=torch.int32)
        ends = torch.arange(n - m + 1, n + 1, device="cuda", dtype=torch.int32)

        def run(tile, fp16=fp16):
            kwargs = {"tile": tile}
            if fp16 is not None:
                kwargs["fp16"] = fp16
            return mqa_logits((q, None), (k, scales), weights, starts, ends, **kwargs)
    else:
        page = 64
        blocks = triton.cdiv(n, page)
        k = torch.randn(batch * blocks, page, 128, device="cuda").to(
            torch.float8_e4m3fn
        )
        scales = torch.rand(batch * blocks, page, device="cuda")
        cache = torch.empty(
            batch * blocks, page, 1, 132, device="cuda", dtype=torch.uint8
        )
        flat = cache.view(batch * blocks, -1)
        flat[:, : page * 128] = k.view(torch.uint8).reshape(batch * blocks, -1)
        flat[:, page * 128 :] = scales.view(torch.uint8).reshape(batch * blocks, -1)
        table = torch.randperm(batch * blocks, device="cuda", dtype=torch.int32).view(
            batch, blocks
        )
        lens = torch.arange(n - next_n + 1, n + 1, device="cuda", dtype=torch.int32)
        lens = lens[None, :].expand(batch, -1).contiguous()
        q = q.view(batch, next_n, heads, 128)

        def run(tile, fp16=fp16):
            kwargs = {"tile": tile}
            if fp16 is not None:
                kwargs["fp16"] = fp16
            return paged_mqa_logits(
                (q, None), cache, weights, lens, table, None, n, **kwargs
            )

    baseline = run((1, 64, 4), fp16=False)
    if verify_dispatch:
        torch.testing.assert_close(run(None, None), baseline, rtol=2e-5, atol=3e-4)
        fns = [lambda: run((1, 64, 4), False), lambda: run(None, None)]
        # Compile and warm both paths before balanced alternating measurements.
        for fn in fns:
            triton.testing.do_bench_cudagraph(fn, rep=200)
        samples = [[], []]
        for repeat in range(6):
            for choice in [0, 1] if repeat % 2 == 0 else [1, 0]:
                samples[choice].append(
                    triton.testing.do_bench_cudagraph(fns[choice], rep=100) * 1000
                )
        base = statistics.median(samples[0])
        selected = statistics.median(samples[1])
        result = {
            "mode": mode,
            "batch": batch,
            "next_n": next_n,
            "heads": heads,
            "kv_len": n,
            "baseline_bf16_row1_us": base,
            "selected_dispatch_us": selected,
            "speedup": base / selected,
            "baseline_samples_us": samples[0],
            "selected_samples_us": samples[1],
        }
        print(json.dumps(result), flush=True)
        return result
    base_us = (
        triton.testing.do_bench_cudagraph(lambda: run((1, 64, 4), fp16=False), rep=40)
        * 1000
    )
    results = []
    max_rows = m if mode == "dense" else next_n
    for br in (1, 2, 4, 8):
        if br > triton.next_power_of_2(max_rows):
            continue
        for bn in (32, 64, 128):
            tile = (br, bn, 4)
            candidate = run(tile)
            torch.testing.assert_close(candidate, baseline, rtol=2e-5, atol=3e-4)
            del candidate
            us = (
                triton.testing.do_bench_cudagraph(lambda tile=tile: run(tile), rep=40)
                * 1000
            )
            results.append({"tile": tile, "us": us})
    best = min(results, key=lambda r: r["us"])
    result = {
        "mode": mode,
        "mma_dtype": "fp16" if fp16 else "bf16",
        "batch": batch,
        "next_n": next_n,
        "heads": heads,
        "kv_len": n,
        "baseline_us": base_us,
        "best": best,
        "speedup": base_us / best["us"],
        "sweep": results,
    }
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--heads", nargs="+", type=int, default=[16, 32])
    parser.add_argument("--contexts", nargs="+", type=int, default=[4096, 32768])
    parser.add_argument("--mode", choices=["dense", "paged", "all"], default="all")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--verify-dispatch", action="store_true")
    parser.add_argument("--predecode", action="store_true")
    args = parser.parse_args()
    if args.predecode and args.mode != "dense":
        parser.error("--predecode requires --mode dense")
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    records = []
    workloads = []
    if args.mode in ("paged", "all"):
        workloads += [("paged", b, s) for b, s in [(1, 1), (1, 6), (4, 6), (16, 1)]]
    if args.mode in ("dense", "all"):
        workloads += [("dense", m, 1) for m in [128, 1024]]
    for heads in args.heads:
        for n in args.contexts:
            cases = (
                [("dense", min(1024, (512 * 1024**2) // (4 * n)), 1)]
                if args.predecode
                else workloads
            )
            for mode, b, s in cases:
                records.append(
                    benchmark_predecode(b, heads, n)
                    if args.predecode
                    else benchmark(
                        mode, b, s, heads, n, args.fp16, args.verify_dispatch
                    )
                )
                Path(args.output).write_text(
                    json.dumps(
                        {
                            "device": str(torch.cuda.get_device_properties(0)),
                            "torch": torch.__version__,
                            "triton": triton.__version__,
                            "results": records,
                        },
                        indent=2,
                    )
                    + "\n"
                )
