# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare sparse MLA launch shapes, checking each against FP32 attention."""

import argparse
import importlib.util
import json
import random
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.kernels.attention.dsa import ampere_mla_flashinfer
from vllm.model_executor.kernels.attention.dsa.ampere_sparse_mla import sparse_mla
from vllm.triton_utils import triton


def cache_and_reference(slots, page=32):
    values = torch.randn(slots, 512, device="cuda", dtype=torch.bfloat16) * 0.2
    quant = values[:, :448].to(torch.float8_e4m3fn)
    reference = torch.cat((quant.float(), values[:, 448:].float()), -1)
    cache = torch.empty(slots // page, page, 584, device="cuda", dtype=torch.uint8)
    flat = cache.view(slots // page, -1)
    data = flat[:, : page * 576].view(-1, page, 576)
    data[:, :, :448] = quant.view(torch.uint8).view(-1, page, 448)
    data[:, :, 448:] = (
        values[:, 448:].contiguous().view(torch.uint8).view(-1, page, 128)
    )
    flat[:, page * 576 :] = 127
    return cache, reference


def main(args):
    torch.accelerator.set_device_index(args.device)
    torch.manual_seed(170)
    torch.set_float32_matmul_precision("highest")
    baseline_module = None
    if args.baseline_source:
        spec = importlib.util.spec_from_file_location(
            "cmp_sparse_mla_baseline", args.baseline_source
        )
        assert spec is not None and spec.loader is not None
        baseline_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_module
        spec.loader.exec_module(baseline_module)
    cache, values = cache_and_reference(args.slots)
    if args.mode == "dense":
        cache = values.to(torch.bfloat16)
    records = []
    for rows in args.rows:
        for heads in args.heads:
            q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16)
            indices = torch.randint(
                0, values.shape[0], (rows, 640), device="cuda", dtype=torch.int32
            )
            lengths = torch.full((rows,), 640, device="cuda", dtype=torch.int32)
            sink = torch.randn(heads, device="cuda")
            out = torch.empty_like(q)
            expected = []
            for row in (0, rows - 1):
                selected = indices[row]
                if args.mode == "split":
                    selected = torch.cat(
                        (
                            selected[: args.main_length],
                            selected[128 : 128 + args.extra_length],
                        )
                    )
                keys = values[selected.long()]
                logits = q[row].float() @ keys.T / 512**0.5
                probs = torch.cat((logits, sink[:, None]), -1).softmax(-1)
                expected.append(probs[:, :-1] @ keys)
            variants = [(32, None, 4, 16)] + [
                (n, s or None, w, h)
                for n in args.blocks
                for s in args.splits
                for w in args.warps
                for h in args.head_blocks
            ]
            main_ids, main_lens = indices, lengths
            extra_cache = extra_ids = extra_lens = None
            if args.mode == "split":
                main_ids, extra_ids = indices[:, :128], indices[:, 128:]
                main_lens = torch.full_like(lengths, args.main_length)
                extra_lens = torch.full_like(lengths, args.extra_length)
                extra_cache = cache
            calls = []
            for block_n, splits, warps, block_h in variants:
                run = partial(
                    sparse_mla,
                    q,
                    cache,
                    main_ids,
                    main_lens,
                    512**-0.5,
                    sink,
                    out,
                    extra_cache,
                    extra_ids,
                    extra_lens,
                    splits=splits,
                    block_n=block_n,
                    num_warps=warps,
                    block_h=block_h,
                    trim_empty_tiles=args.trim_empty_tiles,
                )

                calls.append(run)
            if baseline_module is not None:
                calls.append(
                    partial(
                        baseline_module.sparse_mla,
                        q,
                        cache,
                        main_ids,
                        main_lens,
                        512**-0.5,
                        sink,
                        out,
                        extra_cache,
                        extra_ids,
                        extra_lens,
                    )
                )
                variants.append((0, None, None, None))
            if args.flashinfer:
                workspace = ampere_mla_flashinfer.get_sparse_mla_workspace(
                    max(args.rows), q.device
                )
                calls.append(
                    partial(
                        workspace.run,
                        q,
                        cache,
                        main_ids,
                        main_lens,
                        extra_cache,
                        extra_ids,
                        extra_lens,
                        sink,
                        512**-0.5,
                        out,
                    )
                )
                variants.append((None, None, None, None))
            first_calls = []
            supported_calls = []
            supported_variants = []
            skipped = []
            for variant, run in zip(variants, calls):
                torch.accelerator.synchronize()
                begin = time.perf_counter()
                try:
                    run()
                except triton.runtime.errors.OutOfResources as error:
                    skipped.append({"variant": variant, "error": str(error)})
                    continue
                torch.accelerator.synchronize()
                first_calls.append(time.perf_counter() - begin)
                for row, ref in zip((0, rows - 1), expected):
                    torch.testing.assert_close(
                        out[row].float(), ref, rtol=0.02, atol=0.002
                    )
                triton.testing.do_bench_cudagraph(run, rep=100)
                supported_calls.append(run)
                supported_variants.append(variant)
            assert supported_variants[0] == variants[0], "Baseline must run"
            calls, variants = supported_calls, supported_variants
            samples = [[] for _ in calls]
            order = list(range(len(calls)))
            rng = random.Random(170)
            for _ in range(args.repeats):
                rng.shuffle(order)
                for i in order:
                    samples[i].append(
                        triton.testing.do_bench_cudagraph(calls[i], rep=100) * 1000
                    )
            trials = [
                {
                    "block_n": n,
                    "backend": "baseline"
                    if n == 0
                    else ("flashinfer" if n is None else "triton"),
                    "splits": s,
                    "warps": w,
                    "block_h": h,
                    "us": statistics.median(times),
                    "samples_us": times,
                    "first_call_s": first_call,
                }
                for (n, s, w, h), times, first_call in zip(
                    variants, samples, first_calls
                )
            ]
            baseline = trials[0]["us"]
            best = min(trials, key=lambda t: t["us"])
            record = {
                "physical_gpu": args.device,
                "rows": rows,
                "heads": heads,
                "keys": 640,
                "slots": args.slots,
                "mode": args.mode,
                "main_length": args.main_length if args.mode == "split" else 640,
                "extra_length": args.extra_length if args.mode == "split" else 0,
                "trim_empty_tiles": args.trim_empty_tiles,
                "baseline_us": baseline,
                "best": best,
                "speedup": baseline / best["us"],
                "trials": trials,
                "skipped": skipped,
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 6, 16, 64, 256])
    parser.add_argument("--heads", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--blocks", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--splits", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument("--warps", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--head-blocks", type=int, nargs="+", default=[16])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--mode", choices=["packed", "split", "dense"], default="packed"
    )
    parser.add_argument("--flashinfer", action="store_true")
    parser.add_argument("--main-length", type=int, default=128)
    parser.add_argument("--extra-length", type=int, default=512)
    parser.add_argument("--trim-empty-tiles", action="store_true")
    parser.add_argument("--baseline-source", type=Path)
    parser.add_argument("--slots", type=int, default=16384)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 0 <= args.main_length <= 128 or not 0 <= args.extra_length <= 512:
        parser.error("split cache lengths must fit their 128/512 entry widths")
    main(args)
