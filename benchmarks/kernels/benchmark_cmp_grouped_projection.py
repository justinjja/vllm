# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen SM80 grouped WO_A kernels, including the required output layout."""

import argparse
import json
import statistics
from functools import partial
from pathlib import Path

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def grouped_dot(
    x,
    w,
    partial_out,
    T: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    group = tl.program_id(1)
    split = tl.program_id(2)
    m = tl.arange(0, BM)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = split * (K // SPLITS) + tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for _ in range(K // SPLITS // BK):
        a = tl.load(
            x + (m[:, None] * G + group) * K + k[None, :], mask=m[:, None] < T, other=0
        )
        b = tl.load(
            w + (group * N + n[None, :]) * K + k[:, None], mask=n[None, :] < N, other=0
        )
        acc = tl.dot(a, b, acc)
        k += BK
    idx = ((split * T + m[:, None]) * G + group) * N + n[None, :]
    tl.store(partial_out + idx, acc, mask=(m[:, None] < T) & (n[None, :] < N))


@triton.jit
def grouped_gemv(
    x,
    w,
    partial_out,
    T: tl.constexpr,
    G: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
):
    group = tl.program_id(1) // T
    row = tl.program_id(1) % T
    split = tl.program_id(2)
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    k = split * (K // SPLITS) + tl.arange(0, BK)
    acc = tl.full((BN, BK), 0, tl.float32)
    for _ in range(K // SPLITS // BK):
        a = tl.load(x + (row * G + group) * K + k)
        b = tl.load(
            w + (group * N + n[:, None]) * K + k[None, :], mask=n[:, None] < N, other=0
        )
        acc += a[None, :].to(tl.float32) * b.to(tl.float32)
        k += BK
    val = tl.sum(acc, 1)
    idx = ((split * T + row) * G + group) * N + n
    tl.store(partial_out + idx, val, mask=n < N)


@triton.jit
def finish_split(p, out, SIZE: tl.constexpr, SPLITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    val = tl.full((BLOCK,), 0, tl.float32)
    for split in range(SPLITS):
        val += tl.load(p + split * SIZE + i, mask=i < SIZE, other=0)
    tl.store(out + i, val, mask=i < SIZE)


def invoke(x, w, work, out, kind, bn, bk, splits):
    t, g, k = x.shape
    n = w.shape[1]
    if kind == "dot":
        grouped_dot[(triton.cdiv(n, bn), g, splits)](
            x,
            w,
            work,
            t,
            g,
            n,
            k,
            triton.next_power_of_2(max(16, t)),
            bn,
            bk,
            splits,
            num_warps=4,
        )
    else:
        grouped_gemv[(triton.cdiv(n, bn), g * t, splits)](
            x,
            w,
            work,
            t,
            g,
            n,
            k,
            bn,
            bk,
            splits,
            num_warps=4,
        )
    finish_split[(triton.cdiv(out.numel(), 256),)](
        work,
        out,
        out.numel(),
        splits,
        256,
    )
    return out


def time_graph(functions, calls=32, samples=7):
    for fn in functions:
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            functions[i % len(functions)]()
    times = []
    for _ in range(samples):
        first, last = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        first.record()
        graph.replay()
        last.record()
        last.synchronize()
        times.append(first.elapsed_time(last) * 1000 / calls)
    return statistics.median(times)


def main(args):
    torch.accelerator.set_device_index(args.device)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("highest")
    records = []
    variants = [
        ("dot", bn, bk, s) for bn in (32, 64) for bk in (32, 64) for s in (1, 4, 8)
    ]
    variants += [("gemv", bn, 256, s) for bn in (4, 8) for s in (1, 4)]
    for groups in args.groups:
        for rows in args.rows:
            x = (
                torch.randn(rows, groups, 4096, device="cuda", dtype=torch.bfloat16)
                * 0.2
            )
            w = (
                torch.randn(groups, 1024, 4096, device="cuda", dtype=torch.bfloat16)
                * 0.02
            )
            expected = torch.einsum("tgd,gnd->tgn", x.float(), w.float())
            weights = [w] + [w.clone() for _ in range(max(4, 16 // groups) - 1)]

            def baseline(weight, input_rows=x):
                return torch.einsum("tgd,gnd->tgn", input_rows, weight).flatten(1)

            original = baseline(w).view_as(expected)
            row = {
                "groups": groups,
                "rows": rows,
                "baseline_us": time_graph(
                    [partial(baseline, weight) for weight in weights]
                ),
                "baseline_max_abs_error": (original.float() - expected)
                .abs()
                .max()
                .item(),
                "variants": [],
            }
            for kind, bn, bk, splits in variants:
                work = torch.empty(
                    splits, rows, groups, 1024, device="cuda", dtype=torch.float32
                )
                out = torch.empty(expected.shape, device=x.device, dtype=torch.bfloat16)
                fn = partial(invoke, x, w, work, out, kind, bn, bk, splits)
                actual = fn()
                error = (actual.float() - expected).abs().max().item()
                torch.testing.assert_close(
                    actual.float(), expected, atol=0.003, rtol=0.008
                )
                us = time_graph(
                    [
                        partial(invoke, x, weight, work, out, kind, bn, bk, splits)
                        for weight in weights
                    ]
                )
                record = {
                    "kind": kind,
                    "bn": bn,
                    "bk": bk,
                    "splits": splits,
                    "us": us,
                    "max_abs_error": error,
                    "exact_baseline_fraction": (actual == original)
                    .float()
                    .mean()
                    .item(),
                }
                row["variants"].append(record)
            row["best"] = min(row["variants"], key=lambda v: v["us"])
            records.append(row)
            print({k: v for k, v in row.items() if k != "variants"}, flush=True)
            Path(args.output).write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--groups", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--rows", type=int, nargs="+", default=[1, 5, 6, 12])
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
