# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare dense indexer specialization cost and steady-state GPU time."""

import argparse
import importlib.util
import json
import statistics
import sys
import time
from functools import partial
from pathlib import Path

import torch

from vllm.model_executor.kernels.attention.dsa import ampere_mqa
from vllm.triton_utils import triton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "cmp_mqa_baseline", args.baseline_source
    )
    assert spec is not None and spec.loader is not None
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    torch.manual_seed(2026)
    torch.set_num_threads(1)
    records = []
    shapes = [(m, n, h) for m in (128, 1024) for n in (4096, 32768) for h in (16, 32)]
    shapes += [(128, n, 16) for n in (1001, 2003, 3007, 4009)]
    for m, n, heads in shapes:
        q = torch.randn(m, heads, 128, device="cuda").to(torch.float8_e4m3fn)
        k = torch.randn(n, 128, device="cuda").to(torch.float8_e4m3fn)
        scales = torch.rand(n, device="cuda")
        weights = torch.randn(m, heads, device="cuda")
        starts = torch.zeros(m, device="cuda", dtype=torch.int32)
        ends = torch.arange(n - m + 1, n + 1, device="cuda", dtype=torch.int32)
        functions = [
            partial(module.mqa_logits, (q, None), (k, scales), weights, starts, ends)
            for module in (baseline, ampere_mqa)
        ]
        outputs, first_call = [], []
        for fn in functions:
            torch.accelerator.synchronize()
            begin = time.perf_counter()
            outputs.append(fn())
            torch.accelerator.synchronize()
            first_call.append(time.perf_counter() - begin)
        torch.testing.assert_close(outputs[1], outputs[0], rtol=2e-5, atol=3e-4)
        for fn in functions:
            triton.testing.do_bench_cudagraph(fn, rep=100)
        samples = [[], []]
        for repeat in range(4):
            for i in (0, 1) if repeat % 2 == 0 else (1, 0):
                samples[i].append(
                    triton.testing.do_bench_cudagraph(functions[i], rep=100) * 1000
                )
        old, new = map(statistics.median, samples)
        record = {
            "rows": m,
            "columns": n,
            "heads": heads,
            "baseline_us": old,
            "dynamic_us": new,
            "speedup": old / new,
            "first_call_seconds": first_call,
            "samples_us": samples,
        }
        records.append(record)
        args.output.write_text(json.dumps(records, indent=2) + "\n")
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
