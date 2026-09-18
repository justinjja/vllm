# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure native Marlin launch choices using calls from a routed MoE check."""

import itertools
import statistics
from unittest.mock import patch

import torch


def tune_projections(run, local_routes):
    import vllm._custom_ops as ops

    native = ops.moe_wna16_marlin_gemm
    calls = []

    def record(*args, **kwargs):
        output = native(*args, **kwargs)
        calls.append((args, kwargs, output.clone()))
        return output

    with patch.object(ops, "moe_wna16_marlin_gemm", record):
        run()
    assert len(calls) == 2
    mask = local_routes.flatten()
    variants = [(-1, -1, -1)] + [
        (k, n, blocks)
        for (k, n), blocks in itertools.product(
            [(128, 128), (64, 256), (64, 128), (128, 64)], range(1, 5)
        )
    ]
    records = []
    for projection, (args, kwargs, expected) in zip(("gate_up", "down"), calls):
        expected = expected[mask].float()
        for k, n, blocks in variants:
            choice = dict(thread_k=k, thread_n=n, blocks_per_sm=blocks)
            row = {"projection": projection, **choice}

            def invoke(args=args, kwargs=kwargs, choice=choice):
                return native(*args, **kwargs, **choice)

            def check(output, expected=expected):
                actual = output[mask].float()
                assert torch.isfinite(actual).all()
                if not actual.numel():
                    return 0.0
                error = (actual - expected).abs()
                relative = (error.mean() / expected.abs().mean().clamp_min(1e-9)).item()
                assert relative < 0.01, relative
                assert error.max() < expected.abs().max() * 0.05 + 1e-7
                return relative

            try:
                output = invoke()
            except RuntimeError as error:
                # Unsupported tile/shared-memory combinations are rejected by
                # the native pre-launch validation. Other failures must abort.
                message = str(error)
                if not any(
                    item in message
                    for item in ("Invalid thread config", "Unsupported shapes")
                ):
                    raise
                row["unsupported"] = message
                records.append(row)
                continue
            row["eager_relative_error"] = check(output)
            for _ in range(3):
                invoke()
            torch.accelerator.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = invoke()
            graph.replay()
            torch.accelerator.synchronize()
            row["graph_relative_error"] = check(output)
            times = []
            for _ in range(5):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(20):
                    graph.replay()
                end.record()
                end.synchronize()
                times.append(begin.elapsed_time(end) * 1000 / 20)
            row.update(median_us=statistics.median(times), samples_us=times)
            records.append(row)
            graph.reset()
    return records
