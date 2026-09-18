# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 expert GEMMs with both hot weights and a working set larger than L2."""

import argparse
import json
import math
from functools import partial
from pathlib import Path

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    rand_marlin_weight_mxfp4_like,
)
from vllm.scalar_type import scalar_types
from vllm.triton_utils import triton


def main(args):
    torch.manual_seed(170)
    torch.set_num_threads(1)
    records = []
    for tp, name, k, n in [
        (1, "gate_up", 5120, 4608),
        (1, "down", 2304, 5120),
        (2, "gate_up", 5120, 2304),
        (2, "down", 1152, 5120),
    ]:
        shape = torch.empty(n, k, device="cuda", dtype=torch.bfloat16)
        ref, packed, scales = rand_marlin_weight_mxfp4_like(shape, 32)
        weight_bytes = packed.nbytes + scales.nbytes
        copies = math.ceil(64 * 1024**2 / weight_bytes)
        weights = [(packed, scales)] + [
            (packed.clone(), scales.clone()) for _ in range(copies - 1)
        ]
        workspace = marlin_make_workspace_new(torch.device("cuda"))
        for m in (1, 6, 16, 64):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)

            calls = [
                partial(
                    ops.marlin_gemm,
                    x,
                    out,
                    weight,
                    None,
                    scale,
                    None,
                    None,
                    None,
                    workspace,
                    scalar_types.float4_e2m1f,
                    m,
                    n,
                    k,
                    use_atomic_add=False,
                    use_fp32_reduce=True,
                    is_zp_float=False,
                )
                for weight, scale in weights
            ]
            run = calls[0]

            def stream(calls=calls):
                for call in calls:
                    call()

            expected = x @ ref
            actual = run().clone()
            error = (
                actual.float() - expected.float()
            ).abs().mean() / expected.float().abs().mean()
            assert error.item() < 0.01, (m, n, k, error.item())
            # Warm both launch paths before recording either measurement.
            triton.testing.do_bench_cudagraph(run, rep=100)
            triton.testing.do_bench_cudagraph(stream, rep=100)
            hot_us = triton.testing.do_bench_cudagraph(run, rep=100) * 1000
            stream_us = (
                triton.testing.do_bench_cudagraph(stream, rep=100) * 1000 / copies
            )
            torch.testing.assert_close(out, actual, rtol=0, atol=0)
            record = {
                "tp": tp,
                "projection": name,
                "m": m,
                "n": n,
                "k": k,
                "weight_bytes": weight_bytes,
                "working_set_bytes": copies * weight_bytes,
                "hot_us": hot_us,
                "stream_us": stream_us,
                "weight_read_gb_s": weight_bytes / stream_us / 1000,
                "relative_mean_abs_error": error.item(),
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            Path(args.output).write_text(
                json.dumps(
                    {
                        "device": str(torch.cuda.get_device_properties(0)),
                        "results": records,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
