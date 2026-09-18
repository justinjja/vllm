# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 FP8 scoring of unique causal blocks from the V4.1 candidate selector."""

import torch

from vllm.model_executor.kernels.attention.dsa.ampere_mqa import (
    _dense_tile,
    decode_e4m3_fp16,
)
from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["N"])
def _candidate_logits(
    Q,
    K,
    S,
    W,
    START,
    END,
    C,
    OUT,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    S0: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    N,
    H: tl.constexpr,
    BH: tl.constexpr,
    NC: tl.constexpr,
    BS: tl.constexpr,
    BN: tl.constexpr,
    BR: tl.constexpr,
):
    row = tl.program_id(0)
    packed = tl.program_id(1) * BN + tl.arange(0, BN)
    block = tl.load(C + row * C0 + packed // BS * C1, packed < NC * BS, other=-1)
    lo = tl.load(START + row)
    hi = tl.load(END + row)
    cols = lo + block * BS + packed % BS
    valid = (packed < NC * BS) & (block >= 0) & (cols < hi) & (cols < N)
    if tl.sum(valid.to(tl.int32), 0) > 0:
        d = tl.arange(0, 128)
        # Preserve the dense scorer's MMA and head-reduction layout exactly.
        # Padding rows avoid changing FP32 summation order.
        rh = tl.arange(0, BR * BH)
        h = rh % BH
        qb = tl.load(
            Q + row * Q0 + h[:, None] * Q1 + d[None, :],
            (h[:, None] < H) & (rh[:, None] < BH),
            other=0,
        )
        kb = tl.load(K + cols[None, :] * K0 + d[:, None], valid[None, :], other=0)
        scores = tl.maximum(tl.dot(decode_e4m3_fp16(qb), decode_e4m3_fp16(kb)), 0.0)
        weight = tl.load(W + row * W0 + h * W1, (h < H) & (rh < BH), other=0)
        scale = tl.load(S + cols * S0, valid, other=0)
        result = (
            tl.sum((scores * weight[:, None]).reshape((BR, BH, BN)), 1) * scale[None, :]
        )
        tl.store(
            OUT + row.to(tl.int64) * N + cols[None, :] + tl.zeros((BR, 1), tl.int64),
            result,
            (tl.arange(0, BR)[:, None] == 0) & valid[None, :],
        )


def candidate_mqa_logits(q, kv, weights, starts, ends, candidates, block_size):
    values, qscale = q
    keys, scales = kv
    assert qscale is None
    m, h, d = values.shape
    n = keys.shape[0]
    assert d == 128 and h in (16, 32)
    assert values.dtype == keys.dtype == torch.float8_e4m3fn
    assert weights.dtype == scales.dtype == torch.float32
    assert starts.is_contiguous() and ends.is_contiguous()
    assert candidates.shape[0] == m
    out = torch.full((m, n), -float("inf"), device=values.device, dtype=torch.float32)
    br, bn, warps = _dense_tile(m, h, n)
    if m and n:
        _candidate_logits[(m, triton.cdiv(candidates.shape[1] * block_size, bn))](
            values.view(torch.uint8),
            keys.view(torch.uint8),
            scales,
            weights,
            starts,
            ends,
            candidates,
            out,
            values.stride(0),
            values.stride(1),
            keys.stride(0),
            scales.stride(0),
            weights.stride(0),
            weights.stride(1),
            *candidates.stride(),
            n,
            h,
            max(16, triton.next_power_of_2(h)),
            candidates.shape[1],
            block_size,
            bn,
            br,
            num_warps=warps,
        )
    return out
