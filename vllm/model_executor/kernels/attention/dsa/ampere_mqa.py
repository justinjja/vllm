# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FP8 indexer logits on SM80, with shared KV tiles across query rows.

E4M3 values are exactly representable in FP16 and BF16; both MMA paths use
FP32 accumulators. FP16 permits a cheaper software exponent conversion.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def decode_e4m3_bf16(bits):
    """Decode E4M3FN exactly without FP8 conversion instructions."""
    bits = bits.to(tl.uint32)
    payload = bits & 127
    normal = ((payload << 20) + 0x3C000000).to(tl.float32, bitcast=True)
    magnitude = tl.where(payload < 8, payload.to(tl.float32) / 512.0, normal)
    signed = magnitude.to(tl.uint32, bitcast=True) | ((bits & 128) << 24)
    value = signed.to(tl.float32, bitcast=True)
    return tl.where(payload == 127, float("nan"), value).to(tl.bfloat16)


@triton.jit
def decode_e4m3_fp16(bits):
    """Reinterpret with FP16's bias, then correct the exponent in FP16."""
    bits = bits.to(tl.uint16)
    payload = bits & 127
    half_bits = (payload << 7) | ((bits & 128) << 8)
    value = half_bits.to(tl.float16, bitcast=True)
    value = value * tl.full((), 256.0, tl.float16)
    return tl.where(payload == 127, float("nan"), value).to(tl.float16)


@triton.jit(do_not_specialize=["DENSE_M", "DENSE_N"])
def _mqa_logits(
    Q,
    K,
    S,
    W,
    START,
    END,
    TABLE,
    INDICES,
    OUT,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    K0: tl.constexpr,
    S0: tl.constexpr,
    W0: tl.constexpr,
    W1: tl.constexpr,
    T0: tl.constexpr,
    T1: tl.constexpr,
    M: tl.constexpr,
    N: tl.constexpr,
    H: tl.constexpr,
    BH: tl.constexpr,
    BR: tl.constexpr,
    BN: tl.constexpr,
    PAGED: tl.constexpr,
    PAGE: tl.constexpr,
    NEXT: tl.constexpr,
    END_PER_ROW: tl.constexpr,
    HAS_INDICES: tl.constexpr,
    FP16: tl.constexpr,
    DENSE_M,
    DENSE_N,
):
    m = M if PAGED else DENSE_M
    n = N if PAGED else DENSE_N
    group = tl.program_id(0)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    local_rows = tl.arange(0, BR)
    if PAGED:
        query_req = group // tl.cdiv(NEXT, BR)
        offset = group % tl.cdiv(NEXT, BR) * BR
        rows = query_req * NEXT + offset + local_rows
        row_valid = (offset + local_rows < NEXT) & (rows < m)
        req = tl.load(INDICES + query_req) if HAS_INDICES else query_req
        end_row = rows if END_PER_ROW else query_req + tl.zeros((BR,), tl.int32)
        hi = tl.load(END + end_row, row_valid, other=0)
        lo = tl.full((BR,), 0, tl.int32)
    else:
        rows = group * BR + local_rows
        row_valid = rows < m
        lo = tl.load(START + rows, row_valid, other=n)
        hi = tl.load(END + rows, row_valid, other=0)
    # Captured decode reserves the model's full context. Empty tiles only
    # need the masked sentinel; avoid loading queries and doing zero GEMMs.
    if PAGED and tl.program_id(1) * BN >= tl.max(hi, 0):
        tl.store(
            OUT + rows[:, None] * n + cols[None, :],
            -float("inf"),
            row_valid[:, None] & (cols[None, :] < n),
        )
    else:
        valid_k = (cols < n) & (cols >= tl.min(lo, 0)) & (cols < tl.max(hi, 0))
        dims = tl.arange(0, 128)
        if PAGED:
            page = tl.load(TABLE + req * T0 + cols // PAGE * T1, valid_k, other=0)
            base = K + page.to(tl.int64) * K0
            kp = base[None, :] + (cols % PAGE)[None, :] * 128 + dims[:, None]
            sp = (base + PAGE * 128).to(tl.pointer_type(tl.float32)) + cols % PAGE
        else:
            kp = K + cols[None, :] * K0 + dims[:, None]
            sp = S + cols * S0
        kb = tl.load(kp, valid_k[None, :], other=0)
        scale = tl.load(sp, valid_k, other=0)
        rh = tl.arange(0, BR * BH)
        h = rh % BH
        qr = group * BR + rh // BH
        if PAGED:
            qr = query_req * NEXT + offset + rh // BH
            valid_q = (offset + rh // BH < NEXT) & (qr < m) & (h < H)
        else:
            valid_q = (qr < m) & (h < H)
        qb = tl.load(
            Q + qr[:, None] * Q0 + h[:, None] * Q1 + dims[None, :],
            valid_q[:, None],
            other=0,
        )
        if FP16:
            scores = tl.dot(decode_e4m3_fp16(qb), decode_e4m3_fp16(kb))
        else:
            scores = tl.dot(decode_e4m3_bf16(qb), decode_e4m3_bf16(kb))
        scores = tl.maximum(scores, 0.0)
        weight = tl.load(W + qr * W0 + h * W1, valid_q, other=0)
        weighted = (scores * weight[:, None]).reshape((BR, BH, BN))
        result = tl.sum(weighted, 1) * scale[None, :]
        visible = (cols[None, :] >= lo[:, None]) & (cols[None, :] < hi[:, None])
        tl.store(
            OUT + rows[:, None] * n + cols[None, :],
            tl.where(visible, result, -float("inf")),
            row_valid[:, None] & (cols[None, :] < n),
        )


def _check_query(q, weights):
    values, scales = q
    if scales is not None or values.dtype != torch.float8_e4m3fn:
        raise ValueError("SM80 indexer requires E4M3FN queries with folded scales")
    if values.shape[-1] != 128 or values.stride(-1) != 1:
        raise ValueError("SM80 indexer requires contiguous 128-element heads")
    if weights.dtype != torch.float32:
        raise ValueError("Indexer weights must be float32")
    return values


def _dense_tile(rows, heads, columns):
    if heads not in (16, 32) or rows < 32:
        return (1, 64, 4)
    if heads == 16:
        if rows < 512 and columns > 8192:
            return (4, 128, 4)
        return (8, 64, 4)
    return (4, 64, 4)


def _paged_tile(batch, next_n, heads, columns):
    if heads not in (16, 32):
        return (1, 64, 4)
    if next_n == 1:
        if batch == 1:
            return (1, 64 if heads == 16 else (32 if columns <= 8192 else 128), 4)
        return (1, 64 if heads == 16 and columns <= 8192 else 128, 4)
    if heads == 16:
        if batch == 1 and 4 <= next_n <= 8 and columns <= 4096:
            return (8, 32, 4)
        return (2, 128, 4)
    return (1, 128, 4)


def mqa_logits(
    q, kv, weights, starts, ends, clean_logits=False, *, tile=None, fp16=True, out=None
):
    """Return FP32 sum_h(weight_h * relu(Q_h K)) with [start, end) masking.

    FP8 query scales are folded into weights; keys carry one FP32 scale per
    position. Invalid positions are always -inf, including clean_logits=False.
    ``tile`` is an explicit (query rows, KV columns, warps) benchmark override.
    """
    values = _check_query(q, weights)
    keys, scales = kv
    if keys.dtype != torch.float8_e4m3fn or scales.dtype != torch.float32:
        raise ValueError("SM80 indexer requires E4M3FN keys and float32 scales")
    m, h, d = values.shape
    n = keys.shape[0]
    assert keys.shape == (n, d) and keys.stride(1) == 1
    assert scales.shape == (n,)
    assert weights.shape == (m, h)
    assert starts.shape == ends.shape == (m,)
    assert starts.is_contiguous() and ends.is_contiguous()
    if out is None:
        out = torch.empty((m, n), dtype=torch.float32, device=values.device)
    else:
        assert out.shape == (m, n) and out.is_contiguous()
        assert out.dtype == torch.float32 and out.device == values.device
    if m == 0 or n == 0:
        return out
    br, bn, warps = tile or _dense_tile(m, h, n)
    _mqa_logits[(triton.cdiv(m, br), triton.cdiv(n, bn))](
        values.view(torch.uint8),
        keys.view(torch.uint8),
        scales,
        weights,
        starts,
        ends,
        None,
        None,
        out,
        values.stride(0),
        values.stride(1),
        keys.stride(0),
        scales.stride(0),
        weights.stride(0),
        weights.stride(1),
        0,
        0,
        0,
        0,
        h,
        max(16, triton.next_power_of_2(h)),
        br,
        bn,
        False,
        1,
        1,
        True,
        False,
        fp16,
        m,
        n,
        num_warps=warps,
    )
    return out


def paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_tables,
    schedule_metadata,
    max_model_len,
    clean_logits=False,
    indices=None,
    *,
    tile=None,
    fp16=True,
):
    """Paged FP8 logits, sharing KV loads only within the same request.

    Cache pages contain PAGE*128 key bytes followed by PAGE FP32 scales.
    Context lengths may be [B], [B,1], or [B,next_n]. Optional indices map
    query requests to block-table rows. Scheduling metadata is unused on SM80.
    """
    values = _check_query(q, weights)
    b, next_n, h, d = values.shape
    values = values.reshape(b * next_n, h, d)
    assert kv_cache.dtype == torch.uint8 and kv_cache.stride(-1) == 1
    assert kv_cache.stride(1) == d + 4
    assert kv_cache.shape[-1] == d + 4
    assert weights.shape == (b * next_n, h)
    assert context_lens.is_contiguous() and context_lens.numel() in (b, b * next_n)
    assert block_tables.ndim == 2
    assert indices is None or (indices.shape == (b,) and indices.is_contiguous())
    n = max_model_len
    out = torch.empty((b * next_n, n), dtype=torch.float32, device=values.device)
    if b == 0 or n == 0:
        return out
    br, bn, warps = tile or _paged_tile(b, next_n, h, n)
    _mqa_logits[(b * triton.cdiv(next_n, br), triton.cdiv(n, bn))](
        values.view(torch.uint8),
        kv_cache,
        None,
        weights,
        None,
        context_lens,
        block_tables,
        indices,
        out,
        values.stride(0),
        values.stride(1),
        kv_cache.stride(0),
        0,
        weights.stride(0),
        weights.stride(1),
        block_tables.stride(0),
        block_tables.stride(1),
        b * next_n,
        n,
        h,
        max(16, triton.next_power_of_2(h)),
        br,
        bn,
        True,
        kv_cache.shape[1],
        next_n,
        context_lens.numel() == b * next_n,
        indices is not None,
        fp16,
        0,
        0,
        num_warps=warps,
    )
    return out
