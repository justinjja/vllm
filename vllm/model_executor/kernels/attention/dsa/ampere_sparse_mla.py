# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse 512-dimension MLA on SM80 with direct packed-cache reads."""

import torch

from vllm.triton_utils import tl, triton
from vllm.triton_utils.fp8_compat import decode_fp8


@triton.jit
def _load_keys(
    CACHE,
    IDX,
    LENS,
    row,
    col,
    dims,
    WIDTH: tl.constexpr,
    ISTRIDE: tl.constexpr,
    PAGE: tl.constexpr,
    CSTRIDE: tl.constexpr,
    SLOTS,
    RECORD: tl.constexpr,
):
    length = tl.load(LENS + row)
    slot = tl.load(IDX + row * ISTRIDE + col, col < WIDTH, other=-1)
    valid = (col < WIDTH) & (col < length) & (slot >= 0) & (slot < SLOTS)
    slot = tl.where(valid, slot, 0).to(tl.int64)
    if RECORD == 0:
        values = tl.load(
            CACHE + slot[:, None] * CSTRIDE + dims[None, :], valid[:, None], other=0
        ).to(tl.bfloat16)
    else:
        page = CACHE + slot // PAGE * CSTRIDE
        if RECORD == 528:
            data = page + slot % PAGE * 512
            scales = page + PAGE * 512 + slot % PAGE * 16
            bits = tl.load(data[:, None] + dims[None, :], valid[:, None], other=0)
            exponent = tl.load(
                scales[:, None] + dims[None, :] // 32, valid[:, None], other=127
            )
            scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            values = (decode_fp8(bits) * scale).to(tl.bfloat16)
        else:
            data = page + slot % PAGE * 576
            scales = page + PAGE * 576 + slot % PAGE * 8
            bits = tl.load(
                data[:, None] + dims[None, :],
                valid[:, None] & (dims[None, :] < 448),
                other=0,
            )
            exponent = tl.load(
                scales[:, None] + dims[None, :] // 64,
                valid[:, None] & (dims[None, :] < 448),
                other=127,
            )
            scale = (exponent.to(tl.uint32) << 23).to(tl.float32, bitcast=True)
            nope = decode_fp8(bits) * scale
            rope_ptr = (data + 448).to(tl.pointer_type(tl.bfloat16))
            rope = tl.load(
                rope_ptr[:, None] + dims[None, :] - 448,
                valid[:, None] & (dims[None, :] >= 448),
                other=0,
            )
            values = tl.where(dims[None, :] < 448, nope, rope).to(tl.bfloat16)
    return values, valid


@triton.jit
def _attend(q, keys, valid, acc, maximum, denominator, SCALE: tl.constexpr):
    logits = tl.dot(q, tl.trans(keys)) * SCALE
    logits = tl.where(valid[None, :], logits, -float("inf"))
    new_max = tl.maximum(maximum, tl.max(logits, 1))
    safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
    alpha = tl.exp(maximum - safe_max)
    prob = tl.exp(logits - safe_max[:, None])
    acc = acc * alpha[:, None] + tl.dot(prob.to(tl.bfloat16), keys)
    denominator = denominator * alpha + tl.sum(prob, 1)
    return acc, new_max, denominator


@triton.jit(do_not_specialize=["DENSE_MS", "DENSE_ES"])
def _sparse_mla(
    Q,
    MAIN,
    MID,
    MLEN,
    EXTRA,
    EID,
    ELEN,
    SINK,
    PART,
    STATS,
    OUT,
    H: tl.constexpr,
    Q0: tl.constexpr,
    Q1: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    MW: tl.constexpr,
    MI: tl.constexpr,
    MP: tl.constexpr,
    MC: tl.constexpr,
    MS: tl.constexpr,
    MR: tl.constexpr,
    EW: tl.constexpr,
    EI: tl.constexpr,
    EP: tl.constexpr,
    EC: tl.constexpr,
    ES: tl.constexpr,
    ER: tl.constexpr,
    HAS_SINK: tl.constexpr,
    SCALE: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    SPLITS: tl.constexpr,
    TRIM_EMPTY_TILES: tl.constexpr,
    DENSE_MS,
    DENSE_ES,
):
    main_slots = MS if MR else DENSE_MS
    extra_slots = ES if ER else DENSE_ES
    row, hg, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    heads = hg * BH + tl.arange(0, BH)
    dims = tl.arange(0, 512)
    q = tl.load(
        Q + row * Q0 + heads[:, None] * Q1 + dims[None, :], heads[:, None] < H, other=0
    ).to(tl.bfloat16)
    maximum = tl.full((BH,), -float("inf"), tl.float32)
    denominator = tl.zeros((BH,), tl.float32)
    acc = tl.zeros((BH, 512), tl.float32)
    main_end = (
        tl.minimum(MW, tl.maximum(0, tl.load(MLEN + row))) if TRIM_EMPTY_TILES else MW
    )
    for start in range(split * BN, MW, SPLITS * BN):
        if start < main_end:
            col = start + tl.arange(0, BN)
            keys, valid = _load_keys(
                MAIN, MID, MLEN, row, col, dims, MW, MI, MP, MC, main_slots, MR
            )
            acc, maximum, denominator = _attend(
                q, keys, valid, acc, maximum, denominator, SCALE
            )
    if EW:
        extra_end = (
            tl.minimum(EW, tl.maximum(0, tl.load(ELEN + row)))
            if TRIM_EMPTY_TILES
            else EW
        )
        for start in range(split * BN, EW, SPLITS * BN):
            if start < extra_end:
                col = start + tl.arange(0, BN)
                keys, valid = _load_keys(
                    EXTRA, EID, ELEN, row, col, dims, EW, EI, EP, EC, extra_slots, ER
                )
                acc, maximum, denominator = _attend(
                    q, keys, valid, acc, maximum, denominator, SCALE
                )
    if SPLITS == 1:
        sink = (
            tl.load(SINK + heads, heads < H, other=-float("inf"))
            if HAS_SINK
            else tl.full((BH,), -float("inf"), tl.float32)
        )
        final_max = tl.maximum(maximum, sink)
        final_max = tl.where(final_max == -float("inf"), 0.0, final_max)
        alpha = tl.exp(maximum - final_max)
        divisor = denominator * alpha + tl.exp(sink - final_max)
        result = acc * (alpha / tl.where(divisor > 0, divisor, 1.0))[:, None]
        tl.store(
            OUT + row * O0 + heads[:, None] * O1 + dims[None, :],
            result,
            heads[:, None] < H,
        )
    else:
        offset = (row * H + heads) * SPLITS + split
        tl.store(PART + offset[:, None] * 512 + dims[None, :], acc, heads[:, None] < H)
        tl.store(STATS + offset * 2, maximum, heads < H)
        tl.store(STATS + offset * 2 + 1, denominator, heads < H)


@triton.jit
def _merge_splits(
    PART,
    STATS,
    SINK,
    OUT,
    H: tl.constexpr,
    O0: tl.constexpr,
    O1: tl.constexpr,
    SPLITS: tl.constexpr,
    BS: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    s = tl.arange(0, BS)
    d = tl.arange(0, 512)
    offset = (row * H + head) * SPLITS + s
    maxima = tl.load(STATS + offset * 2, s < SPLITS, other=-float("inf"))
    denominators = tl.load(STATS + offset * 2 + 1, s < SPLITS, other=0)
    sink = tl.load(SINK + head) if HAS_SINK else -float("inf")
    maximum = tl.maximum(tl.max(maxima, 0), sink)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    alpha = tl.exp(maxima - maximum)
    denominator = tl.sum(denominators * alpha, 0) + tl.exp(sink - maximum)
    parts = tl.load(
        PART + offset[:, None] * 512 + d[None, :], s[:, None] < SPLITS, other=0
    )
    result = tl.sum(parts * alpha[:, None], 0) / tl.where(
        denominator > 0, denominator, 1.0
    )
    tl.store(OUT + row * O0 + head * O1 + d, result)


def sparse_mla(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_lengths: torch.Tensor,
    scale: float,
    sink: torch.Tensor | None,
    output: torch.Tensor,
    extra_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_lengths: torch.Tensor | None = None,
    *,
    splits: int | None = None,
    block_n: int | None = None,
    num_warps: int | None = None,
    block_h: int = 16,
    trim_empty_tiles: bool = False,
) -> None:
    """Attend to selected dense rows or packed 584/528-byte cache records.

    Indices are physical token slots; each row's length masks its padded tail.
    The optional second cache is combined in the same softmax, with one sink.
    Empty rows produce zero. All input metadata is read on device for replay.
    """
    rows, heads, dim = q.shape
    assert q.dtype == torch.bfloat16 and dim == 512 and q.stride(-1) == 1
    assert (
        output.shape == q.shape and output.dtype == q.dtype and output.stride(-1) == 1
    )
    assert sink is None or (sink.is_contiguous() and sink.numel() >= heads)
    if rows == 0:
        return
    if num_warps is None:
        # Packed-cache decoding otherwise spills registers on SM80.
        num_warps = 8 if main_cache.dtype == torch.uint8 else 4

    def layout(cache, indices, lengths):
        assert cache.stride(-1) == 1
        indices = indices.reshape(rows, -1)
        assert indices.stride(-1) == 1
        assert lengths.numel() == rows and lengths.is_contiguous()
        if cache.dtype == torch.uint8:
            assert cache.ndim == 3 and cache.shape[-1] in (528, 584)
            page, record = cache.shape[1:]
            slots = cache.shape[0] * page
        else:
            assert (
                cache.dtype == torch.bfloat16
                and cache.ndim == 2
                and cache.shape[1] == 512
            )
            page, record, slots = 1, 0, cache.shape[0]
        return indices, (
            indices.shape[1],
            indices.stride(0),
            page,
            cache.stride(0),
            slots if record else 0,
            record,
        )

    main_indices, main_layout = layout(main_cache, main_indices, main_lengths)
    extra_layout = (0, 0, 1, 0, 0, 0)
    if extra_cache is not None:
        assert extra_indices is not None and extra_lengths is not None
        extra_indices, extra_layout = layout(extra_cache, extra_indices, extra_lengths)
    v41_decode = main_layout[0] == 128 and extra_layout[0] == 512
    if block_n is None:
        block_n = 64 if v41_decode and rows == 12 and heads == 32 else 32
    if splits is None and v41_decode and rows == 4 and heads == 64:
        splits = 4
    if splits is None:
        tiles = max(
            triton.cdiv(main_layout[0], block_n),
            triton.cdiv(extra_layout[0], block_n),
            1,
        )
        splits = min(
            16, tiles, max(1, triton.cdiv(70, rows * triton.cdiv(heads, block_h)))
        )
    assert 1 <= splits <= 32 and block_n in (32, 64)
    assert num_warps in (4, 8, 16)
    assert block_h in (16, 32, 64)
    partial = stats = None
    if splits > 1:
        partial = torch.empty(
            (rows, heads, splits, 512), device=q.device, dtype=torch.float32
        )
        stats = torch.empty(
            (rows, heads, splits, 2), device=q.device, dtype=torch.float32
        )
    _sparse_mla[(rows, triton.cdiv(heads, block_h), splits)](
        q,
        main_cache,
        main_indices,
        main_lengths,
        extra_cache,
        extra_indices,
        extra_lengths,
        sink,
        partial,
        stats,
        output,
        heads,
        q.stride(0),
        q.stride(1),
        output.stride(0),
        output.stride(1),
        *main_layout,
        *extra_layout,
        sink is not None,
        scale,
        block_h,
        block_n,
        splits,
        trim_empty_tiles,
        main_cache.shape[0] if main_cache.dtype == torch.bfloat16 else 0,
        extra_cache.shape[0]
        if extra_cache is not None and extra_cache.dtype == torch.bfloat16
        else 0,
        num_warps=num_warps,
        num_stages=1,
    )
    if splits > 1:
        _merge_splits[(rows, heads)](
            partial,
            stats,
            sink,
            output,
            heads,
            output.stride(0),
            output.stride(1),
            splits,
            triton.next_power_of_2(splits),
            sink is not None,
            num_warps=4,
        )
