# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

from dataclasses import dataclass
from importlib import import_module

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

if current_platform.is_cuda():
    try:
        import_module("vllm._flashmla_C")

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import_module("vllm._flashmla_extension_C")

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


def _is_flashmla_available() -> tuple[bool, str | None]:
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "was not compiled due to a build error.",
        )

    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not (
        current_platform.is_device_capability_family(90)
        or current_platform.is_device_capability_family(100)
    ):
        return (
            False,
            "FlashMLA Sparse is only supported on Hopper and Blackwell devices.",
        )
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


flash_attn_varlen_func = _raise_flashmla_unavailable  # type: ignore[assignment]
flash_attn_varlen_kvpacked_func = (  # type: ignore[assignment]
    _raise_flashmla_unavailable
)
flash_attn_varlen_qkvpacked_func = (  # type: ignore[assignment]
    _raise_flashmla_unavailable
)


@dataclass
class FlashMLASchedMeta:
    """Scheduler placeholder used by FlashMLA and the vLLM SM86 fallback."""

    @dataclass
    class Config:
        b: int
        s_q: int
        h_q: int
        page_block_size: int
        h_k: int
        causal: bool
        is_fp8_kvcache: bool
        topk: int | None
        extra_page_block_size: int | None
        extra_topk: int | None

    have_initialized: bool = False
    config: Config | None = None
    tile_scheduler_metadata: torch.Tensor | None = None
    num_splits: torch.Tensor | None = None


def get_mla_metadata(*args, **kwargs) -> tuple[FlashMLASchedMeta, None]:
    # Arguments are accepted for API compatibility with the old FlashMLA
    # interface. Metadata is initialized by flash_mla_with_kvcache on first use.
    return FlashMLASchedMeta(), None


def _use_torch_sparse_fallback(q: torch.Tensor) -> bool:
    if not q.is_cuda:
        return False
    major, _ = torch.cuda.get_device_capability(q.device)
    return major < 9


def _flash_mla_sparse_fwd_torch_fallback(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s_q, h_q, d_qk = q.shape
    if out is None:
        out = torch.empty((s_q, h_q, d_v), device=q.device, dtype=q.dtype)
    max_logits = torch.empty((s_q, h_q), device=q.device, dtype=torch.float32)
    lse = torch.empty((s_q, h_q), device=q.device, dtype=torch.float32)

    q_f = q.float()
    kv_f = kv[:, 0, :].float()
    for i in range(s_q):
        limit = topk_length[i] if topk_length is not None else None
        key, valid = _gather_bf16_cache_torch_static(
            kv_f,
            indices[i, 0],
            d_qk,
            limit,
        )
        if key.numel() == 0:
            out[i].zero_()
            max_logits[i].fill_(float("-inf"))
            lse[i].fill_(float("-inf"))
            continue
        scores = torch.matmul(q_f[i], key[:, :d_qk].T) * sm_scale
        scores = scores.masked_fill(~valid.unsqueeze(0), -1.0e30)
        row_max = scores.max(dim=-1).values
        exp_scores = torch.exp(scores - row_max.unsqueeze(-1))
        exp_scores = exp_scores * valid.to(exp_scores.dtype).unsqueeze(0)
        denom = exp_scores.sum(dim=-1).clamp_min(1.0e-20)
        probs = exp_scores / denom.unsqueeze(-1)
        out_i = torch.matmul(probs, key[:, :d_v])
        row_lse = row_max + torch.log(denom)

        if attn_sink is not None:
            sink_scale = torch.reciprocal(1.0 + torch.exp(attn_sink.float() - row_lse))
            out_i = out_i * sink_scale.unsqueeze(-1)

        out[i].copy_(out_i.to(out.dtype))
        max_logits[i].copy_(row_max)
        lse[i].copy_(row_lse)
    return out, max_logits, lse


def _gather_bf16_cache_torch_static(
    kv: torch.Tensor,
    indices: torch.Tensor,
    head_dim: int,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    idx = indices.to(torch.long)
    pos_valid = torch.ones_like(idx, dtype=torch.bool)
    if topk_length is not None:
        pos = torch.arange(idx.shape[-1], device=idx.device, dtype=idx.dtype)
        pos_valid = pos < topk_length.to(idx.dtype)
    valid = (idx >= 0) & (idx < kv.shape[0]) & pos_valid
    safe_idx = idx.clamp(0, max(kv.shape[0] - 1, 0))
    return kv.index_select(0, safe_idx)[:, :head_dim], valid


def _gather_deepseek_v4_fp8_cache_torch_static(
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    head_dim: int,
    topk_length: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_cache.shape[-1] != 584:
        raise RuntimeError(
            "SM86 sparse FlashMLA fallback only supports the DeepSeek V4 "
            f"fp8_ds_mla cache layout, got last dim {k_cache.shape[-1]}"
        )
    block_size = k_cache.shape[1]
    max_slots = k_cache.shape[0] * block_size
    if max_slots == 0:
        raise RuntimeError("SM86 sparse FlashMLA fallback received an empty KV cache")
    idx = indices.to(torch.long)
    pos_valid = torch.ones_like(idx, dtype=torch.bool)
    if topk_length is not None:
        pos = torch.arange(idx.shape[-1], device=idx.device, dtype=idx.dtype)
        pos_valid = pos < topk_length.to(idx.dtype)
    valid = (idx >= 0) & (idx < max_slots) & pos_valid
    safe_idx = idx.clamp(0, max_slots - 1)

    block_stride = k_cache.stride(0)
    token_fp8_dim = 448
    token_bf16_dim = head_dim - token_fp8_dim
    token_stride = token_fp8_dim + token_bf16_dim * 2
    scale_dim = 8
    cache_storage = torch.as_strided(
        k_cache,
        (k_cache.shape[0] * block_stride,),
        (1,),
    )
    block_ids = safe_idx // block_size
    pos = safe_idx % block_size
    token_base = block_ids * block_stride + pos * token_stride
    fp8_offsets = torch.arange(token_fp8_dim, device=k_cache.device)
    fp8_bytes = cache_storage[
        token_base.unsqueeze(1) + fp8_offsets.unsqueeze(0)
    ].contiguous()
    fp8_vals = fp8_bytes.view(torch.float8_e4m3fn).float()

    scale_offsets = torch.arange(7, device=k_cache.device)
    scale_base = block_ids * block_stride + block_size * token_stride + pos * scale_dim
    encoded_scales = cache_storage[
        scale_base.unsqueeze(1) + scale_offsets.unsqueeze(0)
    ].float()
    scales = torch.pow(2.0, encoded_scales - 127.0).repeat_interleave(64, dim=-1)

    bf16_offsets = torch.arange(token_bf16_dim * 2, device=k_cache.device)
    bf16_bytes = cache_storage[
        (token_base + token_fp8_dim).unsqueeze(1) + bf16_offsets.unsqueeze(0)
    ].contiguous()
    bf16_vals = bf16_bytes.view(torch.bfloat16).float()
    return torch.cat((fp8_vals * scales, bf16_vals), dim=-1), valid


def _flash_mla_sparse_decode_torch_fallback(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    extra_k_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_topk_length: torch.Tensor | None,
    head_dim_v: int,
    softmax_scale: float,
    out: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_q, heads, head_dim = q.shape
    if out is None:
        out = torch.empty(
            (bsz, seq_q, heads, head_dim_v),
            device=q.device,
            dtype=q.dtype,
        )
    lse = torch.empty((bsz, heads, seq_q), device=q.device, dtype=torch.float32)
    q_f = q.float()
    for b in range(bsz):
        for s in range(seq_q):
            limit = None
            if topk_length is not None:
                limit = topk_length[b] if topk_length.dim() == 1 else topk_length[b, s]
            key, valid = _gather_deepseek_v4_fp8_cache_torch_static(
                k_cache, indices[b, s], head_dim, limit
            )
            keys = [key]
            masks = [valid]
            if extra_k_cache is not None and extra_indices is not None:
                extra_limit = None
                if extra_topk_length is not None:
                    extra_limit = (
                        extra_topk_length[b]
                        if extra_topk_length.dim() == 1
                        else extra_topk_length[b, s]
                    )
                extra_key, extra_valid = _gather_deepseek_v4_fp8_cache_torch_static(
                    extra_k_cache,
                    extra_indices[b, s],
                    head_dim,
                    extra_limit,
                )
                keys.append(extra_key)
                masks.append(extra_valid)
            k_all = torch.cat(keys, dim=0)
            valid_all = torch.cat(masks, dim=0)
            scores = torch.matmul(q_f[b, s], k_all[:, :head_dim].T) * softmax_scale
            scores = scores.masked_fill(~valid_all.unsqueeze(0), -1.0e30)
            row_max = scores.max(dim=-1).values
            exp_scores = torch.exp(scores - row_max.unsqueeze(-1))
            exp_scores = exp_scores * valid_all.to(exp_scores.dtype).unsqueeze(0)
            denom = exp_scores.sum(dim=-1).clamp_min(1.0e-20)
            probs = exp_scores / denom.unsqueeze(-1)
            out_i = torch.matmul(probs, k_all[:, :head_dim_v])
            row_lse = row_max + torch.log(denom)
            if attn_sink is not None:
                sink_scale = torch.reciprocal(
                    1.0 + torch.exp(attn_sink.float() - row_lse)
                )
                out_i = out_i * sink_scale.unsqueeze(-1)
            out[b, s].copy_(out_i.to(out.dtype))
            lse[b, :, s].copy_(row_lse)
    return out, lse


def _flashmla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_C.sparse_prefill_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        out,
    )


def _validate_flashmla_sched_meta(
    sched_meta: FlashMLASchedMeta,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    causal: bool,
    is_fp8_kvcache: bool,
    topk: int | None,
    extra_page_block_size: int | None,
    extra_topk: int | None,
) -> None:
    if not sched_meta.have_initialized:
        sched_meta.have_initialized = True
        sched_meta.config = FlashMLASchedMeta.Config(
            q.shape[0],
            q.shape[1],
            q.shape[2],
            k_cache.shape[1],
            k_cache.shape[2],
            causal,
            is_fp8_kvcache,
            topk,
            extra_page_block_size,
            extra_topk,
        )
        return

    helper_msg = (
        " Input arguments are inconsistent with tile_scheduler_metadata. "
        "Reuse the same metadata object only when shapes and sparse settings "
        "are unchanged."
    )
    config = sched_meta.config
    assert config is not None
    assert config.b == q.shape[0], "batch size changed." + helper_msg
    assert config.s_q == q.shape[1], "query length changed." + helper_msg
    assert config.h_q == q.shape[2], "query head count changed." + helper_msg
    assert config.page_block_size == k_cache.shape[1], (
        "page block size changed." + helper_msg
    )
    assert config.h_k == k_cache.shape[2], "KV head count changed." + helper_msg
    assert config.causal == causal, "causal setting changed." + helper_msg
    assert config.is_fp8_kvcache == is_fp8_kvcache, (
        "FP8 cache setting changed." + helper_msg
    )
    assert config.topk == topk, "sparse top-k changed." + helper_msg
    assert config.extra_page_block_size == extra_page_block_size, (
        "extra cache page block size changed." + helper_msg
    )
    assert config.extra_topk == extra_topk, "extra sparse top-k changed." + helper_msg


def _flashmla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    assert num_splits is None, "num_splits must be None"
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    topk = indices.shape[-1] if indices is not None else None
    extra_page_block_size = (
        extra_k_cache.shape[1] if extra_k_cache is not None else None
    )
    extra_topk = (
        extra_indices_in_kvcache.shape[-1]
        if extra_indices_in_kvcache is not None
        else None
    )
    _validate_flashmla_sched_meta(
        tile_scheduler_metadata,
        q,
        k_cache,
        causal,
        is_fp8_kvcache,
        topk,
        extra_page_block_size,
        extra_topk,
    )

    if indices is not None:
        assert not causal, "causal must be False when sparse attention is enabled"
        assert is_fp8_kvcache, "is_fp8_kvcache must be True for sparse attention"
        out, lse, new_tile_metadata, new_num_splits = (
            torch.ops._flashmla_C.sparse_decode_fwd(
                q,
                k_cache,
                indices,
                topk_length,
                attn_sink,
                tile_scheduler_metadata.tile_scheduler_metadata,
                tile_scheduler_metadata.num_splits,
                extra_k_cache,
                extra_indices_in_kvcache,
                extra_topk_length,
                head_dim_v,
                softmax_scale,
                out,
            )
        )
    else:
        assert attn_sink is None, "attn_sink requires sparse attention"
        assert extra_k_cache is None, "extra_k_cache requires sparse attention"
        assert extra_indices_in_kvcache is None, (
            "extra_indices_in_kvcache requires sparse attention"
        )
        assert topk_length is None, "topk_length requires sparse attention"
        assert extra_topk_length is None, "extra_topk_length requires sparse attention"
        assert block_table is not None, "block_table is required for dense attention"
        assert cache_seqlens is not None, (
            "cache_seqlens is required for dense attention"
        )
        out, lse, new_tile_metadata, new_num_splits = (
            torch.ops._flashmla_C.dense_decode_fwd(
                q,
                k_cache,
                head_dim_v,
                cache_seqlens,
                block_table,
                softmax_scale,
                causal,
                tile_scheduler_metadata.tile_scheduler_metadata,
                tile_scheduler_metadata.num_splits,
                out,
            )
        )

    tile_scheduler_metadata.tile_scheduler_metadata = new_tile_metadata
    tile_scheduler_metadata.num_splits = new_num_splits
    return out, lse


def flash_mla_sparse_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    d_v: int = 512,
    attn_sink: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _use_torch_sparse_fallback(q):
        return _flash_mla_sparse_fwd_torch_fallback(
            q, kv, indices, sm_scale, d_v, attn_sink, topk_length, out
        )
    return _flashmla_sparse_fwd(
        q,
        kv,
        indices,
        sm_scale,
        d_v,
        attn_sink,
        topk_length,
        out,
    )


def flash_mla_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor | None,
    cache_seqlens: torch.Tensor | None,
    head_dim_v: int,
    tile_scheduler_metadata: FlashMLASchedMeta,
    num_splits: None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: torch.Tensor | None = None,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices_in_kvcache: torch.Tensor | None = None,
    topk_length: torch.Tensor | None = None,
    extra_topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if _use_torch_sparse_fallback(q) and indices is not None and is_fp8_kvcache:
        return _flash_mla_sparse_decode_torch_fallback(
            q,
            k_cache,
            indices,
            topk_length,
            attn_sink,
            extra_k_cache,
            extra_indices_in_kvcache,
            extra_topk_length,
            head_dim_v,
            softmax_scale,
            out,
        )
    return _flashmla_with_kvcache(
        q=q,
        k_cache=k_cache,
        block_table=block_table,
        cache_seqlens=cache_seqlens,
        head_dim_v=head_dim_v,
        tile_scheduler_metadata=tile_scheduler_metadata,
        num_splits=num_splits,
        softmax_scale=softmax_scale,
        causal=causal,
        is_fp8_kvcache=is_fp8_kvcache,
        indices=indices,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices_in_kvcache=extra_indices_in_kvcache,
        topk_length=topk_length,
        extra_topk_length=extra_topk_length,
        out=out,
    )


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    return torch.ops._flashmla_extension_C.get_mla_decoding_metadata_dense_fp8(
        cache_seqlens,
        num_q_tokens_per_head_k,
        num_heads_k,
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _is_flashmla_available()[0]:
        _raise_flashmla_unavailable()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    out, softmax_lse = torch.ops._flashmla_extension_C.fwd_kvcache_mla_fp8(
        q,
        k_cache,
        head_dim_v,
        cache_seqlens,
        block_table,
        softmax_scale,
        causal,
        tile_scheduler_metadata,
        num_splits,
        descale_q,
        descale_k,
    )
    return out, softmax_lse
