# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache import (
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
)
from vllm.v1.attention.ops.deepseek_v4_ops.fused_indexer_q import (
    MXFP4_BLOCK_SIZE,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


class CompressorBackend(AttentionBackend):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_name() -> str:
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


@dataclass
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]


class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        num_reqs = common_attn_metadata.num_reqs
        query_lens = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        x = torch.repeat_interleave(torch.arange(num_reqs), query_lens).pin_memory()
        token_to_req_indices = self.token_to_req_indices[: x.shape[0]]
        token_to_req_indices.copy_(x, non_blocking=True)
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
        )


class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576,  # NOTE: FlashMLA requires 576B alignment
        )

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return CompressorBackend


class DeepseekCompressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._fused_kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
            self._num_warps = 4
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._fused_kernel = (
                    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
                )
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._fused_kernel = _fused_kv_compress_norm_rope_insert_indexer_attn
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
            self._num_warps = 1
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def _use_torch_fused_fallback(self, x: torch.Tensor) -> bool:
        if not x.is_cuda or self.use_fp4_cache:
            return False
        major, _ = torch.cuda.get_device_capability(x.device)
        return major < 9

    def _compress_norm_rope_insert_torch(
        self,
        state_cache: torch.Tensor,
        state_width: int,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
    ) -> None:
        head_dim = self.head_dim
        rope_dim = self.rope_head_dim
        nope_dim = head_dim - rope_dim
        history_len = (1 + self.overlap) * self.compress_ratio
        arange_head = torch.arange(head_dim, device=state_cache.device)
        history_offsets = torch.arange(history_len, device=state_cache.device)
        norm_w = self.norm.weight.float()
        cache_bytes = torch.as_strided(
            kv_cache,
            (kv_cache.shape[0] * kv_cache.stride(0),),
            (1,),
        )

        for token_idx in range(slot_mapping.shape[0]):
            slot_id = slot_mapping[token_idx].to(torch.long)
            position = positions[token_idx].to(torch.long)
            kv_slot_id = kv_slot_mapping[token_idx].to(torch.long)
            req_idx = token_to_req_indices[token_idx].to(torch.long)
            active = (
                (slot_id >= 0)
                & (kv_slot_id >= 0)
                & (req_idx >= 0)
                & (((position + 1) % self.compress_ratio) == 0)
            )

            safe_req_idx = req_idx.clamp(0, block_table.shape[0] - 1)
            start = position - history_len + 1
            hist_pos = start + history_offsets
            valid = hist_pos >= 0
            safe_pos = torch.clamp(hist_pos, min=0)
            block_indices = safe_pos // block_size
            block_offsets = safe_pos % block_size
            req_blocks = block_table.index_select(
                0, safe_req_idx.reshape(1)).squeeze(0)
            block_numbers = req_blocks.index_select(
                0, block_indices.to(torch.long)).to(torch.long)

            rows = state_cache[block_numbers, block_offsets.to(torch.long)]
            head_offsets = (history_offsets >= self.compress_ratio).to(
                torch.long) * head_dim
            gather_idx = head_offsets.unsqueeze(1) + arange_head.unsqueeze(0)
            kv_hist = rows.gather(1, gather_idx)
            score_hist = rows.gather(1, state_width + gather_idx)
            score_hist = torch.where(
                valid.unsqueeze(1),
                score_hist,
                torch.full_like(score_hist, float("-inf")),
            )
            weights = torch.softmax(score_hist, dim=0)
            compressed = (kv_hist * weights).sum(dim=0)

            variance = compressed.pow(2).sum() / head_dim
            normed = compressed * torch.rsqrt(variance + self.rms_norm_eps)
            normed = normed * norm_w

            rope = normed.clone()
            if rope_dim:
                compressed_pos = (position // self.compress_ratio
                                  ) * self.compress_ratio
                compressed_pos = compressed_pos.clamp(
                    0, cos_sin_cache.shape[0] - 1)
                cache = cos_sin_cache.index_select(
                    0, compressed_pos.reshape(1)).squeeze(0)
                cos = cache[:rope_dim // 2]
                sin = cache[rope_dim // 2:]
                pairs = rope[nope_dim:head_dim].view(rope_dim // 2, 2)
                even = pairs[:, 0].clone()
                odd = pairs[:, 1].clone()
                pairs[:, 0] = even * cos - odd * sin
                pairs[:, 1] = odd * cos + even * sin

            safe_kv_slot_id = kv_slot_id.clamp(
                0, kv_cache.shape[0] * kv_cache.shape[1] - 1)
            kv_block_idx = safe_kv_slot_id // kv_cache.shape[1]
            kv_pos_in_block = safe_kv_slot_id % kv_cache.shape[1]
            block_base = kv_block_idx * kv_cache.stride(0)
            token_base = block_base + kv_pos_in_block * self._token_stride
            scale_base = (
                block_base
                + kv_cache.shape[1] * self._token_stride
                + kv_pos_in_block * self._scale_dim
            )

            if head_dim == 512:
                quant_input = normed.to(torch.bfloat16).float()
                blocks = quant_input.view(-1, self._quant_block)
                block_absmax = blocks.abs().amax(dim=1).clamp_min(1.0e-4)
                exponents = torch.ceil(torch.log2(block_absmax / 448.0))
                scales = torch.pow(2.0, exponents)
                quant_nope = torch.clamp(
                    blocks[:self._scale_dim - 1]
                    / scales[:self._scale_dim - 1].unsqueeze(1),
                    -448.0,
                    448.0,
                ).reshape(-1)
                fp8_bytes = quant_nope.to(torch.float8_e4m3fn).view(torch.uint8)
                nope_dst = token_base + torch.arange(
                    nope_dim, device=state_cache.device)
                cache_bytes[nope_dst] = torch.where(
                    active, fp8_bytes, cache_bytes[nope_dst])

                rope_bytes = rope[nope_dim:head_dim].to(
                    torch.bfloat16).contiguous().view(torch.uint8)
                rope_dst = token_base + nope_dim + torch.arange(
                    rope_dim * 2, device=state_cache.device)
                cache_bytes[rope_dst] = torch.where(
                    active, rope_bytes, cache_bytes[rope_dst])

                encoded = torch.clamp(
                    exponents[:self._scale_dim - 1] + 127.0,
                    0.0,
                    255.0,
                ).to(torch.uint8)
                scale_dst = scale_base + torch.arange(
                    self._scale_dim - 1, device=state_cache.device)
                cache_bytes[scale_dst] = torch.where(
                    active, encoded, cache_bytes[scale_dst])
            else:
                result_bf16 = rope.to(torch.bfloat16).float()
                absmax = result_bf16.abs().amax().clamp_min(1.0e-4)
                exponent = torch.ceil(torch.log2(absmax / 448.0))
                scale = torch.pow(2.0, exponent)
                fp8_bytes = torch.clamp(result_bf16 / scale, -448.0,
                                        448.0).to(torch.float8_e4m3fn).view(
                                            torch.uint8)
                token_dst = token_base + torch.arange(
                    head_dim, device=state_cache.device)
                cache_bytes[token_dst] = torch.where(
                    active, fp8_bytes, cache_bytes[token_dst])
                scale_bytes = scale.reshape(1).to(torch.float32).view(
                    torch.uint8)
                scale_dst = scale_base + torch.arange(
                    4, device=state_cache.device)
                cache_bytes[scale_dst] = torch.where(
                    active, scale_bytes, cache_bytes[scale_dst])

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2

        # Store the KV and score (with fused APE addition) in the state.
        # NOTE: PDL is disabled — both this kernel and _fused_kernel below
        # depend on preceding kernel outputs (kv/score from the cublas GEMM;
        # state_cache from this kernel) but neither emits/waits on PDL grid
        # dependency primitives, so launch_pdl=True caused a read-after-write
        # race and non-deterministic output.
        _save_partial_states_kernel[(num_actual,)](
            kv,
            kv.stride(0),
            score,
            score.stride(0),
            self.ape,
            self.ape.stride(0),
            positions,
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            slot_mapping,
            block_size,
            HEAD_SIZE=kv.shape[-1],
            TRITON_BLOCK_SIZE=triton.next_power_of_2(kv.shape[-1]),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            launch_pdl=False,
        )

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        kv_cache = self._static_forward_context[self.k_cache_prefix].kv_cache

        if self._use_torch_fused_fallback(kv):
            self._compress_norm_rope_insert_torch(
                state_cache,
                state_width,
                token_to_req_indices,
                positions,
                slot_mapping,
                block_table,
                block_size,
                cos_sin_cache,
                kv_cache,
                k_cache_metadata.slot_mapping,
            )
            return

        self._fused_kernel[(num_actual,)](
            # state cache
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            # metadata
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_table.stride(0),
            block_size,
            # RMSNorm
            self.norm.weight,
            self.rms_norm_eps,
            # RoPE
            cos_sin_cache,
            cos_sin_cache.stride(0),
            # KV cache
            kv_cache,
            k_cache_metadata.slot_mapping,
            kv_cache.shape[1],  # paged KV cache block size (tokens per block)
            # constexprs
            HEAD_SIZE=self.head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            ROPE_HEAD_DIM=self.rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=self._quant_block,
            TOKEN_STRIDE=self._token_stride,
            SCALE_DIM=self._scale_dim,
            KV_BLOCK_STRIDE=kv_cache.stride(0),
            num_warps=self._num_warps,
            launch_pdl=False,
        )


@triton.jit
def _save_partial_states_kernel(
    kv_ptr,
    kv_stride,
    score_ptr,
    score_stride,
    ape_ptr,
    ape_stride,
    positions_ptr,
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    slot_mapping_ptr,
    block_size,
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    # state_cache last dim packs [kv_state, score_state], each STATE_WIDTH wide.
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)

    # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
    # by vLLM).  During CUDA graph replay the batch may contain padding
    # tokens whose slot_mapping is -1; writing to kv_state[-1] would be an
    # illegal memory access.
    if slot_id < 0:
        return

    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = (
        state_cache_ptr
        + block_idx * state_cache_stride0
        + pos_in_block * state_cache_stride1
    )

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE

    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    # Fused: score += ape[position % compress_ratio]
    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(
        base_ptr + STATE_WIDTH + block,
        score + ape,
        mask=mask,
    )
