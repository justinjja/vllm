# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton sparse MLA with BF16 or software-decoded FP8 KV on Ampere."""

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
    SparseMLACommonMetadata,
    SparseMLACommonMetadataBuilder,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import AttentionBackend, AttentionCGSupport, MultipleOf
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)


@dataclass
class TritonMLASparseMetadata(SparseMLACommonMetadata):
    pass


class TritonMLASparseMetadataBuilder(
    SparseMLACommonMetadataBuilder[TritonMLASparseMetadata]
):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    metadata_cls = TritonMLASparseMetadata

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)


class TritonMLASparseImpl(SparseMLACommonImpl[TritonMLASparseMetadata]):
    """Sparse MLA with split-KV decode and padded cache support."""

    # Decomposed BF16 MHA changes GLM outputs relative to absorbed sparse MLA.
    supports_dense_mha_prefill = False
    supports_pcp = False
    supports_dcp = False

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune()

    def _warmup_autotune(self) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        q = torch.empty(1, self.num_heads, _DIM_QK, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, _DIM_QK, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        scale = torch.ones(1, dtype=torch.float32, device=device)
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            kv = torch.zeros(64, 1, _DIM_QK, dtype=torch.uint8, device=device).view(
                torch.float8_e4m3fn
            )
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
                kv_scale=scale,
            )

    def record_logical_topk_ready(self) -> None:
        # Index selection and attention execute on the same stream.
        pass

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        from vllm.v1.attention.backends.mla.sparse_utils import (
            flat_kv_row_view,
            triton_convert_req_index_to_global_index,
        )

        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[: q.shape[0]]
        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: q.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )
        assert isinstance(indices, torch.Tensor)
        output = triton_mla_sparse_attention(
            q,
            kv_rows.unsqueeze(1),
            indices.unsqueeze(1),
            sm_scale=self.scale,
            sm_count=self._sm_count,
            kv_scale=layer._k_scale,
        )
        return output, None


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type[TritonMLASparseMetadata]:
        return TritonMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_DIM_QK]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 8
