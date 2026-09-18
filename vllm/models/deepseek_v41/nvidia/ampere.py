# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse attention and grouped output projection on SM80."""

import torch

from vllm.config import get_current_vllm_config
from vllm.model_executor.kernels.attention.dsa.ampere_sparse_mla import sparse_mla
from vllm.model_executor.kernels.linear.mxfp8.emulation import (
    EmulationMxfp8LinearKernel,
)
from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
    Mxfp8LinearLayerConfig,
)
from vllm.models.deepseek_v41.attention import DeepseekV4Attention
from vllm.models.deepseek_v41.common.ops import compute_global_topk_indices_and_lens
from vllm.models.deepseek_v41.nvidia.flashmla import DeepseekV4FlashMLAAttention
from vllm.models.deepseek_v41.sparse_mla import DeepseekV4FlashMLABackend
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import rocm_inv_rope_einsum


class DeepseekV41AmpereBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability == DeviceCapability(8, 0)


class DeepseekV41AmpereAttention(DeepseekV4FlashMLAAttention):
    backend_cls = DeepseekV41AmpereBackend

    def __init__(self, *args, **kwargs):
        DeepseekV4Attention.__init__(self, *args, **kwargs)
        config = get_current_vllm_config()
        additional = config.additional_config
        choice = (
            additional.get("cmp_sparse_mla_backend", "triton")
            if isinstance(additional, dict)
            else "triton"
        )
        if choice not in ("triton", "flashinfer", "hybrid"):
            raise ValueError(
                "cmp_sparse_mla_backend must be triton, flashinfer or hybrid"
            )
        self.sparse_mla_choice = choice
        self.wide_prefill_heads = (
            additional.get("cmp_wide_prefill_heads", False)
            if isinstance(additional, dict)
            else False
        )
        if type(self.wide_prefill_heads) is not bool:
            raise ValueError("cmp_wide_prefill_heads must be a boolean")
        self.trim_empty_tiles = (
            additional.get("cmp_trim_empty_attention_tiles", False)
            if isinstance(additional, dict)
            else False
        )
        if type(self.trim_empty_tiles) is not bool:
            raise ValueError("cmp_trim_empty_attention_tiles must be a boolean")
        self.use_sparse_fa2 = choice != "triton"
        if self.use_sparse_fa2 and config.parallel_config.use_ubatching:
            raise ValueError("CMP sparse FlashInfer does not support microbatching")
        drafts = (
            config.speculative_config.num_speculative_tokens
            if config.speculative_config
            else 0
        )
        self.sparse_fa2_max_rows = config.scheduler_config.max_num_seqs * (drafts + 1)
        self.sparse_fa2_workspace = None
        # Grouped WO_A reads ordinary [group, rank, hidden] weights. Dequantize
        # this small projection once; other linears retain their Marlin kernels.
        self.wo_a.is_bmm = False
        if hasattr(self.wo_a.quant_method, "kernel"):
            self.wo_a.quant_method.kernel = EmulationMxfp8LinearKernel(
                Mxfp8LinearLayerConfig()
            )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def _sparse_prefill(self, q, kv, indices, sm_scale, attn_sink, topk_length, out):
        launch = {}
        if self.wide_prefill_heads and q.shape[0] >= 128 and q.shape[1] in (32, 64):
            launch = {"block_h": 32, "block_n": 64, "num_warps": 8}
        sparse_mla(
            q,
            kv.squeeze(1),
            indices,
            topk_length,
            sm_scale,
            attn_sink,
            out,
            trim_empty_tiles=self.trim_empty_tiles,
            **launch,
        )

    def _forward_decode(
        self, q, kv_cache, swa_metadata, attn_metadata, swa_only, output
    ):
        indices = lengths = None
        if not swa_only:
            assert attn_metadata is not None and self.topk_indices_buffer is not None
            n = swa_metadata.num_decode_tokens
            indices, lengths = compute_global_topk_indices_and_lens(
                self.topk_indices_buffer[:n],
                swa_metadata.token_to_req_indices,
                attn_metadata.block_table[: swa_metadata.num_decodes],
                attn_metadata.block_size // self.compress_ratio,
                swa_metadata.is_valid_token[:n],
            )
        if (
            self.use_sparse_fa2
            and not swa_only
            and q.shape[0] <= self.sparse_fa2_max_rows
            and (
                self.sparse_mla_choice == "flashinfer"
                or (q.shape[1] == 64 and q.shape[0] >= 12)
            )
        ):
            from vllm.model_executor.kernels.attention.dsa import (
                ampere_mla_flashinfer,
            )

            if self.sparse_fa2_workspace is None:
                self.sparse_fa2_workspace = (
                    ampere_mla_flashinfer.get_sparse_mla_workspace(
                        self.sparse_fa2_max_rows, q.device
                    )
                )
            self.sparse_fa2_workspace.run(
                q,
                self.swa_cache_layer.kv_cache,
                swa_metadata.decode_swa_indices,
                swa_metadata.decode_swa_lens,
                kv_cache,
                indices,
                lengths,
                self.attn_sink,
                self.scale,
                output,
            )
            return
        sparse_mla(
            q,
            self.swa_cache_layer.kv_cache,
            swa_metadata.decode_swa_indices,
            swa_metadata.decode_swa_lens,
            self.scale,
            self.attn_sink,
            output,
            None if swa_only else kv_cache,
            indices,
            lengths,
            trim_empty_tiles=self.trim_empty_tiles,
        )
