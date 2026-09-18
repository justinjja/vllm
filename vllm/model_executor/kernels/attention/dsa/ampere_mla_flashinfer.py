# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Graph-safe FA2 MLA over gathered V4.1 sparse keys on Ampere."""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.kernels.attention.dsa.ampere_sparse_mla import _load_keys
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper


@triton.jit
def _gather(
    CACHE,
    INDICES,
    LENGTHS,
    KV,
    MASK,
    PAGE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    ROWS: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    OFFSET: tl.constexpr,
    RECORD: tl.constexpr,
    BLOCK: tl.constexpr = 4,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    dim = tl.arange(0, 512)
    value, valid = _load_keys(
        CACHE,
        INDICES,
        LENGTHS,
        row,
        col,
        dim,
        WIDTH,
        INDEX_STRIDE,
        PAGE,
        CACHE_STRIDE,
        ROWS,
        RECORD,
    )
    dst = row * CAPACITY + OFFSET + col
    tl.store(KV + dst[:, None] * 512 + dim[None, :], value, col[:, None] < WIDTH)
    mask_dim = tl.arange(0, 64)
    additive_mask = tl.where(valid, 0.0, -1e30)
    tl.store(
        MASK + dst[:, None] * 64 + mask_dim[None, :],
        tl.where(mask_dim[None, :] == 0, additive_mask[:, None], 0.0),
        col[:, None] < WIDTH,
    )


@triton.jit
def _sink_scale(
    X,
    LSE,
    SINK,
    OUT,
    HEADS: tl.constexpr,
    OUT_ROW: tl.constexpr,
    OUT_HEAD: tl.constexpr,
    HAS_SINK: tl.constexpr,
):
    row, head = tl.program_id(0), tl.program_id(1)
    dim = tl.arange(0, 512)
    lse = tl.load(LSE + row * HEADS + head)
    sink = tl.load(SINK + head) if HAS_SINK else -float("inf")
    factor = 1 / (1 + tl.exp(sink - lse))
    value = tl.load(X + (row * HEADS + head) * 512 + dim).to(tl.float32)
    tl.store(OUT + row * OUT_ROW + head * OUT_HEAD + dim, value * factor)


class SparseMLAWorkspace:
    """One compute-stream owner, shared by sequential target/draft layers.

    FA2 plans contain host-derived lengths. Keep them fixed, and encode sparse
    validity in a synthetic extra Q/K coordinate instead. The actual 512-D
    query, key and value (including the model's RoPE coordinates) are unchanged.
    A per-head LSE correction includes the model's zero-valued attention sink.
    """

    def __init__(self, max_rows: int, device: torch.device):
        self.max_rows = max_rows
        self.device = device
        self.float_workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=device
        )
        self.kv = torch.empty(
            max_rows * 640, 1, 512, dtype=torch.bfloat16, device=device
        )
        self.mask = torch.empty(
            max_rows * 640, 1, 64, dtype=torch.bfloat16, device=device
        )
        self.query_mask = torch.zeros(
            max_rows, 64, 64, dtype=torch.bfloat16, device=device
        )
        self.query_mask[:, :, 0] = 1
        self.result = torch.empty(
            max_rows * 64 * 512, dtype=torch.bfloat16, device=device
        )
        self.lse = torch.empty(max_rows * 64, dtype=torch.float32, device=device)
        self.plans: dict[tuple, BatchMLAPagedAttentionWrapper] = {}

    def run(
        self,
        q,
        main_cache,
        main_indices,
        main_lengths,
        extra_cache,
        extra_indices,
        extra_lengths,
        sink,
        scale,
        out,
    ):
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        rows, heads, dim = q.shape
        main_indices = main_indices.reshape(rows, -1)
        main_width = main_indices.shape[1]
        extra_width = 0
        if extra_cache is not None:
            extra_indices = extra_indices.reshape(rows, -1)
            extra_width = extra_indices.shape[1]
        capacity = main_width + extra_width
        assert rows <= self.max_rows and heads <= 64 and dim == 512
        assert 0 < capacity <= 640 and q.dtype == torch.bfloat16
        key = (rows, heads, capacity, scale)
        if key not in self.plans:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Sparse FA2 plan must be warmed before graph capture"
                )
            qo = torch.arange(rows + 1, dtype=torch.int32)
            ki = qo * capacity
            ix = torch.arange(rows * capacity, dtype=torch.int32)
            lengths = torch.full((rows,), capacity, dtype=torch.int32)
            wrapper = BatchMLAPagedAttentionWrapper(
                self.float_workspace,
                use_cuda_graph=True,
                qo_indptr=qo.to(self.device),
                kv_indptr=ki.to(self.device),
                kv_indices=ix.to(self.device),
                kv_len_arr=lengths.to(self.device),
                backend="fa2",
            )
            wrapper.plan(
                qo, ki, ix, lengths, heads, 512, 64, 1, False, scale, q.dtype, q.dtype
            )
            from .ampere_mla_flashinfer_jit import get_noncooperative_mla_module

            wrapper._cached_module = get_noncooperative_mla_module()
            self.plans[key] = wrapper
        wrapper = self.plans[key]
        kv = self.kv[: rows * capacity]
        mask = self.mask[: rows * capacity]
        _gather[(rows, triton.cdiv(main_width, 4))](
            main_cache,
            main_indices,
            main_lengths,
            kv,
            mask,
            main_cache.shape[1] if main_cache.dtype == torch.uint8 else 1,
            main_cache.stride(0),
            main_cache.shape[0]
            * (main_cache.shape[1] if main_cache.dtype == torch.uint8 else 1),
            main_indices.stride(0),
            main_width,
            capacity,
            0,
            main_cache.shape[-1] if main_cache.dtype == torch.uint8 else 0,
        )
        if extra_cache is not None:
            _gather[(rows, triton.cdiv(extra_width, 4))](
                extra_cache,
                extra_indices,
                extra_lengths,
                kv,
                mask,
                extra_cache.shape[1] if extra_cache.dtype == torch.uint8 else 1,
                extra_cache.stride(0),
                extra_cache.shape[0]
                * (extra_cache.shape[1] if extra_cache.dtype == torch.uint8 else 1),
                extra_indices.stride(0),
                extra_width,
                capacity,
                main_width,
                extra_cache.shape[-1] if extra_cache.dtype == torch.uint8 else 0,
            )
        result = self.result[: rows * heads * 512].view(rows, heads, 512)
        lse = self.lse[: rows * heads].view(rows, heads)
        wrapper.run(
            q,
            self.query_mask[:rows, :heads],
            kv,
            mask,
            out=result,
            lse=lse,
            return_lse=True,
            return_lse_base_on_e=True,
        )
        _sink_scale[(rows, heads)](
            result,
            lse,
            sink,
            out,
            heads,
            out.stride(0),
            out.stride(1),
            sink is not None,
        )


_workspaces: dict[tuple, SparseMLAWorkspace] = {}


def get_sparse_mla_workspace(max_rows: int, device: torch.device):
    # DBO/multi-stream execution requires separate owners; the opt-in caller
    # rejects DBO. Layers and graph sizes on this compute stream share scratch.
    key = (device, torch.cuda.current_stream(device).cuda_stream, max_rows)
    if key not in _workspaces:
        _workspaces[key] = SparseMLAWorkspace(max_rows, device)
    return _workspaces[key]
