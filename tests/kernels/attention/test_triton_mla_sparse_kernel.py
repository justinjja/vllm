# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the Triton sparse MLA kernel.

Compares split-KV against the single-pass (`num_kv_splits=1`) path
produced by the same kernel — both paths must agree to within bf16 ULPs.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    triton_mla_sparse_attention,
)

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="Triton sparse MLA kernel requires CUDA/ROCm",
)


@pytest.mark.parametrize("num_heads", [8, 32, 64])
@pytest.mark.parametrize("splits", [1, 4, 16])
@pytest.mark.parametrize(
    "cache_dtype,scale_value",
    [
        (torch.bfloat16, 1.0),
        (torch.float8_e4m3fn, 0.125),
        (torch.float8_e4m3fn, 1.7),
    ],
)
def test_cache_matches_independent_attention(
    num_heads, splits, cache_dtype, scale_value
):
    """Sparse attention matches a dense reference, including empty queries."""
    set_random_seed(170)
    q = torch.randn(3, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    # A wider backing allocation exercises noncontiguous physical cache rows.
    storage = torch.randn(128, 2, _DIM_QK, device="cuda")
    cache = (storage / scale_value).to(cache_dtype)[:, :1]
    scale = torch.tensor([scale_value], dtype=torch.float32, device="cuda")
    indices = torch.full((3, 1, 2048), -1, dtype=torch.int32, device="cuda")
    indices[0, 0, :37] = torch.arange(37, device="cuda")
    indices[1, 0, 100:173] = torch.arange(40, 113, device="cuda")
    output = triton_mla_sparse_attention(
        q, cache, indices, sm_scale=0.04, num_kv_splits=splits, kv_scale=scale
    )
    expected = torch.zeros_like(output, dtype=torch.float32)
    decoded = (cache.float() * scale).to(torch.bfloat16).float()
    for row in range(2):
        selected = indices[row, 0]
        selected = selected[selected >= 0].long()
        keys = decoded[selected, 0]
        scores = (q[row].float() @ keys.T) * 0.04
        expected[row] = scores.softmax(-1) @ keys[:, :512]
    torch.testing.assert_close(output.float(), expected, atol=0.02, rtol=0.02)
    assert torch.equal(output[2], torch.zeros_like(output[2]))


@pytest.fixture(scope="module")
def kv_cache():
    set_random_seed(0)
    return torch.randn(32768, 1, _DIM_QK, dtype=torch.bfloat16, device="cuda")


def _assert_split_matches_single_pass(
    num_tokens: int,
    num_heads: int,
    topk: int,
    num_kv_splits: int | None,
    kv_cache: torch.Tensor,
) -> None:
    set_random_seed(0)
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.randint(
        0, kv_cache.shape[0], (num_tokens, 1, topk), dtype=torch.int32, device="cuda"
    )
    out_ref = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=1,
    )
    out = triton_mla_sparse_attention(
        q,
        kv_cache,
        indices,
        sm_scale=0.1,
        num_kv_splits=num_kv_splits,
    )
    torch.testing.assert_close(
        out.float(),
        out_ref.float(),
        atol=5e-2,
        rtol=5e-3,
    )


@pytest.mark.parametrize(
    "num_tokens,num_heads",
    [(1, 16), (1, 128), (8, 32), (32, 128), (128, 16)],
)
@pytest.mark.parametrize("topk", [768, 1024, 2048, 4096])
@pytest.mark.parametrize("num_kv_splits", [2, 4, 8])
def test_split_kv_matches_single_pass(
    num_tokens, num_heads, topk, num_kv_splits, kv_cache
):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads,
        topk,
        num_kv_splits,
        kv_cache,
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_auto_split_matches_single_pass(num_tokens, kv_cache):
    _assert_split_matches_single_pass(
        num_tokens,
        num_heads=128,
        topk=2048,
        num_kv_splits=None,
        kv_cache=kv_cache,
    )


@pytest.mark.parametrize("num_kv_splits", [1, 2, 4, 8])
def test_short_prefill_no_nan(num_kv_splits, kv_cache):
    """Regression: short prefill where most topk slots are -1 sentinels.

    The indexer fills 2048 topk positions with only a handful of valid
    indices; the rest are -1. Before the NEG_LARGE sentinel fix, the online
    softmax produced NaN via `max(-inf, -inf) = -inf` and
    `exp2(-inf − -inf) = NaN`, poisoning every split.
    """
    set_random_seed(0)
    num_tokens, num_heads, topk = 5, 16, 2048
    q = torch.randn(num_tokens, num_heads, _DIM_QK, dtype=torch.bfloat16, device="cuda")
    indices = torch.full((num_tokens, 1, topk), -1, dtype=torch.int32, device="cuda")
    # Only the first `t+1` slots of each query hold valid indices; the
    # remaining ~2045 slots are -1, producing many all-invalid BLOCK_N tiles.
    for t in range(num_tokens):
        indices[t, 0, : t + 1] = torch.arange(
            64, 64 + t + 1, dtype=torch.int32, device="cuda"
        )
    out = triton_mla_sparse_attention(
        q, kv_cache, indices, sm_scale=0.0417, num_kv_splits=num_kv_splits
    )
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
