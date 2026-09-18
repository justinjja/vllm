# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 indexer numerics, request isolation, and mutable graph inputs."""

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.ampere_mqa import (
    decode_e4m3_bf16,
    decode_e4m3_fp16,
    mqa_logits,
    paged_mqa_logits,
)
from vllm.triton_utils import tl, triton

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.mark.parametrize("heads", [16, 32])
@pytest.mark.parametrize(
    "rows, columns", [(7, 65536), (128, 65536), (512, 65536), (7, 1048576)]
)
def test_candidate_scoring_preserves_dense_masked_logits_and_topk(heads, rows, columns):
    """Keep exact logits and selected indices for packed and partial-block rows."""
    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.attention.dsa.ampere_candidate_mqa import (
        candidate_mqa_logits,
    )
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
        select_candidate_blocks,
    )

    torch.manual_seed(123)
    q = torch.randn(rows, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(columns, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(columns * 2, device="cuda")[::2]
    weights = torch.randn(rows, heads * 2, device="cuda")[:, ::2]
    starts = torch.tensor(
        [0 if i % 2 else columns // 4 for i in range(rows)],
        device="cuda",
        dtype=torch.int32,
    )
    ends = torch.tensor(
        [columns if i % 3 else columns // 2 + 3 for i in range(rows)],
        device="cuda",
        dtype=torch.int32,
    )
    ends[0] = starts[0]
    ends[1] = starts[1] + 1
    candidates = torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
    source_scores = torch.randn(rows, columns, device="cuda")
    select_candidate_blocks(source_scores, starts, ends, 2048, 8, candidates)
    del source_scores
    expected = mqa_logits((q, None), (k, scales), weights, starts, ends)
    apply_candidate_mask(expected, starts, ends, candidates, 8)
    actual = candidate_mqa_logits(
        (q, None), (k, scales), weights, starts, ends, candidates, 8
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    indices = []
    for logits in (expected, actual, expected):
        topk = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
        ops.top_k_per_row_prefill(
            logits, starts, ends, topk, rows, *logits.stride(), 512
        )
        indices.append(topk)
    # The unchanged CUDA selector emits an unsorted list using atomics. Even
    # repeated selection on the same logits may permute its output order.
    for result in indices[1:]:
        torch.testing.assert_close(
            indices[0].sort(dim=-1).values, result.sort(dim=-1).values, rtol=0, atol=0
        )


@triton.jit
def _decode_all(X, Y, FP16: tl.constexpr):
    i = tl.arange(0, 256)
    bits = tl.load(X + i)
    values = decode_e4m3_fp16(bits) if FP16 else decode_e4m3_bf16(bits)
    tl.store(Y + i, values)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_all_e4m3_encodings(dtype):
    """Include subnormals, signed zero, maximum finite values and both NaNs."""
    x = torch.arange(256, device="cuda").to(torch.uint8)
    y = torch.empty(256, device="cuda", dtype=dtype)
    _decode_all[(1,)](x, y, dtype == torch.float16)
    ref = x.view(torch.float8_e4m3fn).to(dtype)
    torch.testing.assert_close(y, ref, rtol=0, atol=0, equal_nan=True)
    assert torch.equal(torch.signbit(y[:255]), torch.signbit(ref[:255]))


def _reference(q, k, s, w):
    scores = torch.einsum("mhd,nd->mhn", q.float(), k.float()).relu()
    return (scores * w[:, :, None]).sum(1) * s[None, :]


@pytest.mark.parametrize("heads", [4, 16, 32])
@pytest.mark.parametrize("tile", [(1, 64, 4), (2, 128, 4), (4, 64, 4)])
@pytest.mark.parametrize("fp16", [False, True])
def test_dense_ragged_rows(heads, tile, fp16):
    """A query tile may span sequences; each output keeps its own KV bounds."""
    torch.manual_seed(9)
    m, n = 7, 197
    q = torch.randn(m, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(n, 128, device="cuda").to(torch.float8_e4m3fn)
    s = torch.rand(n * 2, device="cuda")[::2]
    w = torch.randn(m, heads * 2, device="cuda")[:, ::2]
    starts = torch.tensor(
        [0, 0, 20, 73, 128, 128, 197], device="cuda", dtype=torch.int32
    )
    ends = torch.tensor(
        [0, 1, 63, 110, 129, 197, 197], device="cuda", dtype=torch.int32
    )
    expected = _reference(q, k, s, w)
    col = torch.arange(n, device="cuda")
    expected.masked_fill_((col < starts[:, None]) | (col >= ends[:, None]), -torch.inf)
    result = mqa_logits((q, None), (k, s), w, starts, ends, tile=tile, fp16=fp16)
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-4)


def test_dense_logits_reuses_preallocated_workspace():
    """Growing prefill shapes reuse storage without touching the unused tail."""
    torch.manual_seed(42)
    q = torch.randn(65, 32, 128, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(65536, 128, device="cuda").to(torch.float8_e4m3fn)
    scales = torch.rand(65536, device="cuda")
    weights = torch.randn(65, 32, device="cuda")
    starts = torch.zeros(65, device="cuda", dtype=torch.int32)
    ends = torch.full((65,), 65536, device="cuda", dtype=torch.int32)
    buffer = torch.empty(65 * 65536 + 16, device="cuda")
    for rows, columns in ((1, 197), (65, 65536), (7, 4096)):
        ends[:rows].fill_(columns)
        args = (
            (q[:rows], None),
            (k[:columns], scales[:columns]),
            weights[:rows],
            starts[:rows],
            ends[:rows],
        )
        expected = mqa_logits(*args)
        buffer.fill_(123.0)
        output = buffer[: rows * columns].view(rows, columns)
        torch.accelerator.synchronize()
        allocated = torch.accelerator.memory_allocated()
        torch.accelerator.reset_peak_memory_stats()
        result = mqa_logits(*args, out=output)
        torch.accelerator.synchronize()
        assert torch.accelerator.max_memory_allocated() == allocated
        assert result.data_ptr() == output.data_ptr()
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        assert torch.all(buffer[rows * columns :] == 123.0)


def _paged_case(next_n, heads=16):
    torch.manual_seed(17)
    b, n, page = 3, 197, 64
    blocks = triton.cdiv(n, page)
    k = torch.randn(b * blocks, page, 128, device="cuda").to(torch.float8_e4m3fn)
    s = torch.rand(b * blocks, page, device="cuda")
    # V4.1 packs the indexer beside other caches, giving pages a larger stride.
    storage = torch.full(
        (b * blocks, page * 132 + 128), 0x7F, device="cuda", dtype=torch.uint8
    )
    cache = storage[:, : page * 132].view(b * blocks, page, 1, 132)
    flat = cache.view(b * blocks, -1)
    flat[:, : page * 128] = k.view(torch.uint8).reshape(b * blocks, -1)
    flat[:, page * 128 :] = s.view(torch.uint8).reshape(b * blocks, -1)
    table = torch.randperm(b * blocks, device="cuda", dtype=torch.int32).view(b, blocks)
    # Non-unit column stride and a nontrivial query-to-request permutation.
    table = torch.stack((table, table), -1)[:, :, 0]
    indices = torch.tensor([2, 0, 1], device="cuda", dtype=torch.int32)
    q = torch.randn(b, next_n, heads, 128, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(b * next_n, heads, device="cuda")
    lens = torch.arange(b * next_n, device="cuda", dtype=torch.int32).view(b, next_n)
    lens = (lens * 17 + 1).clamp(max=n)
    lens[0, 0] = 0
    return q, k, s, cache, table, indices, w, lens, n


def _paged_reference(q, k, s, table, indices, w, lens, n):
    b, next_n, h, d = q.shape
    expected = []
    for i in range(b):
        pages = table[indices[i]].long()
        keys = k[pages].reshape(-1, d)[:n]
        scales = s[pages].reshape(-1)[:n]
        result = _reference(q[i], keys, scales, w[i * next_n : (i + 1) * next_n])
        result.masked_fill_(
            torch.arange(n, device="cuda") >= lens[i, :, None], -torch.inf
        )
        expected.append(result)
    return torch.cat(expected)


@pytest.mark.parametrize("next_n", [1, 3, 6])
@pytest.mark.parametrize("tile", [(1, 64, 4), (2, 128, 4), (4, 64, 4)])
@pytest.mark.parametrize("fp16", [False, True])
def test_paged_request_isolation_and_graph_refresh(next_n, tile, fp16):
    """Sharing KV must respect page permutations, tail rows and updated lengths."""
    q, k, s, cache, table, indices, w, lens, n = _paged_case(next_n)

    def run():
        return paged_mqa_logits(
            (q, None),
            cache,
            w,
            lens,
            table,
            None,
            n,
            indices=indices,
            tile=tile,
            fp16=fp16,
        )

    expected = _paged_reference(q, k, s, table, indices, w, lens, n)
    torch.testing.assert_close(run(), expected, rtol=2e-5, atol=2e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    table.copy_(table.flip(0))
    lens.fill_(n - 7)
    indices.copy_(indices.roll(1))
    graph.replay()
    expected = _paged_reference(q, k, s, table, indices, w, lens, n)
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-4)


@pytest.mark.parametrize("fp16", [False, True])
def test_paged_long_allocation_short_rows_graph_refresh(fp16):
    """A large captured window must mask empty tiles and read updated lengths."""
    q, k, s, cache, table, indices, w, lens, n = _paged_case(3)
    columns = 262144

    def run():
        return paged_mqa_logits(
            (q, None),
            cache,
            w,
            lens,
            table,
            None,
            columns,
            indices=indices,
            fp16=fp16,
        )

    def expected():
        result = torch.full((w.shape[0], columns), -torch.inf, device="cuda")
        result[:, :n] = _paged_reference(q, k, s, table, indices, w, lens, n)
        return result

    torch.testing.assert_close(run(), expected(), rtol=2e-5, atol=2e-4)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    table.copy_(table.flip(0))
    indices.copy_(indices.roll(1))
    lens.fill_(n - 1)
    graph.replay()
    torch.testing.assert_close(result, expected(), rtol=2e-5, atol=2e-4)
    lens.zero_()
    graph.replay()
    assert torch.isneginf(result).all()


def test_paged_request_lengths_broadcast_across_queries():
    q, k, s, cache, table, indices, w, lens, n = _paged_case(6)
    lens = lens[:, 0].contiguous()
    result = paged_mqa_logits(
        (q, None), cache, w, lens, table, None, n, indices=indices, tile=(4, 64, 4)
    )
    expected = _paged_reference(
        q, k, s, table, indices, w, lens[:, None].expand(-1, 6), n
    )
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-4)


def test_vllm_dispatch_uses_sm80_indexer_without_deepgemm():
    """Exercise the serving-facing wrapper, including paged table strides."""
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 dispatch test")
    from vllm.utils.deep_gemm import fp8_fp4_mqa_logits, fp8_fp4_paged_mqa_logits

    q, k, s, cache, table, indices, w, lens, n = _paged_case(6)
    result = fp8_fp4_paged_mqa_logits(
        (q, None), cache, w, lens, table, None, n, True, indices=indices
    )
    expected = _paged_reference(q, k, s, table, indices, w, lens, n)
    torch.testing.assert_close(result, expected, rtol=2e-5, atol=2e-4)
    q = q[0]
    keys = k.reshape(-1, 128)[:n]
    scales = s.reshape(-1)[:n]
    starts = torch.zeros(6, device="cuda", dtype=torch.int32)
    ends = torch.full_like(starts, n)
    result = fp8_fp4_mqa_logits((q, None), (keys, scales), w[:6], starts, ends, True)
    torch.testing.assert_close(
        result, _reference(q, keys, scales, w[:6]), rtol=2e-5, atol=2e-4
    )
