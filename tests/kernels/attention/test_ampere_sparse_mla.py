# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 sparse MLA against an independent FP32 softmax, including graph replay."""

import pytest
import torch

from vllm.model_executor.kernels.attention.dsa.ampere_sparse_mla import sparse_mla

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_cache(record):
    rows, page = 80, 16
    values = torch.randn(rows, 512, device="cuda", dtype=torch.bfloat16)
    if record == 0:
        return values, values.float()
    fp8_dims, group, data_bytes, scale_bytes = (
        (512, 32, 512, 16) if record == 528 else (448, 64, 576, 8)
    )
    exponents = torch.randint(
        123, 130, (rows, fp8_dims // group), device="cuda", dtype=torch.uint8
    )
    scales = (exponents.float() - 127).exp2().repeat_interleave(group, -1)
    quant = (values[:, :fp8_dims].float() / scales).to(torch.float8_e4m3fn)
    expected = values.float()
    expected[:, :fp8_dims] = (quant.float() * scales).to(torch.bfloat16).float()
    backing = torch.full(
        (rows // page, page * record + 128), 0x7F, device="cuda", dtype=torch.uint8
    )
    cache = backing[:, : page * record].view(rows // page, page, record)
    flat = cache.view(rows // page, -1)
    data = flat[:, : page * data_bytes].view(rows // page, page, data_bytes)
    data[:, :, :fp8_dims] = quant.view(torch.uint8).view(rows // page, page, fp8_dims)
    if record == 584:
        data[:, :, 448:] = (
            values[:, 448:].contiguous().view(torch.uint8).view(rows // page, page, 128)
        )
    encoded = flat[:, page * data_bytes :].view(rows // page, page, scale_bytes)
    encoded.zero_()
    encoded[:, :, : fp8_dims // group] = exponents.view(
        rows // page, page, fp8_dims // group
    )
    return cache, expected


def reference(q, caches, index_sets, lengths, sink, scale):
    out = []
    for row in range(q.shape[0]):
        keys = []
        for cache, indices, lens in zip(caches, index_sets, lengths):
            ids = indices[row, : max(0, int(lens[row]))]
            ids = ids[(ids >= 0) & (ids < cache.shape[0])]
            keys.append(cache[ids])
        kv = torch.cat(keys).float()
        logits = q[row].float() @ kv.T * scale
        if sink is not None:
            logits = torch.cat((logits, sink[:, None]), -1)
            kv = torch.cat((kv, torch.zeros(1, 512, device="cuda")))
        out.append(
            logits.softmax(-1) @ kv if kv.shape[0] else torch.zeros_like(q[row]).float()
        )
    return torch.stack(out)


@pytest.mark.parametrize(
    "records", [(0, None), (584, None), (584, 584), (528, 528), (584, 528)]
)
@pytest.mark.parametrize("heads,splits", [(8, 1), (16, 3), (32, 16), (64, None)])
@pytest.mark.parametrize("use_sink", [False, True])
@pytest.mark.parametrize("kernel", ["triton", "triton_trim", "flashinfer"])
def test_sparse_attention_masks_packed_pages_and_refreshes_graph(
    records, heads, splits, use_sink, kernel, block_h=16, block_n=32
):
    torch.manual_seed(170)
    rows = 3
    q = torch.randn(rows, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.25
    sink = torch.linspace(-2, 2, heads, device="cuda") if use_sink else None
    scale = 512**-0.5
    caches, refs, indices, lengths = [], [], [], []
    for record in records:
        if record is None:
            continue
        cache, ref = make_cache(record)
        caches.append(cache)
        refs.append(ref)
        ids = torch.randint(-2, 84, (rows, 71), device="cuda", dtype=torch.int32)
        indices.append(ids[:, :67])
        lengths.append(torch.tensor([0, 19, 67], device="cuda", dtype=torch.int32))
    storage = torch.full(
        (rows, heads + 2, 520), 3.0, device="cuda", dtype=torch.bfloat16
    )
    output = storage[:, :heads, :512]
    if kernel == "flashinfer":
        from vllm.model_executor.kernels.attention.dsa.ampere_mla_flashinfer import (
            get_sparse_mla_workspace,
        )

        workspace = get_sparse_mla_workspace(rows, q.device)

    def run():
        if kernel == "flashinfer":
            workspace.run(
                q,
                caches[0],
                indices[0],
                lengths[0],
                caches[1] if len(caches) == 2 else None,
                indices[1] if len(caches) == 2 else None,
                lengths[1] if len(caches) == 2 else None,
                sink,
                scale,
                output,
            )
            return
        sparse_mla(
            q,
            caches[0],
            indices[0],
            lengths[0],
            scale,
            sink,
            output,
            caches[1] if len(caches) == 2 else None,
            indices[1] if len(caches) == 2 else None,
            lengths[1] if len(caches) == 2 else None,
            splits=splits,
            block_h=block_h,
            block_n=block_n,
            num_warps=8 if block_h > 16 else None,
            trim_empty_tiles=kernel == "triton_trim",
        )

    run()
    expected = reference(q, refs, indices, lengths, sink, scale)
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.002)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    q.neg_()
    for ids, lens in zip(indices, lengths):
        ids.copy_(ids.flip(0))
        lens.copy_(torch.tensor([23, 0, 43], device="cuda", dtype=torch.int32))
    graph.replay()
    expected = reference(q, refs, indices, lengths, sink, scale)
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.002)
    assert (storage[:, heads:] == 3).all() and (storage[:, :, 512:] == 3).all()


@pytest.mark.parametrize("block_h,block_n", [(32, 32), (64, 32), (32, 64)])
def test_dense_prefill_head_tiles_preserve_masks_and_graph_updates(block_h, block_n):
    # 48 heads exercises the partial final head tile for both wider choices.
    test_sparse_attention_masks_packed_pages_and_refreshes_graph(
        (0, None), 48, 1, True, "triton", block_h=block_h, block_n=block_n
    )
