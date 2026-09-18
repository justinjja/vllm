# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise eager cache snapshots through multiple actual process boundaries."""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.models.deepseek_v41.common.pipeline_transfer import (
    restore_cache_blocks,
    snapshot_cache_blocks,
    snapshot_updated_cache_blocks,
)


def _relay_worker(rank, rendezvous):
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=30),
    )
    try:
        expected = torch.arange(6 * 3 * 4, dtype=torch.uint8).reshape(6, 3, 4)
        cache = expected.clone() if rank == 0 else torch.zeros_like(expected)
        table = torch.tensor([[1, 4, -1]])
        if rank:
            ids = torch.empty(2, dtype=torch.int64)
            blocks = torch.empty(2, 3, 4, dtype=torch.uint8)
            dist.recv(ids, src=rank - 1)
            dist.recv(blocks, src=rank - 1)
            restore_cache_blocks(cache, ids, blocks)
        torch.testing.assert_close(cache[[1, 4]], expected[[1, 4]], rtol=0, atol=0)
        if rank < 2:
            ids, blocks = snapshot_cache_blocks(cache, [table], max_bytes=1024)
            dist.send(ids, dst=rank + 1)
            dist.send(blocks, dst=rank + 1)
    finally:
        dist.destroy_process_group()


def test_cache_blocks_survive_two_pipeline_hops(tmp_path):
    mp.spawn(_relay_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=3)


def _mixed_relay_worker(rank, port, asynchronous, incremental=False):
    from tests.utils import init_test_distributed_environment
    from vllm.distributed import get_pp_group
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(1, 4, rank, port, local_rank=rank)
    group = get_pp_group()
    try:
        cache = torch.full((6, 2, 3, 4), 17, device="cuda", dtype=torch.uint8)[:, 0]
        mirrored = torch.zeros(6, dtype=torch.bool)
        # Distinct rounds refresh a prefix block and then reuse a physical block.
        for round_id, selected in enumerate(([1, 4], [1, 4], [4], [])):
            expected = (
                torch.arange(6 * 2 * 3 * 4, device="cuda")
                .reshape(6, 2, 3, 4)
                .add(round_id * 19)
                .to(torch.uint8)[:, 0]
            )
            if not incremental:
                cache.fill_(17)
            if rank == 0:
                cache[selected] = expected[selected]
            else:
                if asynchronous:
                    payload, handles, postprocess = group.irecv_tensor_dict(
                        src=rank - 1
                    )
                    for handle in handles:
                        handle.wait()
                    for callback in postprocess:
                        callback()
                else:
                    payload = group.recv_tensor_dict(src=rank - 1)
                assert payload["ids"].device.type == "cpu"
                assert payload["blocks"].device.type == "cuda"
                assert payload["round"] == round_id
                restore_cache_blocks(cache, payload["ids"], payload["blocks"])
                untouched = (
                    [0, 2, 3, 5]
                    if incremental
                    else [i for i in range(6) if i not in selected]
                )
                assert torch.all(cache[untouched] == 17)
            torch.testing.assert_close(
                cache[selected], expected[selected], rtol=0, atol=0
            )
            if rank < 3:
                table = torch.tensor([selected + [-1]], dtype=torch.int32)
                if incremental and rank == 0:
                    ids, blocks = snapshot_updated_cache_blocks(
                        cache,
                        [torch.tensor([[1, 4, -1]])],
                        torch.tensor([b * 3 for b in selected], device="cuda"),
                        3,
                        mirrored,
                        max_bytes=1024,
                    )
                elif incremental:
                    ids, blocks = payload["ids"], payload["blocks"]
                else:
                    ids, blocks = snapshot_cache_blocks(cache, [table], max_bytes=1024)
                payload = {"ids": ids, "blocks": blocks, "round": round_id}
                if asynchronous:
                    handles = group.isend_tensor_dict(payload, dst=rank + 1)
                    for handle in handles:
                        handle.wait()
                    group._reap_completed_isends()
                    # The CPU Gloo handle must also tolerate a caller's second wait.
                    for handle in handles:
                        handle.wait()
                else:
                    group.send_tensor_dict(payload, dst=rank + 1)
            group.barrier()
            if incremental:
                # Compare the complete persistent replica with the source,
                # including cached blocks omitted from this round's payload.
                reference = cache.clone()
                dist.broadcast(reference, src=0)
                torch.testing.assert_close(cache, reference, rtol=0, atol=0)
        torch.accelerator.synchronize()
    finally:
        cleanup_dist_env_and_memory()


def test_host_ids_and_cuda_cache_survive_three_pipeline_hops():
    import pytest

    from vllm.utils.network_utils import get_open_port

    if not torch.cuda.is_available() or torch.accelerator.device_count() < 4:
        pytest.skip("Four CUDA devices required")
    for asynchronous in (False, True):
        for incremental in (False, True):
            mp.spawn(
                _mixed_relay_worker,
                args=(str(get_open_port()), asynchronous, incremental),
                nprocs=4,
            )


def _prefill_tp_worker(rank, port):
    from tests.utils import init_test_distributed_environment
    from vllm import _custom_ops as ops
    from vllm.distributed.parallel_state import cleanup_dist_env_and_memory
    from vllm.model_executor.kernels.attention.dsa.ampere_mqa import mqa_logits
    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        select_candidate_blocks,
    )
    from vllm.model_executor.layers.sparse_attn_indexer import (
        ampere_sharded_prefill_topk,
    )

    torch.accelerator.set_device_index(rank)
    init_test_distributed_environment(2, 1, rank, port, local_rank=rank)
    try:
        for rows in (65, 128):
            for heads in (16, 32):
                for write_candidates in (False, True):
                    torch.manual_seed(123)
                    columns = 131072
                    q = torch.randn(rows, heads, 128, device="cuda").to(
                        torch.float8_e4m3fn
                    )
                    k = torch.randn(columns, 128, device="cuda").to(torch.float8_e4m3fn)
                    scales = torch.rand(columns, device="cuda")
                    weights = torch.randn(rows, heads, device="cuda")
                    starts = torch.zeros(rows, device="cuda", dtype=torch.int32)
                    starts[::2] = columns // 4
                    ends = torch.full(
                        (rows,), columns, device="cuda", dtype=torch.int32
                    )
                    ends[0] = starts[0]
                    ends[1] = starts[1] + 1
                    expected = torch.full(
                        (rows, 512), -1, device="cuda", dtype=torch.int32
                    )
                    actual = torch.empty_like(expected)
                    logits = mqa_logits((q, None), (k, scales), weights, starts, ends)
                    candidates = (
                        torch.empty(rows, 2048, device="cuda", dtype=torch.int32)
                        if write_candidates
                        else None
                    )
                    reference_candidates = (
                        torch.empty_like(candidates) if candidates is not None else None
                    )
                    if write_candidates:
                        assert (
                            candidates is not None and reference_candidates is not None
                        )
                        select_candidate_blocks(
                            logits, starts, ends, 2048, 8, reference_candidates
                        )
                    ops.top_k_per_row_prefill(
                        logits,
                        starts,
                        ends,
                        expected,
                        rows,
                        logits.stride(0),
                        logits.stride(1),
                        512,
                    )
                    ampere_sharded_prefill_topk(
                        q,
                        k,
                        scales,
                        weights,
                        starts,
                        ends,
                        actual,
                        candidates,
                        8,
                        logits_buffer=(
                            torch.empty(((rows + 1) // 2) * columns, device="cuda")
                            if not write_candidates
                            else None
                        ),
                    )
                    torch.testing.assert_close(
                        actual.sort(-1).values, expected.sort(-1).values, rtol=0, atol=0
                    )
                    if write_candidates:
                        assert (
                            candidates is not None and reference_candidates is not None
                        )
                        torch.testing.assert_close(
                            candidates.sort(-1).values,
                            reference_candidates.sort(-1).values,
                            rtol=0,
                            atol=0,
                        )
    finally:
        cleanup_dist_env_and_memory()


def test_tp_query_shards_match_replicated_prefill_on_both_ranks():
    import pytest

    from vllm.utils.network_utils import get_open_port

    if not torch.cuda.is_available() or torch.accelerator.device_count() < 2:
        pytest.skip("Two CUDA devices required")
    mp.spawn(_prefill_tp_worker, args=(str(get_open_port()),), nprocs=2)
