# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v41.common.pipeline import SharingDependency
from vllm.models.deepseek_v41.common.pipeline_transfer import (
    get_sharing_routes,
    prefill_write_blocks,
    restore_cache_blocks,
    snapshot_cache_blocks,
    snapshot_updated_cache_blocks,
)


@pytest.mark.parametrize(
    "async_scheduling, adaptive, dspark, supported",
    [
        (True, False, True, True),
        (False, False, True, False),
        (True, True, True, False),
        (True, False, False, False),
    ],
)
def test_pipeline_speculation_rejects_unverified_execution_modes(
    async_scheduling, adaptive, dspark, supported
):
    from vllm.models.deepseek_v41.common.pipeline_sharing import PipelineSharing

    config = SimpleNamespace(
        additional_config={"deepseek_v41_pp_runner_sharing": True},
        model_config=SimpleNamespace(enforce_eager=False),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
        compilation_config=SimpleNamespace(static_forward_context={}),
        speculative_config=SimpleNamespace(
            use_dspark=lambda: dspark, enable_adaptive_verification=adaptive
        ),
        parallel_config=SimpleNamespace(
            use_ubatching=False,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            distributed_executor_backend="mp",
        ),
        kv_transfer_config=None,
        use_v2_model_runner=True,
    )
    if supported:
        sharing = PipelineSharing(config, "model", 0, (), object, 1024)
        assert sharing.runner_mode
    else:
        with pytest.raises(ValueError, match="async scheduling"):
            PipelineSharing(config, "model", 0, (), object, 1024)


def test_sharing_routes_relay_across_idle_stages_without_duplicate_sends():
    routes = get_sharing_routes(
        (
            SharingDependency("kv", 2, 8, 0, 2),
            SharingDependency("kv", 2, 9, 0, 2),
            SharingDependency("kv", 2, 10, 0, 3),
            SharingDependency("index", 2, 3, 0, 0),
        )
    )
    assert [(r.kind, r.source_layer, r.sender, r.receiver) for r in routes] == [
        ("kv", 2, 0, 1),
        ("kv", 2, 1, 2),
        ("kv", 2, 2, 3),
    ]
    assert len({r.payload_keys for r in routes}) == 1


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_snapshot_refreshes_prefix_blocks_and_keeps_packed_layout(dtype):
    # Interleaved cache allocations have gaps between physical blocks.
    source = torch.arange(6 * 2 * 3 * 4).reshape(6, 2, 3, 4).to(dtype)[:, 0]
    replica = torch.full((6, 2, 3, 4), 17, dtype=dtype)[:, 0]
    ids, blocks = snapshot_cache_blocks(
        source, [torch.tensor([[4, 1, -1], [1, 4, -1]])], max_bytes=1024
    )
    assert ids.tolist() == [1, 4]
    old = source.clone()
    source.zero_()
    restore_cache_blocks(replica, ids, blocks)
    torch.testing.assert_close(replica[ids], old[ids], rtol=0, atol=0)
    assert torch.all(replica[[0, 2, 3, 5]] == 17)


def test_empty_snapshot_preserves_replica():
    cache = torch.ones(2, 3, 4)
    ids, blocks = snapshot_cache_blocks(cache, [torch.tensor([[-1]])], max_bytes=0)
    restore_cache_blocks(cache, ids, blocks)
    assert torch.all(cache == 1)


@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
def test_incremental_cache_matches_full_snapshots_across_reuse_and_prefix_hits(dtype):
    """Old prefixes persist; boundary rewrites and reused blocks must refresh."""
    source = torch.arange(12 * 2 * 4 * 8).reshape(12, 2, 4, 8).to(dtype)[:, 0]
    replica = torch.zeros_like(source)
    full_replica = torch.zeros_like(source)
    mirrored = torch.zeros(12, dtype=torch.bool)
    rounds = [
        ([[2, 5]], [8, 9, 10, 11, 20], [2, 5]),
        ([[2, 5, 7]], [21, 22, 23, 28], [5, 7]),
        ([[2, 5, 7]], [28, 29, -1], [7]),  # rejected draft overwritten
        ([[9], [2, 5, 7]], [36, 30], [7, 9]),  # reordered cohort
        ([[2, 5], [9]], [], []),  # cached prefix, no writes
        ([[2, 5], [9]], [8, 9, 36], [2, 9]),  # physical blocks reused
        ([[2, 5]], [], []),
    ]
    for round_id, (rows, slots, expected_delta) in enumerate(rounds):
        valid_slots = [s for s in slots if s >= 0]
        for slot in valid_slots:
            source[slot // 4, slot % 4] = 31 + round_id
        tables = [torch.tensor(row, dtype=torch.int32) for row in rows]
        full_ids, full_blocks = snapshot_cache_blocks(source, tables, 100000)
        restore_cache_blocks(full_replica, full_ids, full_blocks)
        ids, blocks = snapshot_updated_cache_blocks(
            source,
            tables,
            torch.tensor(slots, dtype=torch.int64),
            4,
            mirrored,
            100000,
        )
        assert ids.tolist() == expected_delta
        restore_cache_blocks(replica, ids, blocks)
        torch.testing.assert_close(
            replica[full_ids], full_replica[full_ids], rtol=0, atol=0
        )


def test_incremental_snapshot_failure_does_not_mark_unsent_blocks():
    source = torch.ones(3, 4, 8)
    mirrored = torch.zeros(3, dtype=torch.bool)
    with pytest.raises(ValueError, match="byte budget"):
        snapshot_updated_cache_blocks(
            source, [torch.tensor([1])], torch.tensor([4]), 4, mirrored, 1
        )
    assert not mirrored.any()


def test_prefill_write_plan_covers_boundaries_and_rejects_optimistic_decode_lengths():
    table = torch.tensor([[5, 6, 7], [10, 11, 12], [-1, -1, -1]])
    ends = torch.tensor([129, 258, 0])
    writes = prefill_write_blocks(table, torch.tensor([0, 129, 259, 259]), ends, 128, 4)
    assert writes.tolist() == [5, 6, 11, 12]
    assert (
        prefill_write_blocks(table, torch.tensor([0, 4, 134, 134]), ends, 128, 4)
        is None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_host_prefill_delta_does_not_synchronize_the_model_stream():
    source = torch.arange(64, device="cuda").reshape(4, 4, 4).to(torch.uint8)
    replica = torch.zeros_like(source)
    mirrored = torch.zeros(4, dtype=torch.bool)
    table = torch.tensor([[1, 2]], dtype=torch.int32)
    slots = torch.tensor([4, 8], device="cuda")
    writes = torch.tensor([1, 2], dtype=torch.int64)
    torch.accelerator.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        ids, blocks = snapshot_updated_cache_blocks(
            source, [table], slots, 4, mirrored, 1000, write_blocks_cpu=writes
        )
        restore_cache_blocks(replica, ids, blocks)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    torch.testing.assert_close(replica[[1, 2]], source[[1, 2]], rtol=0, atol=0)


def test_snapshot_enforces_memory_budget():
    with pytest.raises(ValueError, match="byte budget"):
        snapshot_cache_blocks(torch.ones(2, 3, 4), [torch.tensor([[0, 1]])], 1)


def test_snapshot_rejects_invalid_physical_blocks():
    with pytest.raises(ValueError, match="unallocated cache block"):
        snapshot_cache_blocks(torch.ones(2, 3, 4), [torch.tensor([[2]])], 1024)


@pytest.mark.parametrize("blocks", [torch.zeros(1, 4, 3), torch.zeros(1, 3, 4).int()])
def test_restore_rejects_different_cache_layouts(blocks):
    with pytest.raises(ValueError, match="different block layout"):
        restore_cache_blocks(torch.ones(2, 3, 4), torch.tensor([0]), blocks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bfloat16])
@pytest.mark.parametrize("empty", [False, True])
def test_host_block_plan_restores_cuda_prefix_without_device_readback(dtype, empty):
    """Keep packed prefix bytes exact while host IDs avoid CUDA scalar reads."""
    source = (
        torch.arange(6 * 2 * 3 * 4, device="cuda").reshape(6, 2, 3, 4).to(dtype)[:, 0]
    )
    replica = torch.full((6, 2, 3, 4), 17, device="cuda", dtype=dtype)[:, 0]
    table = (
        torch.tensor([[-1, -1]]) if empty else torch.tensor([[4, 1, -1], [1, 4, -1]])
    )
    expected = source.clone()
    # Initialize lazy CUDA facilities before enabling the synchronization guard.
    warm_ids, warm_blocks = snapshot_cache_blocks(source, [table], max_bytes=1024)
    restore_cache_blocks(replica, warm_ids, warm_blocks)
    replica.fill_(17)
    torch.accelerator.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        ids, blocks = snapshot_cache_blocks(source, [table], max_bytes=1024)
        restore_cache_blocks(replica, ids, blocks)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert ids.device.type == "cpu"
    selected = [] if empty else [1, 4]
    assert ids.tolist() == selected
    torch.testing.assert_close(replica[selected], expected[selected], rtol=0, atol=0)
    untouched = [i for i in range(6) if i not in selected]
    assert torch.all(replica[untouched] == 17)


def test_unpadding_preserves_the_matching_host_block_table():
    from vllm.v1.attention.backend import CommonAttentionMetadata

    table = torch.tensor([[3, 7], [4, 9]], dtype=torch.int32)
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2]),
        query_start_loc_cpu=torch.tensor([0, 1, 2]),
        seq_lens=torch.tensor([1, 1]),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=1,
        block_table_tensor=table,
        slot_mapping=torch.tensor([0, 1]),
        block_table_cpu=table.clone(),
    )
    actual = metadata.unpadded(1, 1)
    torch.testing.assert_close(actual.block_table_cpu, table[:1])
    assert actual.block_table_cpu.shape == actual.block_table_tensor.shape


def test_v2_metadata_passes_scheduler_host_block_plan_to_attention_builder():
    """Prevent V2 falling back to synchronizing GPU unique/scalar reads in PP."""
    from vllm.v1.worker.gpu.attn_utils import build_attn_metadata

    table = torch.tensor([[3, 7], [4, 9]], dtype=torch.int32)
    builder = SimpleNamespace(build=lambda **kw: kw["common_attn_metadata"])
    group = SimpleNamespace(
        layer_names=["shared"], get_metadata_builder=lambda _: builder
    )
    metadata = build_attn_metadata(
        attn_groups=[[group]],
        num_reqs=1,
        num_tokens=1,
        query_start_loc_gpu=torch.tensor([0, 1]),
        query_start_loc_cpu=torch.tensor([0, 1]),
        max_query_len=1,
        seq_lens=torch.tensor([1]),
        max_seq_len=1,
        block_tables=[table],
        block_tables_cpu=[table.clone()],
        slot_mappings=torch.tensor([[0]]),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[group]),
    )
    assert metadata["shared"].block_table_cpu.tolist() == [[3, 7]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("incremental", [False, True])
def test_runner_relay_refreshes_dynamic_cache_inputs_between_graph_replays(incremental):
    """The graph reads static replicas, not the previous transfer's allocation."""
    from types import SimpleNamespace

    from vllm.models.deepseek_v41.common.pipeline_sharing import PipelineSharing
    from vllm.models.deepseek_v41.common.pipeline_transfer import SharingRoute
    from vllm.models.deepseek_v41.nvidia.vl_model import DeepseekV41ForCausalLM
    from vllm.sequence import IntermediateTensors

    route = SharingRoute("kv", 2, 0, 1)

    def wrapper(cache, inbound):
        sharing = PipelineSharing.__new__(PipelineSharing)
        torch.nn.Module.__init__(sharing)
        sharing.runner_mode = True
        sharing.delta_transfers = incremental
        sharing._mirrored_blocks = {}
        sharing._received_blocks = {}
        sharing.max_bytes = 4096
        sharing._prefix = "model"
        sharing._context = {"model.layers.2.attn": SimpleNamespace(kv_cache=cache)}
        sharing.inbound = (route,) if inbound else ()
        sharing.outbound = () if inbound else (route,)
        sharing.payload_keys = frozenset(route.payload_keys)
        model = SimpleNamespace(
            pipeline_sharing=sharing,
            topk_indices_buffer=torch.zeros(1, 4, device="cuda", dtype=torch.int32),
            candidate_block_buffer=None,
        )
        return SimpleNamespace(language_model=SimpleNamespace(model=model))

    source = torch.zeros(4, 8, device="cuda")
    replica = torch.zeros_like(source)
    sender, receiver = wrapper(source, False), wrapper(replica, True)
    static = IntermediateTensors({"hidden_states": torch.ones(1, 8, device="cuda")})
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = replica.sum(0)
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph):
        result = replica.sum(0)

    for value, blocks in ((3, [1]), (7, [1, 3]), (11, [0, 1, 3])):
        source[blocks] = value
        metadata = {
            "model.layers.2.attn": SimpleNamespace(
                block_table_cpu=torch.tensor([blocks]),
                slot_mapping=torch.tensor([b * 8 for b in blocks], device="cuda"),
                cache_block_size=8,
            )
        }
        payload = DeepseekV41ForCausalLM.finish_pipeline_outputs(
            sender, static, len(blocks), metadata
        )
        clean = DeepseekV41ForCausalLM.prepare_pipeline_inputs(receiver, payload, 1)
        assert set(clean.tensors) == {"hidden_states"}
        assert set(static.tensors) == {"hidden_states"}
        graph.replay()
        torch.testing.assert_close(result, source.sum(0), rtol=0, atol=0)
