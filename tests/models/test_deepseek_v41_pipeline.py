# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.cmp_hybrid import (
    combine_replicated_experts,
    get_cmp_hybrid_layout,
)
from vllm.models.deepseek_v41.common.pipeline import (
    get_sharing_dependencies,
    validate_local_sharing,
)


def _config():
    return SimpleNamespace(
        num_hidden_layers=40,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
    )


def _hybrid_config():
    return SimpleNamespace(
        additional_config={"cmp_tp2_ep4": True},
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="deepseek_v41_text")
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            prefill_context_parallel_size=1,
            decode_context_parallel_size=1,
            enable_expert_parallel=True,
            enable_eplb=False,
            enable_elastic_ep=False,
            enable_dbo=False,
            ubatch_size=0,
        ),
    )


def test_hybrid_dense_groups_keep_each_plx_pair_together():
    layout = get_cmp_hybrid_layout(_hybrid_config())
    assert layout is not None
    assert layout.dense_groups([0, 1, 4, 5, 6, 7, 2, 3]) == [
        [0, 1],
        [4, 5],
        [6, 7],
        [2, 3],
    ]
    assert layout.replica_groups([0, 1, 4, 5, 6, 7, 2, 3]) == [
        [0, 4, 6, 2],
        [1, 5, 7, 3],
    ]
    with pytest.raises(ValueError, match="exactly eight"):
        layout.dense_groups(list(range(7)))


def test_hybrid_tp4_keeps_node_groups_and_disjoint_expert_owners():
    config = _hybrid_config()
    config.additional_config = {"cmp_tp4_ep2": True}
    layout = get_cmp_hybrid_layout(config)
    assert layout is not None
    assert layout.dense_groups(list(range(8))) == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert layout.replica_groups(list(range(8))) == [[0, 4], [1, 5], [2, 6], [3, 7]]
    assert layout.expert_ep_size == 2
    config.additional_config["cmp_tp2_ep4"] = True
    with pytest.raises(ValueError, match="only one"):
        get_cmp_hybrid_layout(config)
    config.additional_config = {"cmp_tp4_ep2": "true"}
    with pytest.raises(ValueError, match="boolean"):
        get_cmp_hybrid_layout(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tensor_parallel_size", 4),
        ("pipeline_parallel_size", 2),
        ("data_parallel_size", 4),
        ("enable_expert_parallel", False),
        ("prefill_context_parallel_size", 2),
        ("decode_context_parallel_size", 2),
        ("enable_eplb", True),
        ("enable_elastic_ep", True),
        ("enable_dbo", True),
        ("ubatch_size", 2),
    ],
)
def test_hybrid_rejects_incompatible_parallel_modes(field, value):
    config = _hybrid_config()
    setattr(config.parallel_config, field, value)
    with pytest.raises(ValueError, match=field):
        get_cmp_hybrid_layout(config)


def test_hybrid_is_explicit_and_model_specific():
    config = _hybrid_config()
    config.additional_config = {}
    assert get_cmp_hybrid_layout(config) is None
    config.additional_config = {"cmp_tp2_ep4": "true"}
    with pytest.raises(ValueError, match="boolean"):
        get_cmp_hybrid_layout(config)
    config.additional_config = {"cmp_tp2_ep4": True}
    config.model_config.hf_text_config.model_type = "llama"
    with pytest.raises(ValueError, match="only for DeepSeek"):
        get_cmp_hybrid_layout(config)
    config.model_config.hf_text_config.model_type = "deepseek_v41_text"
    config.additional_config["cmp_hybrid_pair_reduce"] = "true"
    with pytest.raises(ValueError, match="cmp_hybrid_pair_reduce.*boolean"):
        get_cmp_hybrid_layout(config)


@pytest.mark.parametrize("has_shared", [False, True])
@pytest.mark.parametrize("tp_size", [2, 4])
def test_hybrid_sharded_experts_match_dense_reference_without_duplicating_shared(
    has_shared,
    tp_size,
):
    # Shard intermediate features within owners and experts between owners.
    # Distinct routes exercise empty rank contributions and shared replication.
    rng = torch.Generator().manual_seed(170)
    x = torch.randn(3, 8, generator=rng)
    up = torch.randn(9, 8, 12, generator=rng) * 0.1
    gate = torch.randn(9, 8, 12, generator=rng) * 0.1
    down = torch.randn(9, 12, 8, generator=rng) * 0.1
    routes = torch.tensor([[0, 1, 7], [2, 5, 6], [1, 3, 5]])
    weights = torch.randn(3, 3, generator=rng).softmax(-1)

    def project(expert, part=None):
        if part is None:
            part = slice(None)
        return (
            torch.nn.functional.silu(x @ gate[expert, :, part])
            * (x @ up[expert, :, part])
        ) @ down[expert, part, :]

    expected = torch.zeros_like(x)
    for expert in range(8):
        factor = ((routes == expert) * weights).sum(-1, keepdim=True)
        expected += factor * project(expert)
    if has_shared:
        expected += project(8)
    partials = []
    replicas = 8 // tp_size
    shard = 12 // tp_size
    experts_per_owner = 8 // replicas
    for rank in range(8):
        owner, tp_rank = divmod(rank, tp_size)
        part = slice(tp_rank * shard, (tp_rank + 1) * shard)
        routed = torch.zeros_like(x)
        for expert in range(owner * experts_per_owner, (owner + 1) * experts_per_owner):
            factor = ((routes == expert) * weights).sum(-1, keepdim=True)
            routed += factor * project(expert, part)
        shared = project(8, part) if has_shared else None
        partials.append(combine_replicated_experts(routed, shared, replicas))
    torch.testing.assert_close(
        torch.stack(partials).sum(0), expected, rtol=1e-5, atol=1e-7
    )


@pytest.mark.parametrize(
    "ranges", [[(0, 40)], [(0, 20), (20, 40)], [(0, 8), (8, 14), (14, 20), (20, 40)]]
)
def test_group_aligned_pipeline_cuts_keep_all_sources_local(ranges):
    validate_local_sharing(get_sharing_dependencies(_config(), ranges))


def test_equal_pp4_reports_cross_stage_kv_index_and_candidates():
    dependencies = get_sharing_dependencies(
        _config(), [(0, 10), (10, 20), (20, 30), (30, 40)]
    )
    cross = {
        (d.kind, d.source_layer, d.consumer_layer)
        for d in dependencies
        if d.source_stage != d.consumer_stage
    }
    assert {("kv", 8, 10), ("index", 28, 30), ("candidate", 20, 32)} <= cross
    with pytest.raises(NotImplementedError, match="source layer 8.*stage 0.*stage 1"):
        validate_local_sharing(dependencies)


@pytest.mark.parametrize(
    "partition,socket_cut",
    [([20, 20], 1), ([10, 10, 11, 9], 2), ([5, 5, 5, 5, 6, 6, 5, 3], 4)],
)
def test_eight_gpu_socket_boundary_transfers_no_shared_attention_state(
    partition, socket_cut
):
    ranges = []
    start = 0
    for count in partition:
        ranges.append((start, start + count))
        start += count
    dependencies = get_sharing_dependencies(_config(), ranges)
    assert dependencies  # Exercise actual KV, index and candidate dependencies.
    assert {d.kind for d in dependencies} == {"kv", "index", "index_k", "candidate"}
    assert not any(
        d.source_stage < socket_cut <= d.consumer_stage for d in dependencies
    )


@pytest.mark.parametrize("values", [[8, 2], [2, 2], [0, 2], [-1, 2], [2, 40]])
def test_invalid_source_lists_fail_before_layer_construction(values):
    config = _config()
    config.kv_source_layer_ids = values
    with pytest.raises(ValueError, match="kv_source_layer_ids"):
        get_sharing_dependencies(config, [(0, 40)])


def test_compressed_consumer_requires_an_earlier_source():
    config = _config()
    config.kv_source_layer_ids = [8, 14, 20]
    with pytest.raises(ValueError, match="layer 2 has no preceding kv source"):
        get_sharing_dependencies(config, [(0, 40)])


def test_candidate_source_must_publish_indices():
    config = _config()
    config.candidate_source_layer_id = 21
    with pytest.raises(ValueError, match="candidate source must be an index source"):
        get_sharing_dependencies(config, [(0, 40)])


@pytest.mark.parametrize("pipeline_boundary", [False, True])
def test_engram_is_injected_before_attention_pre_mix_at_pipeline_boundary(
    monkeypatch, pipeline_boundary
):
    """A stage starting at Engram layer 14 must consume its prefetched rows."""
    from vllm.models.deepseek_v41.nvidia import model

    seen = []

    def pre(residual, *args, **kwargs):
        seen.append(residual.clone())
        return None, None, residual.mean(1), None

    monkeypatch.setattr(model, "mhc_pre_delayed_tilelang", pre)
    monkeypatch.setattr(model, "mhc_post_tilelang", lambda x, residual, *a: residual)
    monkeypatch.setattr(
        model,
        "mhc_shifted_post_pre",
        lambda x, residual, *a, **kw: (residual, None, None, x, None, None),
    )

    class Inject:
        layer_hash_index = 0

        def __call__(self, residual, hashes, mask):
            assert hashes.tolist() == [[1, 2], [3, 4]]
            return residual + 7

    norm = SimpleNamespace(weight=torch.ones(3), variance_epsilon=1e-6)
    layer = SimpleNamespace(
        use_sequence_parallel=False,
        engram=Inject(),
        attn=lambda positions, x, cache: x,
        ffn=lambda x, input_ids: x,
        attn_norm=norm,
        ffn_norm=norm,
        **dict.fromkeys(
            (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "hc_ffn_fn",
                "hc_ffn_scale",
                "hc_ffn_base",
                "rms_norm_eps",
                "hc_eps",
                "hc_post_alpha",
                "hc_sinkhorn_iters",
            ),
            1,
        ),
    )
    residual = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    model.DeepseekV4DecoderLayer.forward(
        layer,
        residual if pipeline_boundary else residual.mean(1),
        positions=torch.arange(2),
        input_ids=torch.ones(2, dtype=torch.int64),
        residual=None if pipeline_boundary else residual,
        engram_hashes=torch.tensor([[[1, 2]], [[3, 4]]]),
        engram_mask=torch.ones(2, dtype=torch.bool),
    )
    torch.testing.assert_close(seen[0], residual + 7)
