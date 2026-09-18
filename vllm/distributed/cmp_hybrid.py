# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in tensor/expert layouts for the eight-GPU CMP V4.1 experiment."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@dataclass(frozen=True)
class CMPHybridLayout:
    dense_tp_size: int = 2
    expert_ep_size: int = 4
    small_pair_reduce: bool = False

    def dense_groups(self, ranks: list[int]) -> list[list[int]]:
        if len(ranks) != self.dense_tp_size * self.expert_ep_size:
            raise ValueError("CMP hybrid requires exactly eight ranks")
        return [
            ranks[i : i + self.dense_tp_size]
            for i in range(0, len(ranks), self.dense_tp_size)
        ]

    def replica_groups(self, ranks: list[int]) -> list[list[int]]:
        self.dense_groups(ranks)  # Validate the complete set of expert owners.
        return [
            ranks[offset :: self.dense_tp_size] for offset in range(self.dense_tp_size)
        ]


def get_cmp_hybrid_layout(config: "VllmConfig | None" = None) -> CMPHybridLayout | None:
    if config is None:
        from vllm.config.vllm import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
    if config is None or not isinstance(config.additional_config, dict):
        return None
    enabled = config.additional_config.get("cmp_tp2_ep4", False)
    wider = config.additional_config.get("cmp_tp4_ep2", False)
    if type(enabled) is not bool:
        raise ValueError("cmp_tp2_ep4 must be a boolean")
    if type(wider) is not bool:
        raise ValueError("cmp_tp4_ep2 must be a boolean")
    if enabled and wider:
        raise ValueError("Select only one CMP hybrid layout")
    if not enabled and not wider:
        return None
    parallel = config.parallel_config
    required = {
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 1,
        "data_parallel_size": 1,
        "prefill_context_parallel_size": 1,
        "decode_context_parallel_size": 1,
        "enable_expert_parallel": True,
        "enable_eplb": False,
        "enable_elastic_ep": False,
        "enable_dbo": False,
        "ubatch_size": 0,
    }
    for name, value in required.items():
        if getattr(parallel, name) != value:
            raise ValueError(f"CMP hybrid requires {name}={value}")
    model = config.model_config
    if model is not None:
        text_config = model.hf_text_config
        if text_config.model_type not in ("deepseek_v41", "deepseek_v41_text"):
            raise ValueError("CMP hybrid is implemented only for DeepSeek V4.1")
    pair_reduce = config.additional_config.get("cmp_hybrid_pair_reduce", False)
    if type(pair_reduce) is not bool:
        raise ValueError("cmp_hybrid_pair_reduce must be a boolean")
    if wider and pair_reduce:
        raise ValueError("cmp_hybrid_pair_reduce is qualified only for TP2/EP4")
    return CMPHybridLayout(
        dense_tp_size=4 if wider else 2,
        expert_ep_size=2 if wider else 4,
        small_pair_reduce=pair_reduce,
    )


def hybrid_expert_all_reduce(
    states: torch.Tensor, small_pair_reduce: bool
) -> torch.Tensor:
    from vllm.distributed import get_cmp_replica_group, get_ep_group, get_tp_group

    # The staged reduction is faster only for the measured small decode shapes.
    if small_pair_reduce and states.shape[0] <= 6:
        return get_cmp_replica_group().all_reduce(get_tp_group().all_reduce(states))
    return get_ep_group().all_reduce(states)


def combine_replicated_experts(
    routed: torch.Tensor, shared: torch.Tensor | None, replicas: int
) -> torch.Tensor:
    """Form FP32 rank contributions, counting replicated shared experts once."""
    if replicas < 1:
        raise ValueError("Shared expert replica count must be positive")
    result = routed.float()
    if shared is not None:
        result = torch.add(result, shared.float(), alpha=1.0 / replicas)
    return result
