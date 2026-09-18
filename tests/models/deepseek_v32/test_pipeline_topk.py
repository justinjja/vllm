# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-index layers must see the same indices across pipeline cuts."""

from types import SimpleNamespace

import torch
from torch import nn

from vllm.models.deepseek_v32.nvidia import model as model_module


class _IndexedLayer(nn.Module):
    def __init__(self, indices, selects):
        super().__init__()
        self.indices = indices
        self.selects = selects

    def forward(self, positions, hidden_states, residual, attn_in):
        rows = self.indices[: len(positions)]
        if self.selects:
            rows.copy_(torch.stack((positions, positions + 1000), dim=-1))
        return hidden_states + rows.float(), torch.zeros_like(hidden_states)


def _stage(start, end):
    model = object.__new__(model_module.DeepseekV32Model)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(hidden_size=2, index_topk=2)
    model.share_topk_across_pp = True
    model.topk_indices_buffer = torch.full((4, 2), -100, dtype=torch.int32)
    model.use_sequence_parallel = False
    model.replicated_embed = False
    model.start_layer, model.end_layer = start, end
    model.aux_hidden_state_layers = ()
    model.norm = nn.Identity()
    model.layers = nn.ModuleList(
        [
            _IndexedLayer(model.topk_indices_buffer, selects=i in (0, 3))
            for i in range(4)
        ]
    )
    return model


def test_shared_index_layers_match_unsplit_forward_across_multiple_stages(monkeypatch):
    """Relay selected int32 indices across two cuts and successive batches."""
    group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(model_module, "get_pp_group", lambda: group)
    monkeypatch.setattr(
        model_module,
        "fused_allreduce_rms_norm",
        lambda hidden, residual, norm: (hidden + residual, None),
    )
    reference = _stage(0, 4)
    stages = [_stage(0, 1), _stage(1, 2), _stage(2, 4)]
    for positions in (torch.tensor([40000, 70000, 120000]), torch.tensor([70001])):
        inputs = torch.zeros(len(positions), 2)
        group.is_first_rank = group.is_last_rank = True
        expected = reference(None, positions, inputs_embeds=inputs)
        payload = None
        for rank, stage in enumerate(stages):
            group.is_first_rank = rank == 0
            group.is_last_rank = rank == len(stages) - 1
            if payload is not None:
                received = stage.make_empty_intermediate_tensors(
                    len(positions), torch.float32, torch.device("cpu")
                )
                for key, tensor in received.tensors.items():
                    tensor.copy_(payload[key])
                assert received["topk_indices"].dtype == torch.int32
                payload = received
            payload = stage(
                None,
                positions,
                intermediate_tensors=payload,
                inputs_embeds=inputs if rank == 0 else None,
            )
        torch.testing.assert_close(payload, expected, rtol=0, atol=0)
