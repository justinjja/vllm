# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NaN reporting after speculative sampling's asynchronous output copy."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu.metrics.logits import get_num_nans
from vllm.v1.worker.gpu.sample.output import SamplerOutput


@pytest.mark.parametrize(
    ("counts", "boundaries", "expected"),
    [
        ([0, 0, 17, 0], [0, 4], {"0": 17}),
        ([0, 0, 17, 0, 0, 0, 3], [0, 4, 7], {"0": 17, "1": 3}),
        ([0, 3], None, {"0": 0, "1": 3}),
        ([0, 0], [0, 1, 2], {"0": 0, "1": 0}),
    ],
    ids=["late-draft", "variable-drafts-and-bonus", "already-reduced", "no-drafts"],
)
@pytest.mark.parametrize("raise_on_nan", [False, True])
def test_nan_reporting_covers_every_logits_row(
    monkeypatch, counts, boundaries, expected, raise_on_nan
):
    # The CUDA copy has completed; exercise the real CPU completion path.
    output = object.__new__(async_utils.AsyncOutput)
    output.copy_event = Mock()
    output.sampled_token_ids = np.zeros((len(expected), 1), dtype=np.int64)
    output.num_sampled_tokens_np = np.ones(len(expected), dtype=np.int32)
    output.model_runner_output = ModelRunnerOutput(
        req_ids=list(expected),
        req_id_to_index={key: int(key) for key in expected},
        sampled_token_ids=[],
        prompt_logprobs_dict={},
    )
    output.num_nans = np.array(counts, dtype=np.int32)
    output.cu_num_logits = (
        np.array(boundaries, dtype=np.int32) if boundaries is not None else None
    )
    output.sampling_mask_tensors = None
    output.logprobs_tensors = None
    output.prompt_logprobs_dict = {}
    output.routed_experts_cpu = None
    output._has_fault = None
    monkeypatch.setattr(async_utils.envs, "VLLM_RAISE_ON_LOGIT_NANS", raise_on_nan)

    if raise_on_nan and any(expected.values()):
        with pytest.raises(RuntimeError, match="NaN"):
            output.get_output()
    else:
        assert output.get_output().num_nans_in_logits == expected
    output.copy_event.synchronize.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_nan_logits_boundaries_survive_buffer_reuse(monkeypatch):
    monkeypatch.setattr(async_utils.envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    logits = torch.zeros((7, 16), device="cuda")
    logits[2, 0] = float("nan")
    logits[6, :3] = float("nan")
    boundaries = torch.tensor([0, 4, 7], dtype=torch.int32, device="cuda")
    sampled = torch.zeros((2, 1), dtype=torch.int64, device="cuda")
    counts = torch.ones(2, dtype=torch.int32, device="cuda")
    output = async_utils.AsyncOutput(
        ModelRunnerOutput(
            req_ids=["a", "b"],
            req_id_to_index={"a": 0, "b": 1},
            sampled_token_ids=[],
            prompt_logprobs_dict={},
        ),
        SamplerOutput(sampled, None, get_num_nans(logits), counts),
        counts,
        torch.cuda.current_stream(),
        torch.cuda.Stream(),
        cu_num_logits=boundaries,
    )
    # Adaptive verification can overwrite its boundary buffer on the next step.
    boundaries.fill_(0)
    assert output.get_output().num_nans_in_logits == {"a": 1, "b": 3}
