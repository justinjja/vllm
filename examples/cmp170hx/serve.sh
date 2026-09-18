#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

# Match these physical GPU and NUMA orders to your machine before launching.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,4,5,6,7,2,3}"
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export NCCL_P2P_DISABLE=0
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_RAISE_ON_LOGIT_NANS=1
export VLLM_COMPUTE_NANS_IN_LOGITS=1
export VLLM_PP_LAYER_PARTITION=11,9,10,10
export VLLM_NO_USAGE_STATS=1

exec vllm serve "${MODEL_PATH:-deepseek-ai/DeepSeek-V4.1-Flash}" \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --served-model-name deepseek-v4.1-flash \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 4 \
  --distributed-executor-backend mp \
  --engram-config '{"cpu_offload":true}' \
  --attention-config '{"backend":"TRITON_MLA_SPARSE_DSV41"}' \
  --max-model-len 1048576 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.80 \
  --async-scheduling \
  --numa-bind \
  --numa-bind-nodes 0 0 1 1 1 1 0 0 \
  --tool-call-parser deepseek_v41 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v41 \
  --seed 0 \
  --additional-config '{"cmp_sparse_mla_backend": "triton", "deepseek_v41_pp_sharing": true, "deepseek_v41_pp_runner_sharing": true, "pipeline_max_batch_requests": 8, "pipeline_batch_policy": "adaptive", "pipeline_min_batch_requests": 3, "cmp_wide_prefill_heads": true, "cmp_trim_empty_attention_tiles": true, "deepseek_v41_pp_share_max_bytes": 4294967296, "deepseek_v41_pp_delta_sharing": true, "cmp_short_prefill_prompt_tokens": 65536, "cmp_short_prefill_chunk_tokens": 1024}' \
  --speculative-config '{"method": "dspark", "num_speculative_tokens": 3, "enable_adaptive_verification": false, "draft_sample_method": "probabilistic", "rejection_sample_method": "standard"}' \
  --compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 12, 16, 20, 24, 28, 32]}' \
  "$@"
