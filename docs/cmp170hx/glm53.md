# GLM-5.3 NVFP4 on CMP 170HX

`local-inference-lab/GLM-5.3-NVFP4` runs on eight 64-GiB CMP 170HX GPUs with
the Triton sparse MLA backend and Marlin NVFP4 experts. BF16 and software FP8
KV paths pass kernel and engine integration checks. Real-model performance
qualification has begun with BF16 KV; layout and high-context tuning continue.

## Implementation and provenance

The sparse MLA kernel, backend skeleton, and split-KV tests derive from
[@haosdent's vLLM PR #38476](https://github.com/vllm-project/vllm/pull/38476),
pinned at `3740c02bb1223d37823593664ae3eafa397b9937` under Apache-2.0.
[allover326's Ampere DSA/MTP integration](https://github.com/allover326/vllm-dsa-mtp-sm80)
provided additional integration references.

This adaptation uses the fork's Ampere indexer, software E4M3 conversion in
the optimized GLM normalization/RoPE kernels, padded cache addressing, and
the current vLLM metadata interface. Pipeline stages transfer shared top-k
indices as int32 tensors, preserving selections when a stage begins with a
layer that reuses an earlier indexer.

The optional additional configuration `sparse_indexer_max_prefill_tokens`
bounds the indexer's temporary gathered-key workspace. It must be at least
the context limit. Prefill chunking covers every request and query within
that workspace and the existing logits-memory budget. At one-million-token
context, a one-million-token workspace uses about 132 MiB instead of the
default approximately 5.2 GiB. This does not change the model's attention
selection count or context window.

NVFP4 Marlin scale conversion reduces positive scales in bounded chunks.
This preserves the rescaling factor while avoiding the multi-GiB integer
index tensors created by boolean indexing across all experts. The original
conversion attempted a 9-GiB temporary allocation and prevented PP8 loading
on this checkpoint; the bounded version successfully loads the same weights.

## SM80 attention improvements

Sparse MLA now chooses KV splits from the number of active query/head blocks
on SM80 and skips only trailing invalid indices. It retains every selected
key, including valid entries after holes in the index list. Prefill uses
absorbed sparse MLA. The experimental dense FlashAttention path for short
prefills changed generated answers and is disabled on this backend.
The SM80 indexer also reuses a bounded score buffer reserved during memory
profiling, avoiding changing allocation sizes as prefill context grows.

The following measurements use TP2/PP4, layer partition `22,20,20,16`, two
MTP draft tokens, BF16 KV, a 2,048-token batch budget, and adaptive pipeline
batching. The configured context limit is 262,144. Exact samples are in the
[attention measurement record](glm53-attention-20260918.json).

| Matched 512-input / 128-output pilot | Before | After attention changes |
| --- | ---: | ---: |
| Single decode, tokens/s | 49.22 | 47.23 |
| End-to-end generation, concurrency 1 | 38.28 | 38.57 |
| End-to-end generation, concurrency 8 | 134.11 | 145.16 |
| End-to-end generation, concurrency 16 | 146.03 | 193.97 |
| 8K prefill TTFT, seconds | 4.71 | 4.49 |
| 30K prefill TTFT, seconds | 13.31 | 12.97 |

These are one-repeat workload comparisons. Prompt prefixes and generated
tokens differ between runs. The measured concurrency-16 rate increased;
the single-request rate did not improve.

Longer runs with 512 input and 512 output tokens, score-buffer reuse,
sparse-only prefill, and GPU memory utilization 0.978 measured the following
steady decode rates over two repetitions. Accepted output token IDs are
counted during the interval where every request is decoding, after trimming
the first and last 32 tokens of each request. Draft proposals are excluded.
End-to-end rates include prefill and queue time. The different output length
means these rates are not a direct before/after comparison with the table above.

| Concurrency | Steady accepted tokens/s | End-to-end tokens/s |
| --- | ---: | ---: |
| 1 | 48.01–54.47 | 46.46–51.34 |
| 8 | 252.54–265.90 | 202.13–218.05 |
| 16 | 349.86–360.41 | 281.10–282.34 |

These prompts match the earlier dense-prefill run byte for byte. That run
measured 51.41–61.17 single, 263.65–272.30 at concurrency eight, and
341.97–356.32 at concurrency sixteen. Sparse prefill restored the failed
objective answer: 20/20 on two complete runs and 3/3 isolated lowercase
checks. The retained sparse attention improvements preserve the aggregate
throughput gain at concurrency sixteen.

A TP8/PP1 trial with two draft tokens, local draft argmax reduction, and
batch-sharded sampling passed 20/20 objective checks, reasoning/tools, and
8/8 concurrent retrieval checks. Its 32,768-token context and 0.975 memory
utilization left 40,128 BF16 KV tokens. Two matched-corpus runs measured
53.17–59.88 single, 227.06–237.04 aggregate at concurrency eight, and
297.14–307.99 at concurrency sixteen during steady decode. Its 8K prefill
TTFT was 9.00–9.01 seconds, versus 5.10–5.20 seconds for TP2/PP4. TP2/PP4
remains the stronger combined throughput, prefill, and context configuration.
The model now supports the existing opt-in `--enable-batch-sharded-sampling`
path; the default serving recipe does not require it.

The working steady-decode targets are 100 single, 300 aggregate at concurrency
eight, and 450 aggregate at concurrency sixteen. These are planning targets,
not measured results. Single generation remains substantially below target.
Reasoning, tools, three-record retrieval around 28K input, and eight concurrent
retrieval requests passed.

The retained sparse-prefill build recovered all three records from 260,222
input tokens in both fresh and cached retrieval checks. Fresh TTFT was
139.35 seconds; cached TTFT was 1.56 seconds. Decode measured approximately
68 and 60 tokens/s respectively, with different generated output lengths.
The [measurement record](glm53-attention-20260918.json) retains the earlier
dense-prefill results separately from this sparse-prefill qualification.
The cache has 269,056 BF16 tokens, enough for one near-limit request. The
0.985 configuration exhausted prefill activation memory and is not the
qualified high-context recipe. This does not qualify the model's advertised
1,048,576-token limit.

### Serving configuration

Preserve the checkpoint's complete quantization configuration when excluding
its BF16 MTP layer:

```bash
uv run --python .venv/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "local-inference-lab/GLM-5.3-NVFP4", "config.json",
    revision="b472e4ee53f6a9862da5486c56c6ca21be3dab70",
)
quant = json.loads(Path(path).read_text())["quantization_config"]
if "model.layers.78.*" not in quant["ignore"]:
    quant["ignore"].append("model.layers.78.*")
Path("glm-mtp-overrides.json").write_text(json.dumps({"quantization_config": quant}))
PY

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_PP_LAYER_PARTITION=22,20,20,16
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024
uv run --python .venv/bin/python -m vllm.entrypoints.cli.main serve \
  local-inference-lab/GLM-5.3-NVFP4 \
  --revision b472e4ee53f6a9862da5486c56c6ca21be3dab70 \
  --served-model-name glm-5.3-nvfp4 \
  --tensor-parallel-size 2 --pipeline-parallel-size 4 \
  --dtype bfloat16 --kv-cache-dtype auto \
  --attention-config '{"backend":"TRITON_MLA_SPARSE"}' \
  --additional-config '{"sparse_indexer_max_prefill_tokens":262144,"pipeline_max_batch_requests":8,"pipeline_batch_policy":"adaptive","pipeline_min_batch_requests":1}' \
  --hf-overrides "$(cat glm-mtp-overrides.json)" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":2,"attention_backend":"TRITON_MLA_SPARSE"}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[3,6,12,24,48]}' \
  --max-model-len 262144 --max-num-batched-tokens 2048 --max-num-seqs 16 \
  --gpu-memory-utilization 0.978 --numa-bind --async-scheduling \
  --safetensors-load-strategy prefetch \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 127.0.0.1 --port 8000
```

To reproduce the measurements against a running server, supply a text corpus
longer than the largest tested prompt:

```bash
uv run --python .venv/bin/python benchmarks/cmp170hx/benchmark_serving.py \
  --label glm-sm80 --corpus corpus.txt --output measured.json \
  --repeats 3 --concurrency 8 16 --contexts 8192 30000 \
  --output-tokens 512 --steady-trim-tokens 32
```

Use `--output-tokens 128` for the shorter end-to-end workload. Quality checks
are separate from this performance benchmark, which forces the output length.

## Initial real-model baseline

Measured September 18, 2026, using checkpoint revision
`b472e4ee53f6a9862da5486c56c6ca21be3dab70`, TP8/PP1, BF16 KV, 32,768 context,
1,024 batched tokens, 16 sequences, GPU memory utilization 0.95, and full
decode CUDA graphs. All GPUs report PCIe Gen2 x16, with four cards per NUMA
node. The software environment matches the [build guide](README.md).

These are one-repeat baseline samples, including first-use effects.
Exact timings and check outcomes are in the
[baseline record](glm53-baseline-20260918.json).
Generation uses unique corpus prefixes, 512 input tokens, and 128 output
tokens at temperature zero with EOS ignored. Aggregate rates include
prefill and queue time. Decode rate uses the interval from first to last
streamed text. Separate prefill requests generate one token.

| Measurement | Initial result |
| --- | ---: |
| Single-request decode | 36.05 tokens/s |
| Single-request end-to-end generation | 30.91 tokens/s |
| Aggregate generation, concurrency 4 | 60.89 tokens/s |
| Aggregate generation, concurrency 8 | 92.45 tokens/s |
| Aggregate generation, concurrency 16 | 107.94 tokens/s |
| 8,192-token prefill TTFT | 9.36 s |
| 30,000-token prefill TTFT | 37.12 s |
| Decode after 30,000-token prefill | approximately 35 tokens/s |

Reasoning separation, streamed automatic tool calls, and tool-result
continuation passed. A separate long-document check retrieved all three
synthetic records placed at 10%, 50%, and 90% of the document. This baseline
does not establish performance or correctness at the model's advertised
1,048,576-token limit. TP8 has 48,896 aggregate KV tokens in this configuration.

### Pipeline layout pilots

The following one-repeat runs retain BF16 KV, 32K context, a 1,024-token
batch budget, and 16 sequences. TP4/PP2 partitions layers as `40,38`;
TP2/PP4 uses `21,19,19,19` and adaptive pipeline batching with a maximum
of eight requests and minimum of one request per batch. These compare
serving configurations, including their scheduling policy.

| Measurement | TP8/PP1 | TP4/PP2 | TP2/PP4 adaptive |
| --- | ---: | ---: | ---: |
| Single decode, tokens/s | 36.05 | 29.14 | 35.24 |
| End-to-end generation, concurrency 1 | 30.91 | 25.46 | 29.32 |
| Aggregate generation, concurrency 16 | 107.94 | 145.28 | 157.13 |
| 8K prefill TTFT, seconds | 9.36 | 5.55 | 3.89 |
| 30K prefill TTFT, seconds | 37.12 | 20.26 | 12.50 |
| KV token capacity | 48,896 | 138,048 | 293,120 |

Both pipeline layouts passed reasoning, tools, and the same three-record
retrieval check. TP2/PP4 also passed eight concurrent retrieval requests
with different prompt lengths and answers. Exact samples and configuration
differences are retained in the [layout record](glm53-layouts-20260918.json).

PP8 with a 262,144-token limit completed the same performance matrix. Its
single decode rate was approximately 24 tokens/s, aggregate generation at
concurrency 16 was 125.00 tokens/s, and 30K prefill TTFT was 10.75 seconds.
It has 411,776 BF16 KV tokens. Three MTP draft tokens, with a
`12,10,10,10,10,10,10,6` partition and BF16 draft weights, improved single
decode to approximately 36 tokens/s and concurrency 16 to 134.26 tokens/s.
MTP3 KV capacity was 390,592 tokens. These are one-repeat pilots.

A separate PP8 retrieval check used 129,153 input tokens and recovered all
three records. Fresh first-output latency was 52.93 seconds; reusing the
same prompt reduced it to 1.49 seconds. Decode was approximately 24 tokens/s
in both cases. These measurements include API processing and tokenization.

### Observed output limitations

An initial PP8 concurrent synthetic access-code fixture passed seven of
eight cases. One response identified the correct value in reasoning but
refused to repeat it. Rewording the records as explicitly fictional catalog
numbers passed eight of eight, including with MTP.

PP8 MTP3 passed 17 of 20 broader objective checks with low reasoning effort.
One string reversal changed letter case; two answers contained correct JSON
but omitted the closing reasoning delimiter before stopping, leaving final
content empty. Raw token inspection confirmed the missing delimiter in the
model output. A sequential retry corrected the reversal but retained the
delimiter issue. These checks do not establish universal output equivalence.

After the attention changes, the TP2/PP4 MTP2 maximum-effort objective suite
scored 19/20 on two runs, compared with 20/20 in the earlier baseline. Both
runs incorrectly lowercased `CoMPuTe` as `computer`; isolated repeats answered
correctly once and incorrectly twice. A controlled comparison using absorbed
sparse MLA for prefill restored 20/20 on two runs and 3/3 isolated lowercase
answers. Dense short-prefill is now disabled for this backend. This identifies
the path responsible for this observed regression; it does not establish
universal output equivalence. Eight concurrent retrieval requests, reasoning
separation, streamed tools, and tool-result continuation also passed in the
subsequent TP8 trial with sparse prefill and distributed sampling.

## Validation

The sparse MLA tests passed 53 split/sentinel cases and 27 independent
BF16/FP8 reference cases. The fused normalization/RoPE suite passed 68 cases
with four skips. Three pipeline/sequence-parallel cases and eleven indexer
workspace/chunking cases passed.
The updated attention kernel passed 95 cases, including non-power-of-two
top-k. Four independent dense-prefill reference cases and two mixed-batch
decode reference cases passed. Indexer buffer reuse passed a zero-allocation
regression and exact two-rank top-k comparisons; the 20 selected indexer
kernel/distributed cases passed. Routing checks passed 19 cases with seven
hardware-dependent skips.
The scale-conversion fix passed 18 value/dtype cases and a multi-chunk GPU
memory regression requiring less than 64 MiB of temporary allocation for
five million scales.

Small dummy-weight engines passed graph capture and requests at 32, 4,096,
and 7,900 input tokens, plus four concurrent 2,048-token requests. Coverage
includes BF16 KV, FP8 KV, and a two-stage pipeline split inside a shared-index
group. These small engines validate integration, not model quality or speed.

The checkpoint's MTP layer contains approximately 18.54 GiB of BF16 weights,
including its experts. The MTP pilot preserves the checkpoint's complete
quantization configuration and adds `model.layers.78.*` to its exclusions.
Its pipeline partition reserves space for those draft weights on the final
stage. The original TP8 baseline does not use speculative decoding.
