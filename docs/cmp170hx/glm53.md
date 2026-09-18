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

## Validation

The sparse MLA tests passed 53 split/sentinel cases and 27 independent
BF16/FP8 reference cases. The fused normalization/RoPE suite passed 68 cases
with four skips. Three pipeline/sequence-parallel cases and eleven indexer
workspace/chunking cases passed.

Small dummy-weight engines passed graph capture and requests at 32, 4,096,
and 7,900 input tokens, plus four concurrent 2,048-token requests. Coverage
includes BF16 KV, FP8 KV, and a two-stage pipeline split inside a shared-index
group. These small engines validate integration, not model quality or speed.

The checkpoint's MTP layer contains approximately 18.54 GiB of BF16 weights,
including its experts. Speculative decoding requires a corresponding memory
budget and unquantized layer configuration; it is not part of this baseline.
