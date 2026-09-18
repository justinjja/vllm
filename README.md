# vLLM for NVIDIA CMP 170HX GPUs

**This is an optimized build of vLLM for NVIDIA CMP 170HX GPUs**, with
Ampere SM80 kernels and serving improvements for DeepSeek V4.1 Flash and
GLM-5.3 NVFP4. The qualified DeepSeek configuration uses eight 64-GiB cards
with TP2/PP4.

## Progress ledger

Performance entries below were measured on the local eight-GPU system.

| Date (UTC) | Progress | Validation / result |
| --- | --- | --- |
| 2026-09-18 | Reproduced independent compute errors on `b1` and tested a lower core-clock limit. | GPU Burn flagged `b1` outside vLLM, after reset/reload and when tested alone. A verified 900 MHz limit passed a 60-second single-card trial and a five-minute eight-GPU trial, with zero mismatches. This is a candidate workaround; reliable model serving remains unqualified. [Compute-integrity evidence](docs/cmp170hx/glm53.md#independent-compute-corruption). |
| 2026-09-18 | Tested a temporary 1,350 MHz core-clock limit on `b2`. | Short checks and fresh plus four cached 262K requests passed, but the first concurrency-sixteen run stalled with the same `b2` GSP timeout and firmware kernel panic. The clock limit did not establish reliable serving. [Clock-control trial](docs/cmp170hx/glm53.md#core-clock-control-trial). |
| 2026-09-18 | Swapped the two GPU pairs on the second CPU between GLM pipeline stages. | Short checks and fresh 262K retrieval passed, but the first cached request stalled. The same physical card, `b2`, reported a GSP timeout and firmware kernel panic in its new stage. This narrows the investigation but does not distinguish a driver/firmware defect from a card defect. The benchmark was withheld. [Stage-swap trial](docs/cmp170hx/glm53.md#pipeline-stage-swap-trial). |
| 2026-09-18 | Investigated recurring GLM failures using transport, eager-execution, and driver-reset controls. | Peer and repeated host controls failed at concurrency sixteen; eager execution also failed on cached context. The preferred configuration failed again after a full reset/reload, with a power-management firmware halt on `b2`. Recovery required no reboot. Independent near-capacity BF16 compute and NCCL checks passed on all eight GPUs; the cause remains unresolved. [Comparison and diagnostics](docs/cmp170hx/glm53.md#nccl-transport-comparison). |
| 2026-09-18 | Tested two-way GLM context sharding and investigated recurring cached-context crashes. | Fresh million-token retrieval passed in 1,252.55 s to first token; cached generation crashed, as did the unchanged BF16 reference after recovery. Expanded local-memory and PIX peer checks passed on all eight GPUs. Disabling NCCL peer transport and custom all-reduce initially passed fresh plus four cached 262K requests; subsequent concurrent failures are recorded above. [Trial and recovery](docs/cmp170hx/glm53.md#context-sharding-trial). |
| 2026-09-18 | Compared one BF16 MTP draft against two in the full GLM model. | The one-draft trial passed 40/40 short objective answers, reasoning/tools, 8/8 concurrent retrieval checks, and fresh 262K retrieval. It showed no consistent speed gain, and cached retrieval returned empty final content with the answer only in reasoning. Two drafts remain preferred. [Comparison](docs/cmp170hx/glm53.md#one-draft-schedule-comparison). |
| 2026-09-18 | Evaluated INT8 draft experts in the full eight-GPU GLM model. | Draft expert storage fell from 9.0 to 4.57 GiB per TP2 shard; 40/40 objective answers, reasoning/tools, and 8/8 concurrent retrieval checks passed. Matched runs did not establish a consistent aggregate gain, and fresh 262K-context evaluation failed with a CUDA error. BF16 draft weights remain preferred. [Full-model comparison](docs/cmp170hx/glm53.md#full-model-draft-int8-trial). |
| 2026-09-18 | Recovered all eight GPUs and tested expanded driver BAR1 peer eligibility. | The candidate passed 64/64 small allocation/read checks versus 52/64 with the original, but failed a near-capacity integrity check and was rolled back. All eight CUDA computation checks and all eight directed PIX-pair integrity checks passed after rollback. The intermittent lowercase-answer failure remains documented. [Driver trial and recovery](docs/cmp170hx/glm53.md#driver-peer-access-trial). |
| 2026-09-18 | Measured INT8 and NVFP4 compression prototypes for the BF16 draft experts. | Isolated INT8 expert calculations ran 1.65–2.06× faster and reduced TP2 expert storage from 9.0 to 4.57 GiB per shard. A draft-only INT8 prototype passed loading, graphs, and seven random-weight requests at both TP1 and TP2. The later full-model comparison is recorded above; the recipe is unchanged. [Measurements and limits](docs/cmp170hx/glm53.md#draft-expert-compression-measurements). |
| 2026-09-18 | Tested a more balanced PP8 partition for full-context prefill twice. | Fresh retrieval from 1,046,659 input tokens passed in 697.38 s and 711.24 s to first token, versus 892.51 s with the prior partition. Short quality checks passed in both trials, but both cached requests failed, with a CUDA launch error and a firmware timeout respectively. This candidate is **not qualified**; the validated recipes remain preferred. [Trial record](docs/cmp170hx/glm53.md#balanced-partition-trial). |
| 2026-09-18 | Added profiled FP16 conversion workspace for large SM80 indexer prefills. | Matched fresh retrieval from 1,046,659 input tokens fell from 1,580.62 s to 892.51 s first-token latency (**1.77× faster**), with all three records correct. Cached retrieval, 20/20 objective checks, reasoning/tools, and 8/8 concurrent retrieval checks passed. The optional PP8 path uses 264 MiB more workspace and retains full-context capacity. [Measurements and configuration](docs/cmp170hx/glm53.md#predecoded-full-context-prefill). |
| 2026-09-18 | Qualified fresh and cached GLM retrieval from 1,046,659 input tokens with PP8 and FP8 KV. | Both runs returned all three records correctly; 20/20 objective checks, reasoning/tools, and 8/8 concurrent retrieval checks passed. Fresh first-token latency was 1,580.62 s; cached latency was 5.32 s. This establishes full-context retrieval capacity; fresh prefill remains slow and TP2/PP4 remains preferred for generation. [Configuration and measurements](docs/cmp170hx/glm53.md#full-context-pp8-profile). |
| 2026-09-18 | Added an optional symmetric pair-tree serving backend with explicit physical GPU mapping and IPC lifetime handling. | Native graph/reference tests passed on both four-GPU groups. GLM scored 39/40 sequential/concurrent objective answers; reasoning/tools and 8/8 retrieval checks passed. The concurrent lowercase failure leaves this backend experimental; the throughput benchmark was withheld and TP2/PP4 remains preferred. [Evaluation](docs/cmp170hx/glm53.md#optional-pair-tree-backend). |
| 2026-09-18 | Prototyped a symmetric PCIe pair-tree reduction for small GLM TP4 batches. | 40/40 sequential/concurrent objective checks, reasoning/tools, and 8/8 retrieval checks passed. Matched single generation rose from 48.13–51.59 to 60.58–67.80 tokens/s. Aggregate throughput and prefill do not displace TP2/PP4. Standalone source and benchmarks are included; serving-backend integration remains experimental. [Results](docs/cmp170hx/glm53.md#symmetric-pair-tree-prototype). |
| 2026-09-18 | Measured current GLM TP4/PP2 and isolated its increased collective cost. | 20/20 objective checks, reasoning/tools, and 8/8 concurrent retrieval checks passed. Matched runs measured 48.13–51.59 single and 336.20–339.66 aggregate tokens/s at concurrency sixteen; 8K prefill TTFT was 6.50–6.54 s. TP2/PP4 remains preferred. Profiling found closely matched rank times within both four-GPU groups on the symmetric topology. [Comparison and diagnostic](docs/cmp170hx/glm53.md#tp4pp2-comparison). |
| 2026-09-18 | Reduced register spilling in SM80 software FP8 cache decoding. | 28 reference tests passed; sparse-MLA kernel time fell 25–39% in six tested shapes. Paired real-model runs found zero bit differences across 30.6 billion attention outputs and scored 79/80 objective answers. End-to-end FP8 speedup and usable 458K-context output remain unqualified. [Evidence and limitations](docs/cmp170hx/glm53.md#software-fp8-conversion). |
| 2026-09-18 | Fixed invalid autoregressive draft graph shapes under dynamic speculative-token schedules. | 37 graph/scheduling tests passed. A random-weight GLM engine with a four/two-draft schedule passed sequential generation, chunked prefill, 7,900-token input, and four concurrent requests. This validates engine integration, not model quality or a throughput gain. The qualified recipe retains two drafts. |
| 2026-09-18 | Compared GLM TP8 expert parallelism against the retained TP2/PP4 configuration. | 20/20 objective checks, reasoning/tools, and 8/8 concurrent retrieval checks passed. Two matched runs measured 52.80–55.47 single and 293.98–303.63 aggregate tokens/s at concurrency sixteen; 8K prefill TTFT was 8.93 s. TP2/PP4 retains better aggregate throughput, prefill, and context capacity. [Measurements](docs/cmp170hx/glm53.md#sm80-attention-improvements). |
| 2026-09-18 | Restored absorbed sparse MLA prefill after isolating a generated-answer regression in dense short-prefill. | Objective checks returned to 20/20 twice and the isolated lowercase check passed 3/3. Fresh and cached retrieval recovered all three records from 260,222 input tokens with BF16 KV. Matched 512-output measurements retained 349.86–360.41 aggregate tokens/s at concurrency sixteen; single generation was 48.01–54.47. [Details](docs/cmp170hx/glm53.md#sm80-attention-improvements). |
| 2026-09-18 | Reused SM80 indexer score storage and reserved more prefill headroom for GLM high context. | Fresh and cached retrieval recovered all three records from 260,223 input tokens with BF16 KV; TTFT was 137.51 s and 1.48 s. Eight concurrent retrieval checks and tools passed. Broader objective checks scored 19/20 twice; the subsequently isolated dense-prefill regression is [documented](docs/cmp170hx/glm53.md#observed-output-limitations). |
| 2026-09-18 | Improved SM80 sparse MLA occupancy and short-context prefill; added accepted-token streaming measurements. | In 512-input/128-output pilots, concurrency-16 generation rose from 146.03 to 193.97 aggregate tokens/s. Two longer 512-output runs with the final memory setting measured 51.41–61.17 single, 263.65–272.30 aggregate at concurrency eight, and 341.97–356.32 at concurrency sixteen during steady decode. Single generation remains below target. See the [measurements and context limitations](docs/cmp170hx/glm53.md#sm80-attention-improvements). |
| 2026-09-18 | Compared GLM TP8, TP4/PP2, and TP2/PP4; bounded NVFP4 scale-conversion memory to enable PP8 loading. | TP2/PP4 pilot: 35.24 single decode tokens/s, 157.13 aggregate tokens/s at concurrency 16, and 12.50 s TTFT for 30K input tokens. Reasoning, tools, long-document retrieval, and eight mixed-length concurrent retrieval checks passed. The loading fix passed 19 scale/memory cases; PP8 starts with BF16 KV and 411,776 cache tokens. Tuning continues. |
| 2026-09-18 | Enabled GLM-5.3 NVFP4 sparse attention on Ampere with BF16 or FP8 KV; added shared-index transfer across pipeline stages. | 148 kernel cases, 14 model/workspace cases, and three engine integration configurations passed. Real TP8/BF16 passed reasoning, streamed tools, tool continuation, and long-document retrieval. Initial baseline: 36.05 decode tokens/s, 107.94 aggregate tokens/s at concurrency 16, and 37.12 s TTFT for 30K input tokens. Layout and high-context tuning are in progress; see the [GLM record](docs/cmp170hx/glm53.md). |
| 2026-09-18 | Published the qualified September 17 source, regression tests, kernel benchmarks, and portable serving recipe on the public fork. | Verified 79 imported files against the frozen manifest; runtime source unchanged. Updated one test to use the required accelerator API. 84 targeted cases passed; based on upstream `c16bb6068f70878fb8a2f7c4d6cda95cd03a778b`. |
| 2026-09-17 | Added incremental pipeline cache transfers, candidate-only indexer scoring, TP query sharding, and adaptive prefill chunks. | 1,040,109 input tokens processed in 314.17 s versus 1,024.62 s: **3.26× faster**, with all three retrieval records correct. |
| 2026-09-17 | Qualified the 1,048,576-token context configuration with TP2/PP4 and three DSpark draft tokens. | Warmed generation **118.25 tokens/s**; 30K prefill **5.78 s**. Complete GUI loops: 112.15 single-response tokens/s and 757.63 aggregate tokens/s at concurrency 20. |
| 2026-09-17 | Completed compatibility-reference, cached/mixed retrieval, and scheduler qualification. | 20/20 objective answers; 20/27 exact generations; 146/149 fixed-history token choices; six cached/mixed retrieval checks passed. Output differences are documented. |
| 2026-09-16 | Built SM80 support: software FP8 conversion, Ampere sparse attention/indexer kernels, pipeline sharing, and scheduler/worker fixes. | Source build and 158 targeted kernel, quantization, cache, and scheduling tests passed in the local qualification. |

For DeepSeek V4.1, the **200 generation tokens/s target remains unmet**. Its KV capacity
is approximately 1.15M tokens, prioritizing one full-context response. These
measurements do not establish universal output equivalence or performance on
other hardware. The reference was a local SM80 compatibility implementation.

See the [build and serving guide](docs/cmp170hx/README.md),
[qualification record](docs/cmp170hx/qualification-20260917.json), and
[source manifest](docs/cmp170hx/source-manifest-20260917.json).

---

<!-- markdownlint-disable MD001 MD041 -->
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vLLM website to help you get started with vLLM. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has grown into one of the most active open-source AI projects built and maintained by a diverse community of many dozens of academic institutions and companies from over 2000 contributors.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests, chunked prefill, prefix caching
- Fast and flexible model execution with piecewise and full CUDA/HIP graphs
- Quantization: FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO, and [more](https://docs.vllm.ai/en/latest/features/quantization/index.html)
- Optimized attention kernels including FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, and Triton
- Optimized GEMM/MoE kernels for various precisions using CUTLASS, TRTLLM-GEN, CuTeDSL
- Speculative decoding including n-gram, suffix, EAGLE, DFlash
- Automatic kernel generation and graph-level transformations using torch.compile
- Disaggregated prefill, decode, and encode

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data, expert, and context parallelism for distributed inference
- Streaming outputs
- Generation of structured outputs using xgrammar or guidance
- Tool calling and reasoning parsers
- OpenAI-compatible API server, plus Anthropic Messages API and gRPC support
- Efficient multi-LoRA support for dense and MoE layers
- Support for NVIDIA GPUs, AMD GPUs, Intel GPUs, and x86/ARM/PowerPC CPUs. Additionally, diverse hardware plugins such as Google TPUs, Intel Gaudi, IBM Spyre, Huawei Ascend, Rebellions NPU, Apple Silicon, MetaX GPU, and more.

vLLM seamlessly supports 200+ model architectures on Hugging Face, including:

- Decoder-only LLMs (e.g., Llama, Qwen, Gemma)
- Mixture-of-Expert LLMs (e.g., Mixtral, DeepSeek-V3, Qwen-MoE, GPT-OSS)
- Hybrid attention and state-space models (e.g., Mamba, Qwen3.5)
- Multi-modal models (e.g., LLaVA, Qwen-VL, Pixtral)
- Embedding and retrieval models (e.g., E5-Mistral, GTE, ColBERT)
- Reward and classification models (e.g., Qwen-Math)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`:

```bash
uv pip install vllm
```

Or [build from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source) for development.

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
