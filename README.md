# vLLM for NVIDIA CMP 170HX GPUs

**This is an optimized build of vLLM for NVIDIA CMP 170HX GPUs**, with
Ampere SM80 kernels and serving improvements for DeepSeek V4.1 Flash and
GLM-5.3 NVFP4. The qualified DeepSeek configuration uses eight 64-GiB cards
with TP2/PP4.

## Progress ledger

Performance entries below were measured on the local eight-GPU system.

| Date (UTC) | Progress | Validation / result |
| --- | --- | --- |
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
