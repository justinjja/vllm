# CMP 170HX build and serving guide

This fork packages the qualified `20260917-context1m-adaptive` local release
for DeepSeek V4.1 Flash on NVIDIA CMP 170HX (Ampere SM80). It is based on
upstream commit `c16bb6068f70878fb8a2f7c4d6cda95cd03a778b`.
The [main README](../../README.md) contains the ongoing progress ledger.

## What is included

- Software E4M3 FP8 conversion and Ampere sparse attention/indexer kernels.
- Shared DeepSeek V4.1 KV, index, and candidate state across pipeline stages,
  including incremental cache transfers, prefix reuse, and speculative updates.
- Candidate-only indexer scoring, TP query sharding, and empty-tile trimming.
- Adaptive prefill: 1,024-token chunks for prompts up to 65,536 tokens and an
  8,192-token batch budget for longer prompts. Context remains 1,048,576 tokens.
- Pipeline cohort scheduling, worker/block-table fixes, regression tests,
  and kernel/hardware benchmarks under `benchmarks/kernels/`.

The imported release also contains opt-in experimental hybrid TP/EP and
collective paths. The qualified serving recipe uses **TP2/PP4**, with layer
partition **11/9/10/10**, the Triton sparse backend, and three probabilistic
DSpark draft tokens. Other configurations are not covered by these results.

## Recorded environment

The September 17 qualification used eight 64-GiB CMP 170HX GPUs, Python 3.12,
PyTorch 2.13.0+cu130, Triton 3.7.1, CUDA toolkit 13.0, and driver 610.43.02.
The model was `deepseek-ai/DeepSeek-V4.1-Flash` at revision
`dba1be0a40aa45a94ad051997016db3960a90277`.

GPU pairs 0/1 and 2/3 were on NUMA node 0; pairs 4/5 and 6/7 were on node 1.
The serving rank order was `0,1,4,5,6,7,2,3`. Physical GPUs 4 and 5 were capped
at 1,200 MHz, with automatic graphics clocks on the other six cards. Memory
clocks were 1,728 MHz, power limits 250 W, and CPUs used the performance
governor. The example does not change system clocks or install a service.

## Build

Build this checkout from source to retain its changes. The local qualification
used a full CUDA source build because precompiled metadata for the base commit
was unavailable. Native libraries, model weights, caches, and host-specific
service files are not included in this repository.

```bash
git clone --branch cmp170hx-dsv41 https://github.com/justinjja/vllm.git
cd vllm
uv venv --python 3.12
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-13.0
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.0
export CMAKE_BUILD_TYPE=Release
export MAX_JOBS=32
export NVCC_THREADS=2
uv pip install -r requirements/build/cuda.txt --torch-backend=cu130
uv pip install -e . --no-build-isolation --torch-backend=cu130
```

Adjust compiler paths and build parallelism for the machine. For Rust frontend
artifacts, install the toolchain specified in `rust-toolchain.toml`, then run
`uv run --no-project --python .venv/bin/python tools/build_rust.py --release`.
See the [incremental compilation guide](../contributing/incremental_build.md)
for subsequent native changes. The publication process reused the already
built local native libraries for checks; it did not repeat a clean build.

## Serve

Inspect [the serving example](../../examples/cmp170hx/serve.sh) and adapt its
GPU and NUMA mapping to the machine before launching. It preserves the
qualified inference settings, defaults to localhost, and accepts additional
vLLM arguments. `MODEL_PATH` can select an existing local checkpoint snapshot.

```bash
source .venv/bin/activate
bash examples/cmp170hx/serve.sh
```

The recipe enables engram CPU offload, GPU memory utilization 0.80, maximum
32 sequences, a 4-GiB shared-cache transfer budget, and decode CUDA graphs.
Large prefill buffers leave approximately 1,151,443 KV tokens of capacity in
the measured configuration. Concurrent million-token requests may queue or
preempt. The 65,536-token threshold selects prefill chunk size only.

## Results and limits

The [qualification record](qualification-20260917.json) retains measured
metrics, every GUI timing sample, reference-comparison counts, and the
semantic review from September 17. It is a selected export of the historical
local record, identified by its original SHA-256 hash. It omits host paths,
device UUIDs, and raw prompts. The
[source manifest](source-manifest-20260917.json) records all 79 original
source/test/benchmark hashes. Runtime source matches the frozen release byte
for byte. One test now uses `torch.accelerator.synchronize()` to satisfy the
repository's lint policy; its publication hash is recorded separately.

| Measurement | Result |
| --- | ---: |
| Fresh 1,040,109-token retrieval, including short answer | 314.17 s |
| Previous full-length retrieval | 1,024.62 s |
| Full-length speedup | 3.26× |
| Fresh approximately 262K-token retrieval | 54.53 s |
| Warmed GUI generation median, three runs | 118.25 tokens/s |
| Warmed GUI 30K prefill median, three runs | 5.78 s |
| Full GUI loop generation median, three runs | 112.15 tokens/s |
| Full GUI loop aggregate median, concurrency 20 | 757.63 tokens/s |
| Full GUI loop 30K prefill median | 6.08 s |

The full GUI protocol interleaves concurrent requests; the separate warmed
protocol does not. Full GUI aggregate throughput was approximately 5% lower
than the preceding full-context release. The original 200 tokens/s generation
target remains unmet.

The same-commit stock backend rejects SM80. Comparison therefore used a local
SM80 compatibility reference with independent FP32 attention/indexer operations.
Final qualification passed 20/20 objective answers and all retrieval checks,
with 20/27 exact generations and 146/149 fixed-history token choices. Seven
differing generations and three tied-score choices were reviewed. The known
reference reversal error can vary with batch shape, and its list/tuple
overgeneralizations remain; arbitrary prompts are not guaranteed equivalent.
No new 256-question math evaluation was run for this release.

## Validation

Publication checks on September 18 passed 75 selected pipeline, cache-transfer,
hybrid-layout, and scheduler cases, plus nine adaptive prefill scheduler cases.
The synchronization test was rerun after its accelerator API update. Applicable
pre-commit checks passed. These checks reused the qualified native libraries;
full-model benchmarks were not repeated for publication.

Historical checks include 158 targeted initial kernel/quantization/cache tests,
candidate-score bitwise comparisons, two-GPU TP query-sharding checks across
eight combinations, pipeline cache relay/graph/prefix-reuse tests, and eleven
adaptive scheduler cases. The first candidate-score run used an overly strict
ordered-top-k assertion: 100 passed and eight failed. The corrected eight
cases passed by comparing selected sets, since native ordering is nondeterministic.

Relevant suites are in `tests/models/test_deepseek_v41_pipeline*.py`,
`tests/distributed/test_deepseek_v41_pipeline_transfer.py`,
`tests/kernels/test_ampere_mqa_logits.py`,
`tests/kernels/attention/test_ampere_sparse_mla.py`, and the modified scheduler,
worker, and FP8 quantization tests. Run GPU suites on idle compatible hardware.
For the adaptive scheduler checks after installing test dependencies:

```bash
uv pip install -r requirements/test/cuda.in --torch-backend=cu130
uv run --no-project --python .venv/bin/python -m pytest \
  tests/v1/core/test_scheduler.py \
  -k 'prompt_length_selects or short_and_long_prefills' -q
```

## Updating the ledger

Keep dated entries at the top of the main README, directly below the CMP 170HX
note, newest first. Record the change, test or benchmark evidence, hardware
and configuration differences, and unresolved limits. Distinguish new runs
from historical measurements. Preserve the September 17 source manifest and
qualification as the baseline when publishing later changes.
