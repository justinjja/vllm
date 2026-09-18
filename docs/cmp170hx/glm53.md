# GLM-5.3 NVFP4 on CMP 170HX

`local-inference-lab/GLM-5.3-NVFP4` runs on eight 64-GiB CMP 170HX GPUs with
the Triton sparse MLA backend and Marlin NVFP4 experts. BF16 and software FP8
KV paths pass kernel and engine integration checks. TP2/PP4 with BF16 KV and
two MTP drafts remains preferred for generation. A separate PP8/FP8 profile
passes fresh and cached retrieval from 1,046,659 input tokens, although its
fresh prefill is slow; see the [full-context measurements](#full-context-pp8-profile).
Subsequent cached-context and concurrent-generation crashes also affected the
BF16 profile. The [context-sharding trial](#context-sharding-trial) and
[transport and recovery controls](#nccl-transport-comparison) retain those
failures alongside the earlier successful measurements. A full driver reset
and reload did not prevent recurrence; reliable serving remains unresolved.

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

Dynamic speculative-token schedules now preserve the autoregressive drafter's
one-query-per-request graph shape. Previously, applying the target's schedule
to those draft steps could create negative query lengths or divide by zero
during startup. The correction passes 37 graph/scheduling tests and a
random-weight GLM engine check with sequential, concurrent, chunked-prefill,
and 7,900-token inputs. This establishes integration correctness; the
preferred TP2/PP4 serving recipe continues to use two fixed draft tokens.

## Driver peer-access trial

The eight-card physical topology is symmetric: four PIX pairs, with four GPUs
per CPU. A temporary change to the custom driver's final BAR1 eligibility
check expanded its pair-only fallback to the existing eight-device allowlist.
After device recovery, the candidate passed all 64 small allocation/read cases.
A fresh load of the original reproduced 52/64; reloading the candidate and
reversing CUDA visibility each passed 64/64. This isolates a difference in the
driver's mapping behavior on the symmetric hardware.

The candidate **failed large-allocation qualification**. Each directed test
allocated 62.75 GiB and checked one word per 64-KiB page. Fifty cases passed;
owner `db`, reader `1b` then returned 800 incorrect words in allocation chunk
147. The five remaining cases and collective benchmarks were withheld.
An immediate repeat passed, leaving an intermittent integrity failure with
unresolved cause. The candidate was unloaded and is not used for serving.

The original driver was restored with identical parameters and unchanged
installed modules. All eight CUDA allocation/kernel/reduction checks and all
eight directed PIX-pair integrity checks passed. No reboot was needed.
The preferred TP2/PP4 service resumed with reasoning, tools, and 8/8 concurrent
retrieval checks passing. Objective scores were 19/20 at concurrency four,
20/20 sequentially, and 20/20 on a concurrent repeat. The first run again
answered `computer` instead of `compute`; the passing repeats do not erase that
failure. Fresh and cached retrieval from 260,224 input tokens both returned all
three records, with first-token latencies of 143.36 s and 1.46 s respectively.
Matched speed benchmarks were withheld. These results do not establish
that the peer patch causes or fixes the earlier cached-context firmware crash.
See the [complete trial record](glm53-driver-peer-trial-20260918.json).

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

Enabling expert parallelism on the same TP8 configuration also passed 20/20
objective checks, reasoning/tools, and 8/8 concurrent retrieval checks.
Two matched runs measured 52.80–55.47 single, 218.57–224.91 aggregate at
concurrency eight, and 293.98–303.63 at concurrency sixteen. The 8K prefill
TTFT was 8.93 seconds, and BF16 KV capacity was 40,000 tokens. This trial
does not improve the preferred TP2/PP4 operating point; its exact results
are included in the [measurement record](glm53-attention-20260918.json).

### TP4/PP2 comparison

The current kernels with TP4/PP2, partition `42,36`, two MTP drafts, BF16 KV,
local draft argmax reduction, and batch-sharded sampling passed 20/20 objective
checks, reasoning/tools, and 8/8 concurrent retrieval checks. Its configured
context limit was 65,536, with 120,320 cache tokens. Two repetitions used the
same prompt hashes as the retained TP2/PP4 benchmark:

| Measurement | TP2/PP4 | TP4/PP2 |
| --- | ---: | ---: |
| Single steady accepted tokens/s | 48.01–54.47 | 48.13–51.59 |
| Concurrency-eight steady accepted tokens/s | 252.54–265.90 | 237.63–244.19 |
| Concurrency-sixteen steady accepted tokens/s | 349.86–360.41 | 336.20–339.66 |
| 8K prefill TTFT, seconds | 5.10–5.20 | 6.50–6.54 |

A separate CUDA-event diagnostic measured the three-query decode graph.
Averaging ranks within each stage and summing the serial stages, attention
and MLP intervals fell from 36.33 ms with TP2 to 27.32 ms with TP4. The two
reduction-plus-normalization intervals rose from 3.87 ms to 15.00 ms. These
instrumented timings identify a collective cost; they include event overhead
and are separate from the serving benchmark. Nested intervals are not additive.

The hardware topology is symmetric: PIX pairs 0–1, 2–3, 4–5, and 6–7,
with four GPUs per CPU. Total layer intervals closely matched within each
TP4 group: 23.35–23.37 ms on the first stage and 20.26–20.31 ms on the
second. This diagnostic does not establish a GPU 7-specific bottleneck.
TP2/PP4 remains the preferred combined throughput, prefill, and context
configuration. Exact samples, configuration, and diagnostic rank timings
are in the [TP4 record](glm53-tp4-20260918.json).

### Symmetric pair-tree prototype

A standalone four-rank collective uses the same two-pair structure on each
CPU group. It sums each pair in FP32, exchanges partial sums between pair
leaders, and rounds the final sum to BF16. Payloads are pushed to their
destination, and synchronization flags are polled in local GPU memory.
Its arithmetic differs from NCCL's BF16 reduction order.

Independent CPU-reference checks passed on both groups for six row counts
and four launch sizes, including changed inputs, repeated CUDA graph replay,
and uneven rank timing. With two blocks, three-row reductions measured
51.33/51.90 microseconds on the two groups, versus NCCL's 86.08/89.98.
Larger tensors favored NCCL.

A private worker-extension trial selected the prototype only for contiguous
BF16 tensors with 6,144 columns and at most six rows. It passed 20/20
objectives at concurrency four, another 20/20 sequentially, reasoning/tools,
and 8/8 concurrent retrieval checks. Compared with the same TP4/PP2 setup
above, matched single-request steady decode rose to **60.58–67.80 tokens/s**,
a 26–31% improvement. Concurrency-eight throughput was 241.29–243.21 and
concurrency-sixteen throughput was 327.07–337.27 tokens/s. The 8K prefill
TTFT was 6.45–6.50 seconds. TP2/PP4 remains preferred for the combined
throughput, prefill, and context requirements.

The [measurement record](glm53-pair-tree-20260918.json) contains configurations,
samples, and limitations. The standalone implementation and benchmark are
included for reproduction. The optional serving integration is evaluated
separately below; the normal serving recipe does not enable this prototype.

```bash
nvcc -O3 -std=c++17 -arch=sm_80 -shared -Xcompiler=-fPIC \
  benchmarks/kernels/cmp_pair_tree_reduce.cu -o /tmp/cmp_pair_tree.so
uv run --python .venv/bin/python benchmarks/kernels/benchmark_cmp_pair_tree.py \
  --gpus 0,1,2,3 --library /tmp/cmp_pair_tree.so --output pair-tree-node0.json
uv run --python .venv/bin/python benchmarks/kernels/benchmark_cmp_pair_tree.py \
  --gpus 4,5,6,7 --library /tmp/cmp_pair_tree.so --output pair-tree-node1.json
```

These commands assume the eight-card device ordering and qualified PIX pairs
described above, with no other GPU workload running.

### Optional pair-tree backend

The serving communicator now has an opt-in implementation of the symmetric
pair tree. It resolves configured physical GPU pairs independently of process
rank order and uses the same algorithm on both CPU groups. Its separate SM80
library uses 64-bit sequence counters and closes imported IPC handles on every
rank before freeing the owning allocations. Startup checks actual transfers;
a missing library or failed IPC import disables the backend for the whole TP
group. A CUDA execution fault still terminates the worker.

The backend accepts BF16 tensors with 6,144 columns and at most six rows.
Larger reductions use the existing dispatch path. It requires serialized
execution, disables itself with microbatch overlap or batch invariance, and
is off by default. The normal TP2/PP4 recipe remains unchanged.

The native regression tests passed on both four-GPU groups with scrambled
process ranks, different input layouts across ranks, changing graph inputs,
repeated construction/destruction, and sequence counters crossing 2^32.
Injected missing-library and partial IPC-import failures also fell back
collectively and released their allocations.

These checks kept all eight GPUs visible in their normal order. Reversing
`CUDA_VISIBLE_DEVICES` failed during CUDA initialization, before the backend
was constructed, so that visibility configuration is unqualified. This
software failure does not establish any physical connectivity difference.

Build only this optional library from an existing source installation:

```bash
cmake -S csrc/cmp_collectives -B build/cmp-pair-tree \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$PWD"
cmake --build build/cmp-pair-tree --target _cmp_pair_tree
cmake --install build/cmp-pair-tree --component _cmp_pair_tree
```

For the qualified physical ordering, the additional configuration keys are:

```json
{
  "cmp_pair_tree_max_rows": 6,
  "cmp_pair_tree_groups": [[0, 1, 2, 3], [4, 5, 6, 7]]
}
```

Each group lists two adjacent PIX pairs in physical GPU order. Merge these
keys into the existing `--additional-config` object for a TP4/PP2 trial.

The integrated backend started and captured the main/draft model graphs on
all eight GPUs with TP4/PP2, two MTP drafts, BF16 KV, and a 65,536-token limit.
It passed 20/20 sequential objective answers and 19/20 at concurrency four,
plus reasoning/tools and all eight retrieval checks. The concurrent failure
lowercased `CoMPuTe` as `computer`. That error has appeared in other tested
configurations, but these results do not establish its cause. The failed
objective gate withheld the serving benchmark; the earlier prototype's
timings are not a throughput claim for this integration. It remains
experimental. See the [runtime evaluation](glm53-pair-tree-runtime-20260918.json).

### Software FP8 conversion

The SM80 byte decoder now expands E4M3 through an exact FP16 representation
before FP32 scaling. All 256 encodings match the CPU reference, including
subnormals, signed zero, and NaNs. This reduces the compiled sparse-prefill
kernel's per-thread stack from 440 to 16 bytes. In six isolated sparse-MLA
shapes, three alternating timing rounds measured 25–39% less kernel time;
the 2,048-query case fell from 30.06 to 19.05 ms. These are component timings.
The [FP8 record](glm53-fp8-20260918.json) contains configurations and samples.

A paired real-model diagnostic ran both conversions on each identical
attention input and selected either output for subsequent model execution.
Across 478,400 calls and 30,627,987,456 BF16 output elements, there were zero
bit differences. Four objective suites scored 79/80: the revised conversion
scored 20/20 and 19/20, and the original scored 20/20 twice. The failed case
lowercased `CoMPuTe` as `computer`. The uninstrumented revised run also
scored 19/20, although three isolated retries passed. The paired check
rules out different attention values as the explanation on those inputs;
it does not establish universal generated-output equivalence. Reasoning,
tools, and eight concurrent retrieval checks passed.

Before this conversion change, FP8 KV with TP2/PP4 and two drafts provided
516,672 cache tokens at a configured 458,752-token limit. Matched runs
measured 43.83–50.09 single, 235.94–248.58 aggregate at concurrency eight,
and 304.03–339.94 at concurrency sixteen. The 8K prefill TTFT was
7.52–7.76 seconds. A 456,831-input-token low-effort request completed
without exhausting memory, but both fresh and cached responses placed all
three correct codes only in reasoning and left the final answer empty.
Those tests failed usable-answer qualification. Fresh TTFT was 411.56 s;
cached TTFT was 2.48 s.

The revised run's failed objective gate withheld its serving benchmark and
Max-effort long-context test. An end-to-end FP8 speedup and usable output
at 458K remain unqualified for that TP2/PP4 trial. The separate PP8/FP8
full-context result is reported below. BF16 TP2/PP4 remains preferred for
generation.

### Throughput targets and qualified context

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
qualified BF16 high-context recipe. These BF16 measurements cover the
262,144-token setting; the full-context PP8 results follow.

### Full-context PP8 profile

TP1/PP8 with FP8 KV admits the model's 1,048,576-token context setting and
allocates 1,100,928 cache tokens. On the same checkpoint and eight-GPU system,
fresh and cached requests each retrieved all three synthetic records from
1,046,659 input tokens. The records were inserted near 10%, 50%, and 90% of
the document. Both responses contained the correct JSON answer in the final
content, and both finished normally within a 1,024-token output budget.
The requests used temperature zero and maximum reasoning effort.

| Request | First-token latency | Total latency | Output tokens | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Fresh prefix | 1,580.62 s | 1,590.08 s | 195 | 20.70 |
| Cached prefix | 5.32 s | 16.67 s | 225 | 19.88 |

The cached request intentionally reused the identical prompt. Generation
rates include reasoning tokens and describe these short answers, rather
than a matched steady-decode benchmark. The complete
[measurement record](glm53-full-context-20260918.json) includes prompt hashes,
record positions, final answers, and raw quality results. This configuration
also passed 20/20 objective checks, reasoning and streamed tool use with tool
continuation, and eight concurrent mixed-length retrieval requests.

This profile establishes usable retrieval near the advertised context limit.
The baseline fresh prefill took about 26 minutes; the
[predecoded variant](#predecoded-full-context-prefill) reduces it to about
15 minutes. Three-record retrieval does not establish general long-document accuracy.
The cache supports roughly one full-length request; `--max-num-seqs 16` does
not provide sixteen million-token contexts. TP2/PP4 below remains the
preferred generation profile.

The measured PP8 configuration uses no speculative model or worker extension:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export VLLM_PP_LAYER_PARTITION=12,10,10,10,9,9,9,9
export PYTORCH_ALLOC_CONF=pinned_max_round_threshold_mb:1024
uv run --python .venv/bin/python -m vllm.entrypoints.cli.main serve \
  local-inference-lab/GLM-5.3-NVFP4 \
  --revision b472e4ee53f6a9862da5486c56c6ca21be3dab70 \
  --served-model-name glm-5.3-nvfp4 \
  --tensor-parallel-size 1 --pipeline-parallel-size 8 \
  --dtype bfloat16 --kv-cache-dtype fp8 \
  --attention-config '{"backend":"TRITON_MLA_SPARSE"}' \
  --additional-config '{"sparse_indexer_max_prefill_tokens":1048576,"pipeline_max_batch_requests":8,"pipeline_batch_policy":"adaptive","pipeline_min_batch_requests":1}' \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
  --max-model-len 1048576 --max-num-batched-tokens 1024 --max-num-seqs 16 \
  --gpu-memory-utilization 0.978 --numa-bind --async-scheduling \
  --safetensors-load-strategy prefetch \
  --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice \
  --host 127.0.0.1 --port 8000
```

### Predecoded full-context prefill

For large SM80 prefills with 32 indexer heads, the optional
`sparse_indexer_predecode` path converts FP8 queries and keys to FP16 once
per scoring call. This avoids repeating software FP8 conversion inside
each matrix tile. The conversion buffers share the profiled workspace;
paged decode and prefills below 64 query rows or 65,536 keys retain their
existing paths. Tensor-parallel query sharding also supports the workspace.

On the same PP8 configuration and identical 1,046,659-token prompt, fresh
first-token latency fell **43.5%**, from 1,580.62 s to **892.51 s**
(1.77× faster). Both fresh and cached requests retrieved all three records
correctly. The fresh answer, including reasoning, matched the baseline's
text exactly. The configuration also passed 20/20 objective checks,
reasoning and streamed tools with continuation, and 8/8 concurrent retrieval
checks. The [complete record](glm53-predecoded-prefill-20260918.json) contains
source hashes, raw answers, kernel measurements, and the earlier rounding
diagnostic.

| Request | First-token latency | Total latency | Output tokens | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Fresh prefix | 892.51 s | 902.29 s | 195 | 19.99 |
| Cached prefix | 5.01 s | 14.75 s | 195 | 20.09 |

These are one fresh and one cached measurement per configuration. The
cached request deliberately reused the prompt. They establish synthetic
retrieval and a prefill improvement; they do not establish general accuracy
or a generation-throughput gain. TP2/PP4 remains preferred for generation.

The option is off by default. For this faster full-context PP8 variant,
replace the recipe above's `--additional-config` value with:

```json
{"sparse_indexer_max_prefill_tokens":1048576,"sparse_indexer_predecode":true,"pipeline_max_batch_requests":8,"pipeline_batch_policy":"adaptive","pipeline_min_batch_requests":1}
```

With 1,024 batched tokens and a 1,048,576-token prefill workspace, conversion
reserves **264 MiB**. Measured KV capacity falls from 1,100,928 to
**1,055,936 tokens**, retaining the full-context request. This memory cost
must be included when choosing a context limit on other configurations.

The integrated kernel benchmark includes query and key conversion costs:

| Query rows | KV tokens | Existing kernel | Predecoded kernel | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 262,144 | 47.27 ms | 15.66 ms | 3.02× |
| 128 | 1,048,576 | 47.30 ms | 16.73 ms | 2.83× |

Four alternating timing samples were taken for each path. Both benchmark
inputs produced identical scores and top-2048 membership. Conversion is
exact, but the MMA layout can change FP32 rounding: an earlier prototype
input changed one score by 3.81e-6, with unchanged top-k membership.
These checks do not establish universal output equivalence. Seventy-five
distinct kernel, distributed, layer, and workspace cases passed; seven
cases requiring other hardware were skipped.

```bash
PYTHONPATH=. uv run --no-project --python .venv/bin/python \
  benchmarks/kernels/benchmark_ampere_mqa.py --mode dense --heads 32 \
  --contexts 262144 1048576 --predecode --output predecode-kernels.json
```

### Balanced partition trial

Changing only the predecoded PP8 profile's layer partition to
`9,10,10,10,10,10,10,9` reduced fresh first-token latency from 892.51 s to
697.38 s on the identical 1,046,659-token retrieval prompt. All three records
were correct; total latency was 707.32 s with 196 output tokens. Startup
allocated 1,053,440 KV tokens, and 20/20 objective checks, reasoning/tools
with continuation, and 8/8 concurrent retrieval checks passed.

**This partition is not qualified.** The first cached request failed
during generation with an unspecified CUDA launch failure. The driver logged
Xid 32 and 31, and the engine shut down.

An unchanged repeat passed the short checks again and retrieved all three
records from the same full-context prompt: fresh first-token latency was
711.24 s and total latency was 718.53 s. Its cached request stalled after a
GSP firmware crash. The driver logged Xid 119 and 154; the server was stopped
and the waiting client was explicitly terminated. No completed cached
measurement or final answer was recorded. Application cleanup did not restore
the affected card to CUDA enumeration; no driver reset or reload was performed.

The root causes are unresolved, and the two failures have different
signatures. Neither establishes a physical connectivity difference. The
[trial record](glm53-balanced-prefill-trial-20260918.json) retains the successful
fresh measurements and both failed cached checks. The 20.3–21.9% fresh latency
reduction does not qualify this partition as a replacement for the
`12,10,10,10,9,9,9,9` full-context recipe above. TP2/PP4 remains preferred for
generation.

### Context-sharding trial

A private SM80 prototype split the KV cache across each TP2 pair and combined
partial sparse-attention outputs using their log-sum-exp normalizers. Its
attention, indexer, and workspace checks passed 170 cases, with seven skipped.
A small random-weight engine also passed sequential and concurrent generation,
fresh and cached 194K inputs, and an initially empty shard. These checks
establish implementation plumbing, not full-model quality or performance.

The full GLM trial used TP2/PP4, DCP2, partition `21,19,19,19`, FP8 KV,
predecoded indexer keys, a 1,024-token prefill budget, and no MTP drafts.
It allocated **1,419,392 cache tokens** and admitted the 1,048,576-token
context setting. Objective answers scored 19/20, repeating the known
`compute` → `computer` failure. Reasoning, tools, and all eight concurrent
retrieval checks passed.

Fresh retrieval returned all three records from the same 1,046,659-token
document used by the PP8 reference. First-token latency was **1,252.55 s**,
versus **892.51 s** for predecoded PP8: approximately 40% slower. The short
answer generated at 25.99 tokens/s after prefill; this is not a matched
steady-generation comparison.

**The prototype is not qualified.** Its cached request failed during decode.
The scheduler had recorded 58 output tokens when rank 5 on `b2:00.0` reported
an unspecified CUDA launch failure. Driver logs contained Xid 45 channel
cleanup events and Xid 154 reboot-required status on all eight GPUs, followed
by a global UVM error. No completed cached answer or latency was recorded;
the subsequent sequential objectives and generation benchmark were withheld.
The cause remains unresolved, and these events do not establish a physical
topology difference or isolate DCP as the cause.

The original driver was unloaded without force, `b2` received a PCI
function-level reset while unbound, and the original driver was reloaded.
All eight CUDA allocation/kernel/reduction checks passed. Driver parameters,
installed module hashes, and boot configuration remained unchanged, and no
reboot was needed.

The unchanged public BF16/MTP2 reference then passed 20/20 objective answers,
reasoning/tools, and 8/8 concurrent retrieval checks, but failed both long
checks. Its fresh request returned the correct JSON only in reasoning, with
empty final content. Cached generation crashed on `b2` with Xid 31, an MMU
virtual-read fault, after 13 recorded output tokens. This recurrence is not
specific to the private DCP implementation; the underlying cause is unresolved.

All eight GPUs subsequently passed repeated full-word local-memory checks
covering 67,623,976,960 bytes per card, leaving 128 MiB of free memory unused.
Each completed 546–583 pattern passes in at least 60 seconds, with zero
mismatches and successful deliberate-error detection. All eight directed PIX
links also passed full-word reads across 64,448 MiB of retained allocations
per owner, including deliberate-error detection. These expand the earlier
local and sampled-peer coverage. They do not establish correctness of every
multiprocess CUDA IPC mapping or exclude all intermittent faults.

With `NCCL_P2P_DISABLE=1` and `--disable-custom-all-reduce`, the same public
BF16/MTP2 model passed 20/20 objective answers, reasoning/tools, 8/8 concurrent
retrieval, fresh 262K retrieval, and four consecutive cached requests. All
five long-context responses returned the three records in final content.
Fresh first-token latency was 161.57 s; cached latencies were 1.45–1.54 s.
No new GPU fault occurred during this control, which was left serving with
268,608 cache tokens.

This control changes both communication paths and allocation layout. It does
not isolate the failing component or establish a permanent fix. The initial
recovery checks did not include a generation benchmark. To reproduce this
diagnostic on the BF16/MTP2 recipe below, set `NCCL_P2P_DISABLE=1` in the
environment and add `--disable-custom-all-reduce` to the serving command.

The [trial record](glm53-context-sharding-trial-20260918.json) preserves both
failed model configurations, memory checks, and the transport control. The
subsequent [NCCL transport comparison](#nccl-transport-comparison) adds generation
measurements and a failure with custom all-reduce still disabled.

### NCCL transport comparison

The next comparison kept custom all-reduce disabled and changed
`NCCL_P2P_DISABLE` from `1` to `0`. TP2/PP4, the 22/20/20/16 partition, BF16 KV,
two BF16 MTP drafts, 262,144-token context limit, 2,048-token prefill chunks,
and 268,608 cache tokens were unchanged. NCCL 2.29.7 initialization logs
confirmed `P2P/CUMEM` in both directions on all four symmetric PIX pairs;
pipeline links between pairs used shared host memory.

The peer-enabled run passed 20/20 objective answers, reasoning/tools, 8/8
concurrent retrieval checks, fresh 262K retrieval, and four cached requests.
Every long request returned the three records in final content. Fresh
first-token latency was 145.79 s, versus 161.57 s for the host-transport
control; cached latencies were 1.42–1.53 s.

It subsequently **failed during the first concurrency-16 generation run**.
The NaN guard reported invalid logits for three requests and terminated the
engine. The active pipeline microbatch contained four requests and twelve
verification tokens, with 132–161 output tokens already recorded. No new
driver Xid was logged, and all eight CUDA computation checks passed after
the processes exited. No reset, driver reload, or reboot was required.

Only the two single-request and two concurrency-eight timing cases completed
for the peer-enabled run. Its partial timings do not qualify this configuration
for serving. Five successful long-context requests did not predict successful
concurrent generation, and disabling custom all-reduce did not prevent this
failure. The source of the NaNs remains unresolved.

Repeating the host-transport control also passed the short checks, fresh
retrieval, and four cached requests. Fresh first-token latency was 160.05 s.
Its first concurrency-16 run then failed with an unspecified CUDA launch
error and Xid 31 on `b2`, an MMU virtual-read fault. Seven requests remained
active; the scheduled microbatch held two requests with 477 and 488 output
tokens recorded. All eight CUDA computation checks passed after application
exit, and the driver reported no reset requirement. This recurrence with
both peer transport and custom all-reduce disabled prevents treating either
setting as a sufficient fix.

| Graph execution control | Single tokens/s | C8 aggregate tokens/s | C16 aggregate tokens/s | 8K first-token latency |
| --- | ---: | ---: | ---: | ---: |
| Host transport, initial run | 39.49–51.94 | 212.72–253.47 | 282.27–295.52 | 6.09–6.27 s |
| NCCL peer transport | 44.08–47.01 | 205.99–240.91 | Failed: NaN logits | Not reached |
| Host transport, repeated run | 39.53–40.72 | 212.51–214.82 | Failed: MMU fault | Not reached |
| Preferred transport after reset/reload | 57.54–60.54 | 248.58–261.71 | 311.97 first run; second failed | Not reached |

An eager-execution control then disabled CUDA graph replay while retaining
host transport and exactly 268,608 cache tokens. It scored 19/20 objective
answers, with the previously observed `compute` → `computer` failure, and
passed reasoning/tools, 8/8 concurrent retrieval, fresh 262K retrieval,
and the first cached request. Fresh first-token latency was 162.58 s.
The next cached request failed on `b2` with two Xid 31 MMU faults, after
90 recorded output tokens. Its generation benchmark was withheld. This
failure occurred with graph replay, NCCL peer transport, and custom
all-reduce all disabled.

After that failure, the installed driver was unloaded without force and
`b2` received a PCI function-level reset while unbound. The original reset
method was restored, and the unchanged installed driver and persistence
service were reloaded. Driver parameters and all five installed-module
hashes matched their saved values. All eight CUDA computation checks passed;
no reboot or boot-configuration change was needed.

The preferred configuration was then tested with custom all-reduce and
NCCL peer transport enabled, giving 269,056 cache tokens. It passed 20/20
objective answers, reasoning/tools, 8/8 concurrent retrieval, fresh 262K
retrieval, and four cached requests. Fresh first-token latency was 147.28 s.
The first concurrency-16 benchmark completed at 311.97 accepted tokens/s,
but the second failed: `b2` reported a power-management firmware halt,
Xid 62, followed by Xid 154 requiring a function-level reset. Thus a clean
driver reload did not prevent recurrence. This run is **not qualified**.

A second unbound function-level reset and unchanged-driver reload recovered
all eight GPUs without rebooting. Independent checks then passed:

- Eight simultaneous BF16 matrix tests, each retaining 62 GiB of allocations,
  checked every result element against an analytical reference for sixty
  seconds. Each card completed 62,065–65,471 checked products with zero
  mismatches; deliberate single-element corruption was detected on every card.
- NCCL reductions on all four PIX pairs retained 61.75 GiB per card and
  exercised 3/6/12/24/48-token BF16 inputs of width 6,144. Every rank completed
  20,000 reductions in eager execution and 20,000 in graph replay, checking
  the final output of each sixteen-reduction group against exact values.
  All checks and deliberate-corruption detector checks passed.

These bounded tests do not exercise every model kernel, allocation, transfer,
or buffer lifetime. Their success neither establishes the cause of the model
failures nor rules out all hardware or software faults.

The [comparison record](glm53-transport-comparison-20260918.json) retains the
matched prompt hashes, accepted-token timing summaries, quality responses,
and failure details. Generation uses 512 input and 512 output tokens with
the same corpus and prompts in both configurations. The benchmark forces
output length; quality checks are separate.

### Pipeline-stage swap trial

The two PIX pairs attached to the second CPU were swapped between pipeline
stages using `--device-ids 0,1,2,3,6,7,4,5`. CUDA enumeration remained in its
original order. This preserved TP pair connectivity and CPU grouping while
moving physical `b2` from PP2/TP1 to PP3/TP1. Its former model stage moved
to the `da`/`db` pair. Worker PCI assignments were verified directly; model
weights, kernels, partition, transport, and 269,056-token cache capacity
were unchanged.

The run passed 20/20 objective answers, reasoning/tools, 8/8 concurrent
retrieval, and fresh retrieval of all three records from 260,224 input
tokens. Fresh first-token latency was 150.25 s. The first cached request
then stalled after 34 logged output tokens. Physical `b2` reported Xid 119
(GSP RPC timeout) and Xid 1 (GSP kernel panic). No completed cached answer
or generation benchmark was obtained.

The recurrence on the same physical card in a different model stage weakens
attribution to its former layer assignment. It does **not** distinguish a
driver/firmware defect from a physical card defect. The symmetric topology
is unchanged, and this configuration is not qualified.

Four middle-stage workers also logged allocator retries during fresh prefill;
that request subsequently passed. The [trial record](glm53-pair-swap-trial-20260918.json)
retains these observations, the successful checks, and the cached failure.

### Core-clock control trial

A temporary 1,350 MHz core-clock limit on physical `b2` tested whether clock
behavior contributed to its firmware failures. The same swapped stage
assignment, model settings, 269,056-token cache, and original installed driver
were retained. The limit was verified during model loading and full-context
prefill; all memory clocks remained at 1,728 MHz and power limits at 250 W.
All eight cards reported the same VBIOS version and maximum clock settings.

The run passed 20/20 objective answers, reasoning/tools, 8/8 concurrent
retrieval, fresh 262K retrieval, and four cached requests. Fresh first-token
latency was 149.30 s, and cached latencies were 1.44–1.57 s. The first
concurrency-sixteen benchmark then stalled with sixteen requests active.
Physical `b2` again reported Xid 119 and Xid 1, a GSP timeout and firmware
kernel panic. The core-clock limit therefore did not establish reliability.

Only the two single-request and two concurrency-eight timing cases completed:
48.69–50.01 and 227.22–263.40 accepted steady tokens/s, respectively. No
concurrency-sixteen or later prefill result completed. These are partial
diagnostic measurements; they do not establish a performance improvement
or a clock-only performance effect. The matching uncapped stage-swap trial
failed before benchmarking. The [clock-control record](glm53-clock-trial-20260918.json)
preserves all completed checks and the failure.

### Draft expert compression measurements

The checkpoint stores its MTP routed experts in BF16. An isolated comparison
used eight actual draft experts, replicated into 256 physical expert slots,
with TP2 rank-local dimensions, synthetic routes, and Gaussian activations.
Each path received three alternating timing samples for batches of one, four,
eight, and sixteen activation rows. The BF16 baseline used the existing
Triton expert kernel and its default configuration; no CMP-specific tuning
file was present.

| Draft expert weights | Storage per TP2 shard | Isolated expert speedup | Output relative L2 error versus BF16 |
| --- | ---: | ---: | ---: |
| INT8, group 128 | 4.57 GiB | 1.65–2.06× | 1.35–1.40% |
| NVFP4, group 16 | 2.53 GiB | 2.14–2.68× | 16.0–17.1% |

The BF16 expert shard occupies 9.0 GiB. Storage totals cover routed-expert
weights and their scales. Timings cover expert calculations with fixed
routing; they are not whole-draft or model-serving speedups. Synthetic
activation error does not measure model quality or draft acceptance.

A private draft-only INT8 conversion prototype also passed engine checks at
TP1 and TP2. It converted each projection with bounded per-expert temporaries,
loaded both TP2 shards, captured CUDA graphs, and generated the requested
output lengths for three sequential and four concurrent requests per layout.
Sequential input lengths were 32, 4,096, and 7,900 tokens; concurrent requests
each used 2,048 tokens. The small target and full-sized draft used random
weights. TP2 accepted zero of 98 proposed draft tokens, so these checks do not
establish accepted-draft behavior or real-model accuracy.

**Neither compression candidate is qualified for serving.** The later INT8
full-model comparison is recorded below; NVFP4 remains an isolated experiment.
The published recipe retains BF16 draft weights. The
[measurement record](glm53-draft-quantization-20260918.json) includes kernel
samples, numerical errors, storage, packing checks, and engine-check results.

### Full-model draft INT8 trial

The private prototype converted only the routed experts in MTP layer 78 to
group-128 INT8 weights, using BF16 activations and scales with existing Marlin
kernels. Both TP2 shards shrank from 9.0 to 4.57 GiB. The target model weights
were unchanged. KV capacity remained 269,056 tokens because an earlier
pipeline stage limited capacity.

The comparison retained TP2/PP4, the `22,20,20,16` partition, two draft tokens,
BF16 KV, and a 262,144-token context limit. Each timing column contains two
samples using identical prompt hashes, benchmark code, and workload settings.
Decode rates count accepted output tokens in the shared steady interval;
they exclude prefill and the trimmed request boundaries.

| Measurement | BF16 before | INT8 draft | BF16 after restoration |
| --- | ---: | ---: | ---: |
| Single generation, tokens/s | 47.60 / 47.70 | 63.35 / 48.40 | 49.31 / 52.65 |
| Concurrency 8, aggregate tokens/s | 258.15 / 253.88 | 289.87 / 246.26 | 261.46 / 262.16 |
| Concurrency 16, aggregate tokens/s | 340.72 / 324.57 | 328.53 / 316.06 | 348.14 / 320.49 |
| 8K prefill, seconds to first token | 5.49 / 5.34 | 5.55 / 5.33 | 5.51 / 5.31 |

These samples do not establish a consistent aggregate speedup. The INT8 run
passed 20/20 sequential and 20/20 concurrent objective answers, reasoning,
streamed tools and tool continuation, and all eight concurrent retrieval
checks. The BF16 reference's earlier intermittent lowercase-answer failure
remains documented; these are diagnostic measurements, not a claim of
universal output equivalence.

The INT8 run then failed on the first decode step after a fresh 260,224-token
prefill. Physical device `b2:00.0` reported three Xid 32 events, followed by a
CUDA launch failure. The cached-context check was withheld. This occurred on
rank 5, while draft conversion was confined to ranks 6 and 7; similar device
faults occurred in earlier trials without INT8. The cause remains unresolved.
All eight GPUs passed a CUDA computation check afterward, and the BF16
reference was restored without another reset or driver reload.

After the matched benchmark, the restored reference passed both fresh and
cached retrieval from the same 260,224-token document, returning all three
records correctly. First-token latency was 151.30 s fresh and 1.49 s cached.
This control did not reproduce the failure; it does not establish its cause.

**The INT8 draft candidate is not qualified for serving.** The
[comparison record](glm53-draft-int8-full-model-20260918.json) contains the
matched timings, prompt hashes, acceptance snapshots, quality answers,
failure details, and recovery checks.

### One-draft schedule comparison

A matched TP2/PP4 trial reduced the BF16 MTP draft count from two to one,
with corresponding decode graph sizes. Target weights, kernels, BF16 KV,
pipeline partition, memory utilization, and context limit were unchanged.
Both columns use two samples with identical prompt hashes and token budgets.

| Measurement | Two drafts, reference | One draft |
| --- | ---: | ---: |
| Single steady generation, tokens/s | 49.31 / 52.65 | 50.18 / 45.68 |
| Concurrency 8, steady aggregate tokens/s | 261.46 / 262.16 | 252.97 / 249.31 |
| Concurrency 16, steady aggregate tokens/s | 348.14 / 320.49 | 323.70 / 321.52 |
| 8K prefill, seconds to first token | 5.51 / 5.31 | 5.59 / 5.37 |

The one-draft trial passed 20/20 concurrent and 20/20 sequential objective
answers, reasoning/tools, and eight concurrent retrieval checks. Fresh
retrieval from 260,224 input tokens returned all three records correctly,
with 150.67 s to first token. The cached request ended with empty final
content: its correct JSON appeared only in the reasoning field. This fails
the final-answer retrieval check. No driver fault occurred during the trial.

**One draft is not qualified as a replacement.** It did not establish a
consistent speed gain and failed cached-context output validation. The
two-draft reference's earlier intermittent lowercase-answer failure remains
documented. See the [measurement record](glm53-draft-schedule-20260918.json).

An independent CUDA-core expert-projection prototype was also evaluated in
isolation, using raw row-major NVFP4 weights and BF16 activations. Across
eight synthetic projection cases and two block sizes it was 2.32–8.46× slower
than production Marlin. Numerical checks against FP32 matrix multiplication
and graph/eager agreement passed, but no model integration was attempted.
These results reject this prototype as a speed improvement; they do not
establish a limit for other kernel designs.

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
decode CUDA graphs. All GPUs report PCIe Gen2 x16. The topology has four
symmetric PIX pairs, with four cards per CPU/NUMA node. The software
environment matches the [build guide](README.md).

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
answers. Dense short-prefill is now disabled for this backend. That comparison
implicated dense prefill in those samples, but did not eliminate all observed
variation: the later [driver recovery checks](#driver-peer-access-trial) again
scored 19/20 in one concurrent run with sparse prefill, followed by two 20/20
runs. Eight concurrent retrieval requests, reasoning separation, streamed tools,
and tool-result continuation also passed in the
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
