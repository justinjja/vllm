# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Split FlashInfer FA2's cooperative attention/merge into stream-ordered kernels.

The planner, tensor-core attention and merge arithmetic are unchanged. Ordinary
launches let NCCL pipeline traffic overlap without requiring every attention CTA
to be resident simultaneously. Patch only a verified source revision and build
in the isolated FlashInfer JIT cache; never modify the installed dependency.
"""

import functools
import hashlib

import torch
from filelock import FileLock

from vllm.logger import init_logger

logger = init_logger(__name__)


@functools.cache
def get_noncooperative_mla_module():
    from flashinfer.jit import env, gen_batch_mla_module
    from flashinfer.jit.core import gen_jit_spec
    from flashinfer.jit.utils import write_if_different

    header = (env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/mla.cuh").read_text()
    expected = "3084a832425fa3872ce00b4b3ec52a32f9f319eb8c0403bc991e91c6d443d132"
    if hashlib.sha256(header.encode()).hexdigest() != expected:
        raise RuntimeError(
            "Sparse FA2 split launch requires the validated FlashInfer header"
        )
    for name, target in (
        ("../profiler.cuh", "profiler.cuh"),
        ("mla_params.cuh", "attention/mla_params.cuh"),
        ("prefill.cuh", "attention/prefill.cuh"),
        ("variant_helper.cuh", "attention/variant_helper.cuh"),
    ):
        header = header.replace(f'#include "{name}"', f"#include <flashinfer/{target}>")
    begin = header.index("  auto grid = cg::this_grid();")
    end = header.index("\n#define DISPATCH_SMEM_CONFIG", begin)
    header = (
        header[:begin]
        + """}

template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchMLAMergeKernel(
    const __grid_constant__ Params params) {
  DevicePersistentMergeStates<KTraits>(
      params.merge_packed_offset_start, params.merge_packed_offset_end,
      params.merge_partial_packed_offset_start, params.merge_partial_packed_offset_end,
      params.merge_partial_stride, params.partial_o, params.partial_lse,
      params.final_o, params.final_lse, params.o_stride_n, params.o_stride_h,
      params.num_heads, params.return_lse_base_on_e);
}
"""
        + header[end:]
    )
    launch = (
        "cudaLaunchCooperativeKernel((void*)kernel, nblks, nthrs, "
        "args, smem_size, stream)"
    )
    assert header.count(launch) == 1
    header = header.replace(
        launch,
        """cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream));
    auto merge_kernel = BatchMLAMergeKernel<KTraits, Params>;
    FLASHINFER_CUDA_CALL(
        cudaLaunchKernel((void*)merge_kernel, nblks, nthrs, args, 0, stream)""",
    )
    digest = hashlib.sha256(header.encode()).hexdigest()[:16]
    name = f"ds41_mla_noncooperative_{digest}"
    directory = env.FLASHINFER_GEN_SRC_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    # Serialize first-boot generation as well as compilation across PP ranks.
    with FileLock(str(directory / ".generate.lock")):
        original = gen_batch_mla_module(
            "fa2",
            torch.bfloat16,
            torch.bfloat16,
            torch.bfloat16,
            torch.int32,
            512,
            64,
            False,
        )
        write_if_different(directory / "ds41_mla.cuh", header)
        config = original.sources[0].parent / "batch_mla_config.inc"
        write_if_different(directory / config.name, config.read_text())
        sources = []
        for source_path in original.sources:
            source = source_path.read_text().replace(
                "#include <flashinfer/attention/mla.cuh>", '#include "ds41_mla.cuh"'
            )
            target = directory / source_path.name
            write_if_different(target, source)
            sources.append(target)
        module = gen_jit_spec(name, sources).build_and_load()
    logger.info_once("Ampere sparse FA2 uses noncooperative attention/merge launches")
    return module
