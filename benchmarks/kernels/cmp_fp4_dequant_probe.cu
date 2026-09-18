// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "libtorch_stable/quantization/marlin/dequant.h"

static void check(cudaError_t error) {
  if (error != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error));
    std::exit(1);
  }
}

__device__ __forceinline__ uint32_t direct_pair(uint32_t q) {
  uint32_t p = (q & 0x70007000u) >> 6;
  uint32_t nonzero = ((p | (p << 1) | (p << 2)) & 0x01000100u) >> 8;
  uint32_t half = p & ~(p >> 1) & ~(p >> 2) & 0x00400040u;
  return (p - half + nonzero * 0x3f00u) | (q & 0x80008000u);
}

template <bool direct>
__global__ void convert(uint32_t* output, int iterations) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t q = ((index & 15) << 8) | (((index >> 4) & 15) << 24) |
               (((index >> 8) & 15) << 12) | (((index >> 12) & 15) << 28);
  uint32_t accum0 = 0, accum1 = 0;
  for (int i = 0; i < iterations; ++i) {
    uint32_t x, y;
    if constexpr (direct) {
      x = direct_pair(q << 4);
      y = direct_pair(q);
    } else {
      nv_bfloat162 frag[2];
      marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), false>(q, frag);
      x = *reinterpret_cast<uint32_t*>(&frag[0]);
      y = *reinterpret_cast<uint32_t*>(&frag[1]);
    }
    accum0 ^= x;
    accum1 ^= y;
    q = q * 1664525u + 1013904223u;
  }
  output[2 * index] = accum0;
  output[2 * index + 1] = accum1;
}

static float timing(uint32_t* output, bool direct) {
  cudaEvent_t begin, end;
  check(cudaEventCreate(&begin));
  check(cudaEventCreate(&end));
  check(cudaEventRecord(begin));
  for (int repeat = 0; repeat < 10; ++repeat) {
    if (direct)
      convert<true><<<256, 256>>>(output, 512);
    else
      convert<false><<<256, 256>>>(output, 512);
  }
  check(cudaEventRecord(end));
  check(cudaEventSynchronize(end));
  float ms;
  check(cudaEventElapsedTime(&ms, begin, end));
  check(cudaEventDestroy(begin));
  check(cudaEventDestroy(end));
  return ms * 100;
}

int main() {
  constexpr size_t count = 2 * 65536;
  uint32_t *baseline, *direct;
  check(cudaMalloc(&baseline, count * sizeof(uint32_t)));
  check(cudaMalloc(&direct, count * sizeof(uint32_t)));
  std::vector<uint32_t> reference(count), actual(count);
  for (int iterations : {1, 512}) {
    convert<false><<<256, 256>>>(baseline, iterations);
    convert<true><<<256, 256>>>(direct, iterations);
    check(cudaGetLastError());
    check(cudaMemcpy(reference.data(), baseline, count * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    check(cudaMemcpy(actual.data(), direct, count * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost));
    if (reference != actual) {
      std::fprintf(stderr, "Conversion mismatch for %d iterations\n",
                   iterations);
      return 1;
    }
  }
  timing(baseline, false);
  timing(direct, true);
  std::vector<float> a, b;
  for (int repeat = 0; repeat < 8; ++repeat) {
    if (repeat % 2) {
      b.push_back(timing(direct, true));
      a.push_back(timing(baseline, false));
    } else {
      a.push_back(timing(baseline, false));
      b.push_back(timing(direct, true));
    }
  }
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  std::printf(
      "{\"packed_cases\":65536,\"exact\":true,\"baseline_us\":%.3f,"
      "\"direct_us\":%.3f,\"speedup\":%.4f,"
      "\"scope\":\"Compute-only conversion loop; not model latency\"}\n",
      (a[3] + a[4]) / 2, (b[3] + b[4]) / 2, (a[3] + a[4]) / (b[3] + b[4]));
  check(cudaFree(baseline));
  check(cudaFree(direct));
}
