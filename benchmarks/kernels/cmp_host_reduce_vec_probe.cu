// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Standalone experiment: vector FP32 transfers through mapped RAM.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                      \
  do {                                                   \
    cudaError_t e = (call);                              \
    if (e != cudaSuccess) {                              \
      fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(e));                    \
      exit(2);                                           \
    }                                                    \
  } while (0)

__device__ float4 load4(const volatile float* pointer) {
  float4 value;
  asm volatile("ld.volatile.global.v4.f32 {%0,%1,%2,%3}, [%4];"
               : "=f"(value.x), "=f"(value.y), "=f"(value.z), "=f"(value.w)
               : "l"(pointer)
               : "memory");
  return value;
}

__device__ void store4(volatile float* pointer, float4 value) {
  asm volatile("st.volatile.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(pointer),
               "f"(value.x), "f"(value.y), "f"(value.z), "f"(value.w)
               : "memory");
}

// One writer per naturally aligned flag; no remote read-modify-write atomics.
__device__ bool barrier(volatile unsigned* flags, int ranks, int rank,
                        int blocks, unsigned sequence, int* error) {
  __shared__ int ok;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ok = 1;
    flags[(rank * blocks + blockIdx.x) * 32] = sequence;
    const auto started = clock64();
    for (int peer = 0; peer < ranks; ++peer) {
      while (flags[(peer * blocks + blockIdx.x) * 32] < sequence) {
        if (clock64() - started > 500000000ULL) {
          ok = 0;
          *error = 1;
          break;
        }
      }
      if (!ok) break;
    }
  }
  __syncthreads();
  __threadfence_system();
  return ok != 0;
}

__global__ void host_reduce(const float* input, float* output,
                            volatile float* exchange, volatile float* reduced,
                            volatile unsigned* flags, unsigned* counters,
                            int* error, int count, int ranks, int rank) {
  const int blocks = gridDim.x;
  const int width = (count + blocks - 1) / blocks;
  const int begin = blockIdx.x * width;
  const int end = min(begin + width, count);
  const unsigned call = counters[blockIdx.x];
  const unsigned sequence = call * 3 + 1;
  const float variation = float(call % 17) * 0.03125f;
  for (int i = begin + threadIdx.x * 4; i < end; i += blockDim.x * 4) {
    float4 value = reinterpret_cast<const float4*>(input)[i / 4];
    value.x += variation;
    value.y += variation;
    value.z += variation;
    value.w += variation;
    store4(exchange + rank * count + i, value);
  }
  if (!barrier(flags, ranks, rank, blocks, sequence, error)) return;
  const int shard = ((width + ranks * 4 - 1) / (ranks * 4)) * 4;
  const int own_end = min(begin + (rank + 1) * shard, end);
  for (int i = begin + rank * shard + threadIdx.x * 4; i < own_end;
       i += blockDim.x * 4) {
    float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int peer = 0; peer < ranks; ++peer) {
      float4 value = load4(exchange + peer * count + i);
      sum.x += value.x;
      sum.y += value.y;
      sum.z += value.z;
      sum.w += value.w;
    }
    store4(reduced + i, sum);
  }
  if (!barrier(flags, ranks, rank, blocks, sequence + 1, error)) return;
  for (int i = begin + threadIdx.x * 4; i < end; i += blockDim.x * 4)
    reinterpret_cast<float4*>(output)[i / 4] = load4(reduced + i);
  if (!barrier(flags, ranks, rank, blocks, sequence + 2, error)) return;
  if (threadIdx.x == 0) counters[blockIdx.x] = call + 1;
}

struct Device {
  float *input, *output;
  unsigned* counters;
  int* error;
  cudaStream_t stream;
  cudaEvent_t start, end;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
};

int main(int argc, char** argv) {
  const int ranks = argc > 1 ? atoi(argv[1]) : 8;
  const int count = argc > 2 ? atoi(argv[2]) : 30720;
  const int blocks = argc > 3 ? atoi(argv[3]) : 8;
  constexpr int iterations = 32;
  int devices = 0;
  CHECK(cudaGetDeviceCount(&devices));
  if (ranks < 2 || ranks > devices || count < 1 || blocks < 1 || blocks > 32 ||
      count % (blocks * 4) != 0)
    return 3;
  float *exchange, *reduced;
  unsigned* flags;
  const unsigned alloc_flags = cudaHostAllocPortable | cudaHostAllocMapped;
  CHECK(cudaHostAlloc(&exchange, ranks * count * sizeof(float), alloc_flags));
  CHECK(cudaHostAlloc(&reduced, count * sizeof(float), alloc_flags));
  CHECK(cudaHostAlloc(&flags, ranks * blocks * 32 * sizeof(unsigned),
                      alloc_flags));
  memset(flags, 0, ranks * blocks * 32 * sizeof(unsigned));
  std::vector<Device> d(ranks);
  std::vector<std::vector<float>> inputs(ranks, std::vector<float>(count));
  for (int rank = 0; rank < ranks; ++rank) {
    CHECK(cudaSetDevice(rank));
    auto& x = d[rank];
    CHECK(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
    CHECK(cudaEventCreate(&x.start));
    CHECK(cudaEventCreate(&x.end));
    CHECK(cudaMalloc(&x.input, count * sizeof(float)));
    CHECK(cudaMalloc(&x.output, count * sizeof(float)));
    CHECK(cudaMalloc(&x.counters, blocks * sizeof(unsigned)));
    CHECK(cudaMalloc(&x.error, sizeof(int)));
    CHECK(cudaMemset(x.counters, 0, blocks * sizeof(unsigned)));
    CHECK(cudaMemset(x.error, 0, sizeof(int)));
    for (int i = 0; i < count; ++i)
      inputs[rank][i] = float(((i * 17 + rank * 1117) % 4093) - 2046) *
                        (rank % 2 ? 0.00013f : -0.0317f);
    CHECK(cudaMemcpy(x.input, inputs[rank].data(), count * sizeof(float),
                     cudaMemcpyHostToDevice));
    float *device_exchange, *device_reduced;
    unsigned* device_flags;
    CHECK(cudaHostGetDevicePointer(&device_exchange, exchange, 0));
    CHECK(cudaHostGetDevicePointer(&device_reduced, reduced, 0));
    CHECK(cudaHostGetDevicePointer(&device_flags, flags, 0));
    CHECK(cudaStreamBeginCapture(x.stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iterations; ++i)
      host_reduce<<<blocks, 256, 0, x.stream>>>(
          x.input, x.output, device_exchange, device_reduced, device_flags,
          x.counters, x.error, count, ranks, rank);
    CHECK(cudaStreamEndCapture(x.stream, &x.graph));
    CHECK(cudaGraphInstantiate(&x.executable, x.graph, nullptr, nullptr, 0));
  }
  std::vector<float> timings;
  int checked = 0;
  for (int repeat = 0; repeat < 7; ++repeat) {
    for (int rank = 0; rank < ranks; ++rank) {
      CHECK(cudaSetDevice(rank));
      auto& x = d[rank];
      CHECK(cudaEventRecord(x.start, x.stream));
      CHECK(cudaGraphLaunch(x.executable, x.stream));
      CHECK(cudaEventRecord(x.end, x.stream));
    }
    float slowest = 0;
    const unsigned calls = (repeat + 1) * iterations;
    for (int rank = 0; rank < ranks; ++rank) {
      CHECK(cudaSetDevice(rank));
      auto& x = d[rank];
      CHECK(cudaEventSynchronize(x.end));
      float elapsed;
      CHECK(cudaEventElapsedTime(&elapsed, x.start, x.end));
      slowest = std::max(slowest, elapsed * 1000 / iterations);
      int error;
      CHECK(cudaMemcpy(&error, x.error, sizeof(int), cudaMemcpyDeviceToHost));
      if (error) {
        fprintf(stderr, "Barrier timed out on rank %d\n", rank);
        return 4;
      }
      std::vector<unsigned> counters(blocks);
      CHECK(cudaMemcpy(counters.data(), x.counters, blocks * sizeof(unsigned),
                       cudaMemcpyDeviceToHost));
      for (auto n : counters)
        if (n != calls) return 5;
      std::vector<float> output(count);
      CHECK(cudaMemcpy(output.data(), x.output, count * sizeof(float),
                       cudaMemcpyDeviceToHost));
      for (int i = 0; i < count; ++i) {
        float expected = 0;
        for (int peer = 0; peer < ranks; ++peer)
          expected += inputs[peer][i] + float((calls - 1) % 17) * 0.03125f;
        if (output[i] != expected) {
          fprintf(stderr, "Mismatch rank %d element %d: %.9g != %.9g\n", rank,
                  i, output[i], expected);
          return 6;
        }
        ++checked;
      }
    }
    if (repeat >= 2) timings.push_back(slowest);
  }
  std::sort(timings.begin(), timings.end());
  printf(
      "{\"ranks\":%d,\"elements\":%d,\"blocks\":%d,\"calls\":%d,"
      "\"checked_outputs\":%d,\"median_us\":%.3f,\"min_us\":%.3f,"
      "\"max_us\":%.3f}\n",
      ranks, count, blocks, 7 * iterations, checked, timings[2],
      timings.front(), timings.back());
  for (int rank = 0; rank < ranks; ++rank) {
    CHECK(cudaSetDevice(rank));
    auto& x = d[rank];
    CHECK(cudaGraphExecDestroy(x.executable));
    CHECK(cudaGraphDestroy(x.graph));
    CHECK(cudaFree(x.input));
    CHECK(cudaFree(x.output));
    CHECK(cudaFree(x.counters));
    CHECK(cudaFree(x.error));
    CHECK(cudaEventDestroy(x.start));
    CHECK(cudaEventDestroy(x.end));
    CHECK(cudaStreamDestroy(x.stream));
  }
  CHECK(cudaFreeHost(exchange));
  CHECK(cudaFreeHost(reduced));
  CHECK(cudaFreeHost(flags));
}
