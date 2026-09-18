// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Standalone experiment: peer writes with a GPU 6 relay for GPU 7.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define RT(call)                                                 \
  do {                                                           \
    auto e = (call);                                             \
    if (e != cudaSuccess) {                                      \
      fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
      exit(2);                                                   \
    }                                                            \
  } while (0)
#define DR(call)                                   \
  do {                                             \
    auto e = (call);                               \
    if (e != CUDA_SUCCESS) {                       \
      const char* message;                         \
      cuGetErrorString(e, &message);               \
      fprintf(stderr, "%s: %s\n", #call, message); \
      exit(3);                                     \
    }                                              \
  } while (0)

struct Shared {
  volatile float* input[8];
  volatile float* partial[7];
  volatile unsigned* flags[8];
  volatile float* relay_output;
};

template <bool ParallelFlags>
__device__ bool phase(Shared shared, int rank, unsigned sequence,
                      int wait_ranks, bool publish, int* error) {
  __shared__ int ok;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    ok = 1;
    if (publish) shared.flags[rank][blockIdx.x * 32] = sequence;
  }
  __syncthreads();
  const auto started = clock64();
  if constexpr (ParallelFlags) {
    const int peer = wait_ranks == -1 ? 7 : (wait_ranks == 1 ? 6 : threadIdx.x);
    if (threadIdx.x < (wait_ranks == -1 ? 1 : wait_ranks)) {
      while (shared.flags[peer][blockIdx.x * 32] < sequence) {
        if (clock64() - started > 500000000ULL) {
          atomicExch(&ok, 0);
          break;
        }
      }
    }
  } else if (threadIdx.x == 0) {
    const int first = wait_ranks == -1 ? 7 : (wait_ranks == 1 ? 6 : 0);
    const int last = wait_ranks == -1 ? 8 : (wait_ranks == 1 ? 7 : wait_ranks);
    for (int peer = first; peer < last; ++peer) {
      while (shared.flags[peer][blockIdx.x * 32] < sequence) {
        if (clock64() - started > 500000000ULL) {
          ok = 0;
          break;
        }
      }
      if (!ok) break;
    }
  }
  __syncthreads();
  if (!ok && threadIdx.x == 0) *error = sequence;
  __threadfence_system();
  return ok != 0;
}

template <bool ParallelFlags>
__global__ void relay_reduce(const float* input, float* output, Shared shared,
                             unsigned* counters, int* error, int count,
                             int rank) {
  const int width = (count + gridDim.x - 1) / gridDim.x;
  const int begin = blockIdx.x * width;
  const int end = min(begin + width, count);
  const unsigned call = counters[blockIdx.x];
  const unsigned sequence = call * 3 + 1;
  const float variation = float(call % 17) * 0.03125f;
  if (rank == 7) {
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
      shared.input[7][i] = input[i] + variation;
    if (!phase<ParallelFlags>(shared, rank, sequence, 0, true, error)) return;
    if (!phase<ParallelFlags>(shared, rank, sequence + 2, 1, false, error))
      return;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x)
      output[i] = shared.relay_output[i];
    if (!phase<ParallelFlags>(shared, rank, sequence + 2, 0, true, error))
      return;
  } else {
    const int shard = (width + 6) / 7;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
      const int owner = min((i - begin) / shard, 6);
      shared.input[owner][rank * count + i] = input[i] + variation;
    }
    if (rank == 6) {
      if (!phase<ParallelFlags>(shared, rank, sequence, -1, false, error))
        return;
      for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const int owner = min((i - begin) / shard, 6);
        shared.input[owner][7 * count + i] = shared.input[7][i];
      }
    }
    if (!phase<ParallelFlags>(shared, rank, sequence, 7, true, error)) return;
    const int own_end = min(begin + (rank + 1) * shard, end);
    for (int i = begin + rank * shard + threadIdx.x; i < own_end;
         i += blockDim.x) {
      float sum = 0.f;
      for (int peer = 0; peer < 8; ++peer)
        sum += shared.input[rank][peer * count + i];
      for (int peer = 0; peer < 7; ++peer) shared.partial[peer][i] = sum;
    }
    if (!phase<ParallelFlags>(shared, rank, sequence + 1, 7, true, error))
      return;
    for (int i = begin + threadIdx.x; i < end; i += blockDim.x) {
      const float value = shared.partial[rank][i];
      output[i] = value;
      if (rank == 6) shared.relay_output[i] = value;
    }
    if (!phase<ParallelFlags>(shared, rank, sequence + 2, 8, true, error))
      return;
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = call + 1;
}

struct Allocation {
  int owner;
  size_t bytes;
  CUdeviceptr pointer;
  CUmemGenericAllocationHandle handle;
};

Allocation allocate(int owner, size_t requested) {
  RT(cudaSetDevice(owner));
  CUmemAllocationProp property = {};
  property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  property.location.id = owner;
  size_t granularity;
  DR(cuMemGetAllocationGranularity(&granularity, &property,
                                   CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  Allocation a{};
  a.owner = owner;
  a.bytes = ((requested + granularity - 1) / granularity) * granularity;
  DR(cuMemCreate(&a.handle, a.bytes, &property, 0));
  DR(cuMemAddressReserve(&a.pointer, a.bytes, 0, 0, 0));
  DR(cuMemMap(a.pointer, a.bytes, 0, a.handle, 0));
  for (int reader = owner == 7 ? 6 : 0; reader < (owner >= 6 ? 8 : 7);
       ++reader) {
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = reader;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DR(cuMemSetAccess(a.pointer, a.bytes, &access, 1));
  }
  RT(cudaMemset(reinterpret_cast<void*>(a.pointer), 0, a.bytes));
  RT(cudaDeviceSynchronize());
  return a;
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
  const int count = argc > 1 ? atoi(argv[1]) : 30720;
  const int blocks = argc > 2 ? atoi(argv[2]) : 8;
  const bool parallel_flags = argc > 3 ? atoi(argv[3]) != 0 : true;
  const bool host_flags = argc > 4 ? atoi(argv[4]) != 0 : false;
  constexpr int ranks = 8, iterations = 32;
  int devices;
  RT(cudaGetDeviceCount(&devices));
  if (devices != ranks || count < 1 || blocks < 1 || blocks > 32) return 4;
  // Initialize every primary context before establishing peer mappings.
  for (int rank = 0; rank < ranks; ++rank) {
    RT(cudaSetDevice(rank));
    RT(cudaFree(nullptr));
  }
  Shared shared{};
  unsigned* host_flags_pointer = nullptr;
  if (host_flags) {
    RT(cudaHostAlloc(&host_flags_pointer,
                     ranks * blocks * 32 * sizeof(unsigned),
                     cudaHostAllocPortable | cudaHostAllocMapped));
    memset(host_flags_pointer, 0, ranks * blocks * 32 * sizeof(unsigned));
  }
  std::vector<Allocation> allocations;
  for (int rank = 0; rank < ranks; ++rank) {
    const int owner = rank == 7 ? 6 : rank;
    allocations.push_back(
        allocate(owner, (rank == 7 ? 1 : 8) * count * sizeof(float)));
    shared.input[rank] = reinterpret_cast<float*>(allocations.back().pointer);
    allocations.push_back(allocate(owner, blocks * 32 * sizeof(unsigned)));
    shared.flags[rank] =
        reinterpret_cast<unsigned*>(allocations.back().pointer);
    if (rank < 7) {
      allocations.push_back(allocate(owner, count * sizeof(float)));
      shared.partial[rank] =
          reinterpret_cast<float*>(allocations.back().pointer);
    }
  }
  allocations.push_back(allocate(7, count * sizeof(float)));
  shared.relay_output = reinterpret_cast<float*>(allocations.back().pointer);
  std::vector<Device> d(ranks);
  std::vector<std::vector<float>> inputs(ranks, std::vector<float>(count));
  for (int rank = 0; rank < ranks; ++rank) {
    RT(cudaSetDevice(rank));
    auto& x = d[rank];
    if (host_flags) {
      unsigned* mapped;
      RT(cudaHostGetDevicePointer(&mapped, host_flags_pointer, 0));
      for (int peer = 0; peer < ranks; ++peer)
        shared.flags[peer] = mapped + peer * blocks * 32;
    }
    RT(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
    RT(cudaEventCreate(&x.start));
    RT(cudaEventCreate(&x.end));
    RT(cudaMalloc(&x.input, count * sizeof(float)));
    RT(cudaMalloc(&x.output, count * sizeof(float)));
    RT(cudaMalloc(&x.counters, blocks * sizeof(unsigned)));
    RT(cudaMalloc(&x.error, sizeof(int)));
    RT(cudaMemset(x.counters, 0, blocks * sizeof(unsigned)));
    RT(cudaMemset(x.error, 0, sizeof(int)));
    for (int i = 0; i < count; ++i)
      inputs[rank][i] = float(((i * 17 + rank * 1117) % 4093) - 2046) *
                        (rank % 2 ? 0.00013f : -0.0317f);
    RT(cudaMemcpy(x.input, inputs[rank].data(), count * sizeof(float),
                  cudaMemcpyHostToDevice));
    RT(cudaStreamBeginCapture(x.stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iterations; ++i) {
      if (parallel_flags)
        relay_reduce<true><<<blocks, 256, 0, x.stream>>>(
            x.input, x.output, shared, x.counters, x.error, count, rank);
      else
        relay_reduce<false><<<blocks, 256, 0, x.stream>>>(
            x.input, x.output, shared, x.counters, x.error, count, rank);
    }
    RT(cudaStreamEndCapture(x.stream, &x.graph));
    RT(cudaGraphInstantiate(&x.executable, x.graph, nullptr, nullptr, 0));
  }
  std::vector<float> timings;
  int checked = 0;
  for (int repeat = 0; repeat < 7; ++repeat) {
    for (int rank = 0; rank < ranks; ++rank) {
      RT(cudaSetDevice(rank));
      auto& x = d[rank];
      RT(cudaEventRecord(x.start, x.stream));
      RT(cudaGraphLaunch(x.executable, x.stream));
      RT(cudaEventRecord(x.end, x.stream));
    }
    float slowest = 0;
    const unsigned calls = (repeat + 1) * iterations;
    for (int rank = 0; rank < ranks; ++rank) {
      RT(cudaSetDevice(rank));
      auto& x = d[rank];
      RT(cudaEventSynchronize(x.end));
      float elapsed;
      RT(cudaEventElapsedTime(&elapsed, x.start, x.end));
      slowest = std::max(slowest, elapsed * 1000 / iterations);
      int error;
      RT(cudaMemcpy(&error, x.error, sizeof(int), cudaMemcpyDeviceToHost));
      if (error) {
        fprintf(stderr, "Barrier timed out on rank %d sequence %d\n", rank,
                error);
        return 5;
      }
      std::vector<unsigned> counters(blocks);
      RT(cudaMemcpy(counters.data(), x.counters, blocks * sizeof(unsigned),
                    cudaMemcpyDeviceToHost));
      for (auto n : counters)
        if (n != calls) return 6;
      std::vector<float> output(count);
      RT(cudaMemcpy(output.data(), x.output, count * sizeof(float),
                    cudaMemcpyDeviceToHost));
      for (int i = 0; i < count; ++i) {
        float expected = 0;
        for (int peer = 0; peer < ranks; ++peer)
          expected += inputs[peer][i] + float((calls - 1) % 17) * 0.03125f;
        if (output[i] != expected) {
          fprintf(stderr, "Mismatch rank %d element %d: %.9g != %.9g\n", rank,
                  i, output[i], expected);
          return 7;
        }
        ++checked;
      }
    }
    if (repeat >= 2) timings.push_back(slowest);
  }
  std::sort(timings.begin(), timings.end());
  printf(
      "{\"ranks\":8,\"elements\":%d,\"blocks\":%d,\"calls\":%d,"
      "\"checked_outputs\":%d,\"median_us\":%.3f,\"min_us\":%.3f,"
      "\"max_us\":%.3f,\"parallel_flags\":%s,\"host_flags\":%s}\n",
      count, blocks, 7 * iterations, checked, timings[2], timings.front(),
      timings.back(), parallel_flags ? "true" : "false",
      host_flags ? "true" : "false");
  for (int rank = 0; rank < ranks; ++rank) {
    RT(cudaSetDevice(rank));
    auto& x = d[rank];
    RT(cudaGraphExecDestroy(x.executable));
    RT(cudaGraphDestroy(x.graph));
    RT(cudaFree(x.input));
    RT(cudaFree(x.output));
    RT(cudaFree(x.counters));
    RT(cudaFree(x.error));
    RT(cudaEventDestroy(x.start));
    RT(cudaEventDestroy(x.end));
    RT(cudaStreamDestroy(x.stream));
  }
  for (auto a : allocations) {
    RT(cudaSetDevice(a.owner));
    DR(cuMemUnmap(a.pointer, a.bytes));
    DR(cuMemAddressFree(a.pointer, a.bytes));
    DR(cuMemRelease(a.handle));
  }
  if (host_flags_pointer) RT(cudaFreeHost(host_flags_pointer));
}
