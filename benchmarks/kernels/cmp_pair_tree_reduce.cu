// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_runtime.h>
#include <cuda_bf16.h>

__device__ unsigned acquire(unsigned* pointer) {
  unsigned value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(pointer)
               : "memory");
  return value;
}

__device__ void release(unsigned* pointer, unsigned value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(pointer), "r"(value)
               : "memory");
}

__device__ uint4 read_vector(const void* pointer) {
  uint4 value;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(pointer)
               : "memory");
  return value;
}

__device__ void write_vector(void* pointer, uint4 value) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(pointer),
               "r"(value.x), "r"(value.y), "r"(value.z), "r"(value.w)
               : "memory");
}

__device__ bool wait_for(unsigned* flag, unsigned sequence, int* error) {
  const auto started = clock64();
  while (acquire(flag) < sequence) {
    if (clock64() - started > 1000000000ULL) {
      *error = sequence;
      asm volatile("trap;");
      return false;
    }
  }
  return true;
}

// Each adjacent pair shares a leader-owned allocation. Only the two leaders
// map one another's allocations. The same algorithm runs on either CPU group.
__global__ void pair_tree_kernel(
    const __nv_bfloat16* input, __nv_bfloat16* output, char* local, char* other,
    unsigned* self_flags, unsigned* pair_flags, unsigned* remote_flags,
    __nv_bfloat16* self_result, __nv_bfloat16* pair_result, unsigned* counters,
    int* errors, int count, int capacity, int pair_rank) {
  __shared__ int ok;
  const unsigned sequence = counters[blockIdx.x] + 1;
  const int width = ((count + gridDim.x * 8 - 1) / (gridDim.x * 8)) * 8;
  const int begin = blockIdx.x * width;
  const int end = min(begin + width, count);
  auto* inputs = reinterpret_cast<__nv_bfloat16*>(local);
  auto* partial = reinterpret_cast<float*>(local + capacity * 4);
  auto* incoming = reinterpret_cast<float*>(local + capacity * 8);
  auto* flags = self_flags + blockIdx.x * 192;
  auto* mate_flags = pair_flags + blockIdx.x * 192;

  for (int i = begin + threadIdx.x * 8; i < end; i += blockDim.x * 8)
    write_vector(inputs + pair_rank * capacity + i, read_vector(input + i));
  __syncthreads();
  if (threadIdx.x == 0) {
    if (pair_rank == 1) release(mate_flags, sequence);
    ok = pair_rank == 0 ? wait_for(flags, sequence, errors + blockIdx.x)
                        : wait_for(flags + 32, sequence, errors + blockIdx.x);
  }
  __syncthreads();
  if (!ok) return;

  if (pair_rank == 0) {
    for (int i = begin + threadIdx.x * 8; i < end; i += blockDim.x * 8) {
      uint4 a = read_vector(inputs + i);
      uint4 b = read_vector(inputs + capacity + i);
      auto* av = reinterpret_cast<__nv_bfloat16*>(&a);
      auto* bv = reinterpret_cast<__nv_bfloat16*>(&b);
      float values[8];
#pragma unroll
      for (int j = 0; j < 8; ++j)
        values[j] = __bfloat162float(av[j]) + __bfloat162float(bv[j]);
      write_vector(partial + i, *reinterpret_cast<uint4*>(values));
      write_vector(partial + i + 4, *reinterpret_cast<uint4*>(values + 4));
      auto* peer_incoming = reinterpret_cast<float*>(other + capacity * 8);
      write_vector(peer_incoming + i, *reinterpret_cast<uint4*>(values));
      write_vector(peer_incoming + i + 4,
                   *reinterpret_cast<uint4*>(values + 4));
    }
    __syncthreads();
    auto* other_flags = remote_flags + blockIdx.x * 192;
    if (threadIdx.x == 0) {
      release(other_flags + 64, sequence);
      ok = wait_for(flags + 64, sequence, errors + blockIdx.x);
    }
    __syncthreads();
    if (!ok) return;
    for (int i = begin + threadIdx.x * 8; i < end; i += blockDim.x * 8) {
      uint4 a[2] = {read_vector(partial + i), read_vector(partial + i + 4)};
      uint4 b[2] = {read_vector(incoming + i), read_vector(incoming + i + 4)};
      auto* av = reinterpret_cast<float*>(a);
      auto* bv = reinterpret_cast<float*>(b);
      uint4 pack;
      auto* values = reinterpret_cast<__nv_bfloat16*>(&pack);
#pragma unroll
      for (int j = 0; j < 8; ++j)
        values[j] = __float2bfloat16_rn(av[j] + bv[j]);
      write_vector(pair_result + i, pack);
      write_vector(output + i, pack);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      release(other_flags + 96, sequence);
      release(mate_flags + 32, sequence);
      ok = wait_for(flags + 96, sequence, errors + blockIdx.x) &&
           wait_for(flags + 128, sequence, errors + blockIdx.x);
    }
    __syncthreads();
    if (!ok) return;
  } else {
    for (int i = begin + threadIdx.x * 8; i < end; i += blockDim.x * 8)
      write_vector(output + i, read_vector(self_result + i));
    __syncthreads();
    if (threadIdx.x == 0) release(mate_flags + 128, sequence);
  }
  if (threadIdx.x == 0) counters[blockIdx.x] = sequence;
}

extern "C" int pair_tree_zero(void* pointer, size_t bytes) {
  return cudaMemset(pointer, 0, bytes);
}

extern "C" int pair_tree_launch(void* input, void* output, void* local,
                                void* other, void* self_flags, void* pair_flags,
                                void* other_flags, void* self_result,
                                void* pair_result, void* counters, void* errors,
                                int count, int capacity, int pair_rank,
                                int blocks, void* stream) {
  if (count <= 0 || count > capacity || count % 8 || capacity % 128 ||
      blocks < 1 || blocks > 32 || pair_rank < 0 || pair_rank > 1)
    return cudaErrorInvalidValue;
  pair_tree_kernel<<<blocks, 128, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<__nv_bfloat16*>(input), static_cast<__nv_bfloat16*>(output),
      static_cast<char*>(local), static_cast<char*>(other),
      static_cast<unsigned*>(self_flags), static_cast<unsigned*>(pair_flags),
      static_cast<unsigned*>(other_flags),
      static_cast<__nv_bfloat16*>(self_result),
      static_cast<__nv_bfloat16*>(pair_result),
      static_cast<unsigned*>(counters), static_cast<int*>(errors), count,
      capacity, pair_rank);
  return cudaGetLastError();
}
