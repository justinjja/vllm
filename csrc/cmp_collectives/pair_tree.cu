// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace cmp_pair_tree {
using Flag = unsigned long long;

__device__ Flag acquire(Flag* pointer) {
  Flag value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];"
               : "=l"(value)
               : "l"(pointer)
               : "memory");
  return value;
}

__device__ void release(Flag* pointer, Flag value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::"l"(pointer), "l"(value)
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

__device__ bool wait_for(Flag* flag, Flag sequence, Flag* error) {
  const auto started = clock64();
  while (acquire(flag) != sequence) {
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
    Flag* self_flags, Flag* pair_flags, Flag* remote_flags,
    __nv_bfloat16* self_result, __nv_bfloat16* pair_result, Flag* counters,
    Flag* errors, int count, int capacity, int pair_rank) {
  __shared__ int ok;
  const Flag sequence = counters[blockIdx.x] + 1;
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

}  // namespace cmp_pair_tree

namespace {
constexpr int kBlocks = 2;
constexpr int kMaxElements = 6 * 6144;
constexpr size_t kFlagBytes = kBlocks * 192 * sizeof(cmp_pair_tree::Flag);
constexpr size_t kCounterBytes = kBlocks * sizeof(cmp_pair_tree::Flag);
size_t workspace_bytes(int capacity) {
  return size_t(capacity) * 14 + kFlagBytes + 2 * kCounterBytes;
}
}  // namespace

extern "C" int cmp_pair_tree_abi_version() { return 1; }
extern "C" int cmp_pair_tree_handle_size() {
  return sizeof(cudaIpcMemHandle_t);
}
extern "C" const char* cmp_pair_tree_error(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}

extern "C" int cmp_pair_tree_allocate(void** output, void* handle,
                                      int capacity) {
  if (capacity < 128 || capacity > kMaxElements || capacity % 128)
    return cudaErrorInvalidValue;
  *output = nullptr;
  auto status = cudaMalloc(output, workspace_bytes(capacity));
  if (status == cudaSuccess)
    status = cudaMemset(*output, 0, workspace_bytes(capacity));
  if (status == cudaSuccess)
    status =
        cudaIpcGetMemHandle(static_cast<cudaIpcMemHandle_t*>(handle), *output);
  if (status == cudaSuccess) status = cudaDeviceSynchronize();
  if (status != cudaSuccess) {
    if (*output) {
      cudaFree(*output);
      *output = nullptr;
    }
    cudaGetLastError();
  }
  return status;
}

extern "C" int cmp_pair_tree_open(void** output, const void* handle) {
  *output = nullptr;
  auto status = cudaIpcOpenMemHandle(
      output, *static_cast<const cudaIpcMemHandle_t*>(handle),
      cudaIpcMemLazyEnablePeerAccess);
  if (status != cudaSuccess) cudaGetLastError();
  return status;
}

extern "C" int cmp_pair_tree_close(void* pointer) {
  return cudaIpcCloseMemHandle(pointer);
}

extern "C" int cmp_pair_tree_free(void* pointer) { return cudaFree(pointer); }

extern "C" int cmp_pair_tree_reduce(void* input, void* output, void* self,
                                    void* mate, void* other, int count,
                                    int capacity, int rank, void* stream) {
  if (count <= 0 || count > capacity || count % 8 || capacity % 128 ||
      capacity > kMaxElements || rank < 0 || rank > 3 || !input || !output ||
      !self || !mate || (rank % 2 == 0 && !other))
    return cudaErrorInvalidValue;
  using cmp_pair_tree::Flag;
  auto* own = static_cast<char*>(self);
  auto* pair = static_cast<char*>(mate);
  auto* remote = static_cast<char*>(other);
  auto* flags = reinterpret_cast<Flag*>(own + capacity * 14);
  auto* pair_flags = reinterpret_cast<Flag*>(pair + capacity * 14);
  auto* remote_flags =
      remote ? reinterpret_cast<Flag*>(remote + capacity * 14) : nullptr;
  auto* counters = reinterpret_cast<Flag*>(own + capacity * 14 + kFlagBytes);
  auto* errors =
      reinterpret_cast<Flag*>(own + capacity * 14 + kFlagBytes + kCounterBytes);
  cmp_pair_tree::
      pair_tree_kernel<<<kBlocks, 128, 0, static_cast<cudaStream_t>(stream)>>>(
          static_cast<__nv_bfloat16*>(input),
          static_cast<__nv_bfloat16*>(output), rank % 2 == 0 ? own : pair,
          remote, flags, pair_flags, remote_flags,
          reinterpret_cast<__nv_bfloat16*>(own + capacity * 12),
          reinterpret_cast<__nv_bfloat16*>(pair + capacity * 12), counters,
          errors, count, capacity, rank % 2);
  return cudaGetLastError();
}
