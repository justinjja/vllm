// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Test allocation-scoped peer mappings without enabling whole-device peer
// access.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define RT(call)                                                 \
  do {                                                           \
    auto e = (call);                                             \
    if (e != cudaSuccess) {                                      \
      fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); \
      return 2;                                                  \
    }                                                            \
  } while (0)
#define DR(call)                                   \
  do {                                             \
    auto e = (call);                               \
    if (e != CUDA_SUCCESS) {                       \
      const char* message;                         \
      cuGetErrorString(e, &message);               \
      fprintf(stderr, "%s: %s\n", #call, message); \
      return 3;                                    \
    }                                              \
  } while (0)

__global__ void read_peer(const float* source, float* output) {
  output[0] = source[0];
}

int main() {
  int devices;
  RT(cudaGetDeviceCount(&devices));
  if (devices != 8) return 4;
  std::vector<float*> outputs(devices);
  for (int device = 0; device < devices; ++device) {
    RT(cudaSetDevice(device));
    RT(cudaMalloc(&outputs[device], sizeof(float)));
  }
  int success = 0;
  for (int owner = 0; owner < devices; ++owner) {
    RT(cudaSetDevice(owner));
    CUmemAllocationProp property = {};
    property.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    property.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    property.location.id = owner;
    size_t bytes;
    DR(cuMemGetAllocationGranularity(&bytes, &property,
                                     CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    CUmemGenericAllocationHandle handle;
    CUdeviceptr pointer;
    DR(cuMemCreate(&handle, bytes, &property, 0));
    DR(cuMemAddressReserve(&pointer, bytes, 0, 0, 0));
    DR(cuMemMap(pointer, bytes, 0, handle, 0));
    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = owner;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DR(cuMemSetAccess(pointer, bytes, &access, 1));
    float expected = owner + 0.25f;
    RT(cudaMemcpy(reinterpret_cast<void*>(pointer), &expected, sizeof(float),
                  cudaMemcpyHostToDevice));
    for (int reader = 0; reader < devices; ++reader) {
      access.location.id = reader;
      CUresult mapping = cuMemSetAccess(pointer, bytes, &access, 1);
      const char* message;
      cuGetErrorString(mapping, &message);
      printf(
          "{\"owner\":%d,\"reader\":%d,\"mapping_status\":%d,\"mapping_"
          "message\":\"%s\"",
          owner, reader, int(mapping), message);
      if (mapping == CUDA_SUCCESS) {
        RT(cudaSetDevice(reader));
        read_peer<<<1, 1>>>(reinterpret_cast<const float*>(pointer),
                            outputs[reader]);
        RT(cudaDeviceSynchronize());
        float actual;
        RT(cudaMemcpy(&actual, outputs[reader], sizeof(float),
                      cudaMemcpyDeviceToHost));
        printf(",\"correct\":%s", actual == expected ? "true" : "false");
        if (actual != expected) return 5;
        ++success;
      }
      printf("}\n");
      fflush(stdout);
    }
    RT(cudaSetDevice(owner));
    DR(cuMemUnmap(pointer, bytes));
    DR(cuMemAddressFree(pointer, bytes));
    DR(cuMemRelease(handle));
  }
  printf("{\"successful_mappings\":%d,\"total\":64}\n", success);
  for (int device = 0; device < devices; ++device) {
    RT(cudaSetDevice(device));
    RT(cudaFree(outputs[device]));
  }
  return success == 64 ? 0 : 6;
}
