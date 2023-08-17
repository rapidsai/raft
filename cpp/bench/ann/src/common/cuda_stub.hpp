/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#ifdef ANN_BENCH_LINK_CUDART
#include <cuda_runtime_api.h>
#else
#define CPU_ONLY
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
#endif

#include <dlfcn.h>

namespace raft::bench::ann {

struct cuda_lib_handle {
  void* handle{nullptr};
  explicit cuda_lib_handle()
  {
#ifdef ANN_BENCH_LINK_CUDART
    handle = dlopen(ANN_BENCH_LINK_CUDART, RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND | RTLD_NODELETE);
#endif
  }
  ~cuda_lib_handle() noexcept
  {
    if (handle != nullptr) { dlclose(handle); }
  }

  [[nodiscard]] inline auto found() const -> bool { return handle != nullptr; }
};

static inline cuda_lib_handle cudart{};

#ifndef CPU_ONLY
namespace stub {

[[gnu::weak, gnu::noinline]] cudaError_t cudaMemcpy(void* dst,
                                                    const void* src,
                                                    size_t count,
                                                    enum cudaMemcpyKind kind)
{
  return cudaSuccess;
}

[[gnu::weak, gnu::noinline]] cudaError_t cudaMalloc(void** ptr, size_t size)
{
  *ptr = nullptr;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaMemset(void* devPtr, int value, size_t count)
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaFree(void* devPtr) { return cudaSuccess; }
[[gnu::weak, gnu::noinline]] cudaError_t cudaStreamCreate(cudaStream_t* pStream)
{
  *pStream = 0;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream,
                                                                   unsigned int flags)
{
  *pStream = 0;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaStreamDestroy(cudaStream_t pStream)
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaStreamSynchronize(cudaStream_t pStream)
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaEventCreate(cudaEvent_t* event)
{
  *event = 0;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaEventElapsedTime(float* ms,
                                                              cudaEvent_t start,
                                                              cudaEvent_t end)
{
  *ms = 0;
  return cudaSuccess;
}
[[gnu::weak, gnu::noinline]] cudaError_t cudaEventDestroy(cudaEvent_t event) { return cudaSuccess; }
[[gnu::weak, gnu::noinline]] cudaError_t cudaGetDevice(int* device)
{
  *device = 0;
  return cudaSuccess;
};
[[gnu::weak, gnu::noinline]] cudaError_t cudaDriverGetVersion(int* driver)
{
  *driver = 0;
  return cudaSuccess;
};
[[gnu::weak, gnu::noinline]] cudaError_t cudaRuntimeGetVersion(int* runtime)
{
  *runtime = 0;
  return cudaSuccess;
};
[[gnu::weak, gnu::noinline]] cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop,
                                                                 int device)
{
  *prop = cudaDeviceProp{};
  return cudaSuccess;
}

}  // namespace stub

#define RAFT_DECLARE_CUDART(fun)                                                        \
  static inline decltype(&stub::fun) fun =                                              \
    cudart.found() ? reinterpret_cast<decltype(&stub::fun)>(dlsym(cudart.handle, #fun)) \
                   : &stub::fun

RAFT_DECLARE_CUDART(cudaMemcpy);
RAFT_DECLARE_CUDART(cudaMalloc);
RAFT_DECLARE_CUDART(cudaMemset);
RAFT_DECLARE_CUDART(cudaFree);
RAFT_DECLARE_CUDART(cudaStreamCreate);
RAFT_DECLARE_CUDART(cudaStreamCreateWithFlags);
RAFT_DECLARE_CUDART(cudaStreamDestroy);
RAFT_DECLARE_CUDART(cudaStreamSynchronize);
RAFT_DECLARE_CUDART(cudaEventCreate);
RAFT_DECLARE_CUDART(cudaEventRecord);
RAFT_DECLARE_CUDART(cudaEventSynchronize);
RAFT_DECLARE_CUDART(cudaEventElapsedTime);
RAFT_DECLARE_CUDART(cudaEventDestroy);
RAFT_DECLARE_CUDART(cudaGetDevice);
RAFT_DECLARE_CUDART(cudaDriverGetVersion);
RAFT_DECLARE_CUDART(cudaRuntimeGetVersion);
RAFT_DECLARE_CUDART(cudaGetDeviceProperties);

#undef RAFT_DECLARE_CUDART
#endif

};  // namespace raft::bench::ann
