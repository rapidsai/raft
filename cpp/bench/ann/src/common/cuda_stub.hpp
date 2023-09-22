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

/*
The content of this header is governed by two preprocessor definitions:

  - BUILD_CPU_ONLY - whether none of the CUDA functions are used.
  - ANN_BENCH_LINK_CUDART - dynamically link against this string if defined.

______________________________________________________________________________
|BUILD_CPU_ONLY | ANN_BENCH_LINK_CUDART |         cudart      | cuda_runtime_api.h |
|         |                       |  found    |  needed |      included      |
|---------|-----------------------|-----------|---------|--------------------|
|   ON    |    <not defined>      |  false    |  false  |       NO           |
|   ON    |   "cudart.so.xx.xx"   |  false    |  false  |       NO           |
|  OFF    |     <nod defined>     |   true    |   true  |      YES           |
|  OFF    |   "cudart.so.xx.xx"   | <runtime> |   true  |      YES           |
------------------------------------------------------------------------------
*/

#ifndef BUILD_CPU_ONLY
#include <cuda_runtime_api.h>
#ifdef ANN_BENCH_LINK_CUDART
#include <cstring>
#include <dlfcn.h>
#endif
#else
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
#endif

namespace raft::bench::ann {

struct cuda_lib_handle {
  void* handle{nullptr};
  explicit cuda_lib_handle()
  {
#ifdef ANN_BENCH_LINK_CUDART
    constexpr int kFlags = RTLD_NOW | RTLD_GLOBAL | RTLD_DEEPBIND | RTLD_NODELETE;
    // The full name of the linked cudart library 'cudart.so.MAJOR.MINOR.PATCH'
    char libname[] = ANN_BENCH_LINK_CUDART;  // NOLINT
    handle         = dlopen(ANN_BENCH_LINK_CUDART, kFlags);
    if (handle != nullptr) { return; }
    // try strip the PATCH
    auto p = strrchr(libname, '.');
    p[0]   = 0;
    handle = dlopen(libname, kFlags);
    if (handle != nullptr) { return; }
    // try set the MINOR version to 0
    p      = strrchr(libname, '.');
    p[1]   = '0';
    p[2]   = 0;
    handle = dlopen(libname, kFlags);
    if (handle != nullptr) { return; }
    // try strip the MINOR
    p[0]   = 0;
    handle = dlopen(libname, kFlags);
    if (handle != nullptr) { return; }
    // try strip the MAJOR
    p      = strrchr(libname, '.');
    p[0]   = 0;
    handle = dlopen(libname, kFlags);
#endif
  }
  ~cuda_lib_handle() noexcept
  {
#ifdef ANN_BENCH_LINK_CUDART
    if (handle != nullptr) { dlclose(handle); }
#endif
  }

  template <typename Symbol>
  auto sym(const char* name) -> Symbol
  {
#ifdef ANN_BENCH_LINK_CUDART
    return reinterpret_cast<Symbol>(dlsym(handle, name));
#else
    return nullptr;
#endif
  }

  /** Whether this is NOT a cpu-only package. */
  [[nodiscard]] constexpr inline auto needed() const -> bool
  {
#if defined(BUILD_CPU_ONLY)
    return false;
#else
    return true;
#endif
  }

  /** CUDA found, either at compile time or at runtime. */
  [[nodiscard]] inline auto found() const -> bool
  {
#if defined(BUILD_CPU_ONLY)
    return false;
#elif defined(ANN_BENCH_LINK_CUDART)
    return handle != nullptr;
#else
    return true;
#endif
  }
};

static inline cuda_lib_handle cudart{};

#ifdef ANN_BENCH_LINK_CUDART
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

#define RAFT_DECLARE_CUDART(fun)           \
  static inline decltype(&stub::fun) fun = \
    cudart.found() ? cudart.sym<decltype(&stub::fun)>(#fun) : &stub::fun

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
