/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include "cudart_utils.h"

namespace raft {

/**
 * @brief Interface for an asynchronous device/host allocator.
 *
 * An implementation of this interface can make the following assumptions:
 * - It does not need to be but it can allow async allocate and deallocate.
 *
 * @note This interface does NOT support RAII. Thus, if you need RAII-enabled
 *       interface, better to use `device_buffer` or `host_buffer`.
 */
class allocator {
 public:
  /**
   * @brief Asynchronously allocates a memory region.
   *
   * An implementation of this need to return a allocation of n bytes properly
   * align bytes on the configured device. The allocation can optionally be
   * asynchronous in the sense that it is only save to use after all work
   * submitted to the passed in stream prior to the call to allocate has
   * completed. If the allocation is used before, e.g. in another stream the
   * behaviour may be undefined.
   * @todo: Add alignment requirments.
   *
   * @param[in] n         number of bytes to allocate
   * @param[in] stream    stream to issue the possible asynchronous allocation in
   */
  virtual void* allocate(std::size_t n, cudaStream_t stream) = 0;

  /**
   * @brief Asynchronously deallocates device memory
   *
   * An implementation of this need to ensure that the allocation that the
   * passed in pointer points to remains usable until all work sheduled in
   * stream prior to the call to deallocate has completed.
   *
   * @param[inout] p      pointer to the buffer to deallocte
   * @param[in] n         size of the buffer to deallocte in bytes
   * @param[in] stream    stream in which the allocation might be still in use
   */
  virtual void deallocate(void* p, std::size_t n, cudaStream_t stream) = 0;

  virtual ~allocator() = default;
};  // class Allocator

/**
 * @brief An explicit interface for an asynchronous device allocator.
 *
 * This is mostly done in order to reduce work needed in cuML codebase.
 * An implementation of this interface can make the following assumptions,
 * further to the ones listed in `Allocator`:
 * - Allocations may be always on the device that was specified on construction.
 */
class device_allocator : public allocator {};

/**
 * @brief An explicit interface for an asynchronous host allocations.
 *
 * This is mostly done in order to reduce work needed in cuML codebase.
 * An implementation of this interface can make the following assumptions,
 * further to the ones listed in `Allocator`:
 * - Allocations don't need to be zero copy accessible form a device.
 */
class host_allocator : public allocator {};

/** Default cudaMalloc/cudaFree based device allocator */
class default_device_allocator : public device_allocator {
 public:
  void* allocate(std::size_t n, cudaStream_t stream) override {
    void* ptr = 0;
    CUDA_CHECK(cudaMalloc(&ptr, n));
    return ptr;
  }

  void deallocate(void* p, std::size_t n, cudaStream_t stream) override {
    ///@todo: enable this once logging is enabled in raft
    //CUDA_CHECK_NO_THROW(cudaFree(p));
    CUDA_CHECK(cudaFree(p));
  }
};  // class default_device_allocator

/** Default cudaMallocHost/cudaFreeHost based host allocator */
class default_host_allocator : public host_allocator {
 public:
  void* allocate(std::size_t n, cudaStream_t stream) override {
    void* ptr = 0;
    CUDA_CHECK(cudaMallocHost(&ptr, n));
    return ptr;
  }

  void deallocate(void* p, std::size_t n, cudaStream_t stream) override {
    ///@todo: enable this once logging is enabled in raft
    //CUDA_CHECK_NO_THROW(cudaFreeHost(p));
    CUDA_CHECK(cudaFreeHost(p));
  }
};  // class default_host_allocator

};  // end namespace raft
