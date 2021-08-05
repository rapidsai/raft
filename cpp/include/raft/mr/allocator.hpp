/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cuda_runtime_api.h>

#include <cstddef>

namespace raft {
namespace mr {

/**
 * @brief Interface for an asynchronous device/host allocator.
 *
 * An implementation of this interface can make the following assumptions:
 * - It does not need to be but it can allow async allocate and deallocate.
 *
 * @note This interface does NOT support RAII. Thus, if you need RAII-enabled
 *       interface, better to use `device_buffer` or `host_buffer`.
 */
class base_allocator {
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

  virtual ~base_allocator() = default;
};  // class base_allocator

};  // namespace mr
};  // namespace raft
