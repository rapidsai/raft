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

#include <raft/cudart_utils.h>
#include <raft/mr/allocator.hpp>

#include <cuda_runtime.h>

#include <cstddef>

namespace raft {
namespace mr {
namespace host {

/**
 * @brief An explicit interface for an asynchronous host allocations.
 *
 * This is mostly done in order to reduce work needed in cuML codebase.
 * An implementation of this interface can make the following assumptions,
 * further to the ones listed in `Allocator`:
 * - Allocations don't need to be zero copy accessible form a device.
 */
class allocator : public base_allocator {
};

/** Default cudaMallocHost/cudaFreeHost based host allocator */
class default_allocator : public allocator {
 public:
  void* allocate(std::size_t n, cudaStream_t stream) override
  {
    void* ptr = nullptr;
    RAFT_CHECK_CUDA(cudaMallocHost(&ptr, n));
    return ptr;
  }

  void deallocate(void* p, std::size_t n, cudaStream_t stream) override
  {
    // Must call _NO_THROW here since this is called frequently from object
    // destructors which are "nothrow" by default
    RAFT_CHECK_CUDA_NO_THROW(cudaFreeHost(p));
  }
};  // class default_allocator

};  // namespace host
};  // namespace mr
};  // namespace raft
