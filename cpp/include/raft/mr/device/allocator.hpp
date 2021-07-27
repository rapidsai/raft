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

#include <cstddef>
#include <raft/mr/allocator.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace raft {
namespace mr {
namespace device {

/**
 * @brief An explicit interface for an asynchronous device allocator.
 *
 * This is mostly done in order to reduce work needed in cuML codebase.
 * An implementation of this interface can make the following assumptions,
 * further to the ones listed in `Allocator`:
 * - Allocations may be always on the device that was specified on construction.
 */
class allocator : public base_allocator {
};

/** Default device allocator based on the one provided by RMM */
class default_allocator : public allocator {
 public:
  void* allocate(std::size_t n, cudaStream_t stream) override
  {
    void* ptr = rmm::mr::get_current_device_resource()->allocate(n, stream);
    return ptr;
  }

  void deallocate(void* p, std::size_t n, cudaStream_t stream) override
  {
    rmm::mr::get_current_device_resource()->deallocate(p, n, stream);
  }
};  // class default_allocator

};  // namespace device
};  // namespace mr
};  // namespace raft
