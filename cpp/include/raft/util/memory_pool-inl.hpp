/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/detail/macros.hpp>  // RAFT_INLINE_CONDITIONAL

#include <rmm/aligned.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstddef>
#include <memory>

namespace raft {

/**
 * @defgroup memory_pool Memory Pool
 * @{
 */
/**
 * @brief Get a pointer to a pooled memory resource within the scope of the lifetime of the returned
 * unique pointer.
 *
 * This function is useful in the code where multiple repeated allocations/deallocations are
 * expected.
 * Use case example:
 * @code{.cpp}
 *   void my_func(..., size_t n, rmm::mr::device_memory_resource* mr = nullptr) {
 *     auto pool_guard = raft::get_pool_memory_resource(mr, 2 * n * sizeof(float));
 *     if (pool_guard){
 *       RAFT_LOG_INFO("Created a pool");
 *     } else {
 *       RAFT_LOG_INFO("Using the current default or explicitly passed device memory resource");
 *     }
 *     rmm::device_uvector<float> x(n, stream, mr);
 *     rmm::device_uvector<float> y(n, stream, mr);
 *     ...
 *   }
 * @endcode
 * Here, the new memory resource would be created within the function scope if the passed `mr` is
 * null and the default resource is not a pool. After the call, `mr` contains a valid memory
 * resource in any case.
 *
 * @param[inout] mr if not null do nothing; otherwise get the current device resource and wrap it
 * into a `pool_memory_resource` if necessary and return the pointer to the result.
 * @param initial_size if a new memory pool is created, this would be its initial size (rounded up
 * to 256 bytes).
 *
 * @return if a new memory pool is created, it returns a unique_ptr to it;
 *   this managed pointer controls the lifetime of the created memory resource.
 */
RAFT_INLINE_CONDITIONAL std::unique_ptr<rmm::mr::device_memory_resource> get_pool_memory_resource(
  rmm::mr::device_memory_resource*& mr, size_t initial_size)
{
  using pool_res_t = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  std::unique_ptr<pool_res_t> pool_res{nullptr};
  if (mr) return pool_res;
  mr = rmm::mr::get_current_device_resource();
  if (!dynamic_cast<pool_res_t*>(mr) &&
      !dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>*>(mr) &&
      !dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>*>(mr)) {
    pool_res = std::make_unique<pool_res_t>(
      mr, rmm::align_down(initial_size, rmm::CUDA_ALLOCATION_ALIGNMENT));
    mr = pool_res.get();
  }
  return pool_res;
}

/** @} */
}  // namespace raft
