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
#include <cstddef>
#include <memory>                                    // std::unique_ptr
#include <rmm/mr/device/device_memory_resource.hpp>  // rmm::mr::device_memory_resource

namespace raft {

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
 *       RAFT_LOG_INFO("Created a pool %zu bytes", pool_guard->pool_size());
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
std::unique_ptr<rmm::mr::device_memory_resource> get_pool_memory_resource(
  rmm::mr::device_memory_resource*& mr, size_t initial_size);

}  // namespace raft
