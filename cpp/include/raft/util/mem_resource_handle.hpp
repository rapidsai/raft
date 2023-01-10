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

#include <raft/core/error.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/optional.h>

#include <memory>
#include <optional>

namespace raft {

/** An alias to an RMM memory resource with its lifetime managed by a smart pointer. */
using device_mem_resource = std::shared_ptr<rmm::mr::device_memory_resource>;

/**
 * Create a new device memory resource.
 * *
 * @return A shared pointer to an RMM device memory resource.
 */
inline auto make_cuda_mem_resource() -> device_mem_resource
{
  return std::make_shared<rmm::mr::cuda_memory_resource>();
}

/**
 * Create a new managed memory resource.
 * *
 * @return A shared pointer to an RMM managed memory resource.
 */
inline auto make_managed_mem_resource() -> device_mem_resource
{
  return std::make_shared<rmm::mr::managed_memory_resource>();
}

/**
 * Create a new pool memory resource.
 *
 * @param upstream
 *   optional underlying memory resource; the current global device resource is used,
 *   if none provided.
 * @param initial_pool_size
 *   Minimum size, in bytes, of the initial pool.
 *   The default value is selected as in `rmm::mr::pool_memory_resource`.
 * @param maximum_pool_size
 *   Maximum size, in bytes, that the pool can grow to.
 *   The default value is selected as in `rmm::mr::pool_memory_resource`.
 *
 * @return A shared pointer to an RMM pool memory resource.
 */
inline auto make_pool_mem_resource(std::optional<device_mem_resource> upstream  = std::nullopt,
                                   std::optional<std::size_t> initial_pool_size = std::nullopt,
                                   std::optional<std::size_t> maximum_pool_size = std::nullopt)
  -> device_mem_resource
{
  return std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>(
    upstream.has_value() ? upstream.value().get() : rmm::mr::get_current_device_resource(),
    initial_pool_size.has_value() ? thrust::make_optional(initial_pool_size.value())
                                  : thrust::nullopt,
    maximum_pool_size.has_value() ? thrust::make_optional(maximum_pool_size.value())
                                  : thrust::nullopt);
}

}  // namespace raft
