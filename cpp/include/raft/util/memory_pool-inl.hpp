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
#include <memory>

#include <raft/util/inline.hpp>  // RAFT_INLINE_CONDITIONAL
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

namespace raft {

RAFT_INLINE_CONDITIONAL std::unique_ptr<rmm::mr::device_memory_resource> get_pool_memory_resource(
  rmm::mr::device_memory_resource*& mr, size_t initial_size)
{
  using pool_res_t = rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>;
  std::unique_ptr<pool_res_t> pool_res{};
  if (mr) return pool_res;
  mr = rmm::mr::get_current_device_resource();
  if (!dynamic_cast<pool_res_t*>(mr) &&
      !dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>*>(mr) &&
      !dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource>*>(mr)) {
    pool_res = std::make_unique<pool_res_t>(mr, (initial_size + 255) & (~255));
    mr       = pool_res.get();
  }
  return pool_res;
}

}  // namespace raft
