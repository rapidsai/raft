/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include "execution_device_id_base.hpp"
#include <raft/core/device_type.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/cuda_device.hpp>

namespace raft {
namespace detail {
template <>
struct execution_device_id<device_type::gpu> {
  using value_type = typename rmm::cuda_device_id::value_type;
  execution_device_id() noexcept(false)
    : id_{[]() {
        auto raw_id = value_type{};
        RAFT_CUDA_TRY(cudaGetDevice(&raw_id));
        return raw_id;
      }()} {};
  /* We do not mark this constructor as explicit to allow public API
   * functions to accept `device_id` arguments without requiring
   * downstream consumers to explicitly construct a device_id. Thus,
   * consumers can use the type they expect to use when specifying a device
   * (int), but once we are inside the public API, the device type remains
   * attached to this value and we can easily convert to the strongly-typed
   * rmm::cuda_device_id if desired.
   */
  execution_device_id(value_type dev_id) noexcept : id_{dev_id} {};

  auto value() const noexcept { return id_.value(); }
  auto rmm_id() const noexcept { return id_; }

 private:
  rmm::cuda_device_id id_;
};
}  // namespace detail
}  // namespace raft