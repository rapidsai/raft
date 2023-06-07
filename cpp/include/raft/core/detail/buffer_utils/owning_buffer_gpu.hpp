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
#include "owning_buffer_base.hpp"
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdarray.hpp>
#include <cuda_runtime_api.h>
#include <raft/core/device_type.hpp>

namespace raft {
namespace detail {
  template <typename ElementType,
  typename Extents,
  typename LayoutPolicy                        = layout_c_contiguous,
  template <typename> typename ContainerPolicy = device_uvector_policy>
struct owning_buffer<ElementType, device_type::gpu, Extents, LayoutPolicy, ContainerPolicy> {
  using element_type     = std::remove_cv_t<ElementType>;
  using container_policy = ContainerPolicy<element_type>;
  using owning_device_buffer = device_mdarray<element_type, Extents, LayoutPolicy, container_policy>;
  
  owning_buffer() : data_{} {}

  owning_buffer(raft::resources const& handle, Extents extents) noexcept(false)
    : extents_{extents}, data_{[&extents, handle]() {
        typename owning_device_buffer::mapping_type layout{extents};
        typename owning_device_buffer::container_policy_type policy{};
      owning_device_buffer device_data{handle, layout, policy};
      return device_data;
      }()}
  {
  }

  auto* get() const { return const_cast<ElementType*>(data_.data_handle()); }

 private:
  Extents extents_;
  owning_device_buffer data_;
};
}  // namespace detail
}  // namespace raft