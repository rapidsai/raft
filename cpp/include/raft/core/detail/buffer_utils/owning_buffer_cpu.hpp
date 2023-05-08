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
#include <raft/core/host_mdarray.hpp>
#include <memory>
#include <raft/core/device_type.hpp>
#include <raft/core/host_container_policy.hpp>
#include <type_traits>

namespace raft {
namespace detail {
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy = layout_c_contiguous,
          template <typename T>
          typename ContainerPolicy = host_vector_policy>
struct owning_buffer<ElementType, device_type::cpu, Extents, LayoutPolicy, ContainerPolicy> {
  using element_type     = std::remove_cv_t<ElementType>;
  using index_type       = typename Extents::index_type;
  using container_policy = ContainerPolicy<element_type>;
  using owning_host_buffer = host_mdarray<element_type, Extents, LayoutPolicy, container_policy>;
  owning_buffer(raft::resources const& handle, Extents extents) noexcept(false)
    : extents_{extents}, data_{[&extents, handle]() {
        // return rmm::device_buffer{size * sizeof(value_type), raft::resource::get_cuda_stream(handle)};
        typename owning_host_buffer::mapping_type layout{extents};
        typename owning_host_buffer::container_policy_type policy{};
      return owning_host_buffer{handle, layout, policy};
      }()}
  {
  }

   auto* get() const { return reinterpret_cast<ElementType*>(data_.data_handle()); }

 private:
  // TODO(wphicks): Back this with RMM-allocated host memory
  Extents extents_;
  owning_host_buffer data_;
};
}  // namespace detail
}  // namespace raft