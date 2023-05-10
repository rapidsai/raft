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
          typename Extents>
struct owning_buffer<ElementType, device_type::gpu, Extents> {
  using element_type     = std::remove_cv_t<ElementType>;
  using container_policy = device_uvector_policy<element_type>;
  using owning_device_buffer = device_mdarray<element_type, Extents, layout_c_contiguous, container_policy>;
  
  owning_buffer() : data_{} {}

  owning_buffer(raft::resources const& handle, Extents extents) noexcept(false)
    : extents_{extents}, data_{[&extents, handle]() {
        // return rmm::device_buffer{size * sizeof(value_type), raft::resource::get_cuda_stream(handle)};
        typename owning_device_buffer::mapping_type layout{extents};
        typename owning_device_buffer::container_policy_type policy{};
      return owning_device_buffer{handle, layout, policy};
      }()}
  {
  }

  auto* get() const { return reinterpret_cast<ElementType*>(data_.data_handle()); }

 private:
  Extents extents_;
  owning_device_buffer data_;
};
}  // namespace detail
}  // namespace raft