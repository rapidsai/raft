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
#include "raft/core/buffer_container_policy.hpp"
#include "raft/core/host_container_policy.hpp"
#include "raft/core/host_device_accessor.hpp"
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>
#include <type_traits>

namespace raft {
namespace detail {
template <typename ElementType,
          memory_type M,
          typename Extents,
          typename LayoutPolicy = layout_c_contiguous,
          template <typename> typename ContainerPolicy = buffer_container_policy>
struct non_owning_buffer {
  using container_policy = std::conditional_t<std::is_same_v<buffer_container_policy<ElementType>, ContainerPolicy<ElementType>>,
                                                          std::variant_alternative_t<0, buffer_container_policy<ElementType>>,
                                                          ContainerPolicy<ElementType>>;
  using accessor_policy = typename container_policy::accessor_policy;
  using index_type       = typename Extents::index_type;

  non_owning_buffer() : data_{nullptr} {}

  non_owning_buffer(ElementType* ptr, Extents extents) : data_{ptr}, extents_{extents} {
  }

  auto* get() const { return data_; }

  auto view() {
    using accessor_type = host_device_accessor<
    accessor_policy, M>();
    return mdspan<ElementType, Extents, LayoutPolicy, accessor_type>{data_, extents_};
  }
 private:
  ElementType* data_;
  Extents extents_;
};

}  // namespace detail
}  // namespace raft