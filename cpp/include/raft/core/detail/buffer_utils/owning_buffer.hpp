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
#include <raft/core/host_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/device_mdarray.hpp>
#endif
#include <memory>
#include <raft/core/device_type.hpp>
#include <raft/core/host_container_policy.hpp>
#include <raft/core/buffer_container_policy.hpp>
#include <type_traits>
#include <variant>

namespace raft {
namespace detail {
  template <typename ElementType,
  typename Extents,
  typename LayoutPolicy,
  template <typename> typename ContainerPolicy>
struct owning_host_buffer {
  using element_type     = std::remove_cv_t<ElementType>;
  using container_policy = std::conditional_t<std::is_same_v<buffer_container_policy<element_type>, ContainerPolicy<element_type>>,
                                                          std::variant_alternative_t<0, buffer_container_policy<element_type>>,
                                                          ContainerPolicy<element_type>>;
  using index_type       = typename Extents::index_type;
  using buffer = host_mdarray<element_type, Extents, LayoutPolicy, container_policy>;
  owning_host_buffer(raft::resources const& handle, Extents extents) noexcept(false)
    : extents_{extents}, data_{[&extents, handle]() {
        typename buffer::mapping_type layout{extents};
        typename buffer::container_policy_type policy{};
      buffer host_data{handle, layout, policy};
      return host_data;
      }()}
  {
  }

   auto* get() const { return const_cast<ElementType*>(data_.data_handle()); }

   auto view() {
    return data_.view();
  }

 private:
  Extents extents_;
  buffer data_;
};

#ifndef RAFT_DISABLE_CUDA
template <typename ElementType,
  typename Extents,
  typename LayoutPolicy,
  template <typename> typename ContainerPolicy>
struct owning_device_buffer {
  using element_type     = std::remove_cv_t<ElementType>;
  using container_policy = std::conditional_t<std::is_same_v<buffer_container_policy<element_type>, ContainerPolicy<element_type>>,
                                                          std::variant_alternative_t<1, buffer_container_policy<element_type>>,
                                                          ContainerPolicy<element_type>>;
  using index_type       = typename Extents::index_type;
  using buffer = device_mdarray<element_type, Extents, LayoutPolicy, container_policy>;
  
  owning_device_buffer() : data_{} {}

  owning_device_buffer(raft::resources const& handle, Extents extents) noexcept(false)
    : extents_{extents}, data_{[&extents, handle]() {
        typename buffer::mapping_type layout{extents};
        typename buffer::container_policy_type policy{};
      buffer device_data{handle, layout, policy};
      return device_data;
      }()}
  {
  }

  auto* get() const {return const_cast<ElementType*>(data_.data_handle());}

  auto view() {
    data_.view();
  }
 private:
  Extents extents_;
  buffer data_;
};
#else
template <typename ElementType,
  typename Extents,
  typename LayoutPolicy,
  template <typename> typename ContainerPolicy>
struct owning_device_buffer {
  owning_device_buffer(raft::resources const& handle, Extents extents) : extents_(extents){}
  auto* get() const { return static_cast<ElementType*>(nullptr); }

  auto view() {
    return host_mdspan<ElementType, Extents>(nullptr, exts);
  }

  private:
  Extents extents_;
};
#endif
}  // namespace detail
}  // namespace raft