/*
 * Copyright (2019) Sandia Corporation
 *
 * The source code is licensed under the 3-clause BSD license found in the LICENSE file
 * thirdparty/LICENSES/mdarray.license
 */

/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <vector>
#ifndef RAFT_DISABLE_CUDA
#include <thrust/host_vector.h>
#include <thrust/mr/allocator.h>
#include <thrust/system/cuda/memory_resource.h>
#else
#include <raft/core/detail/fail_container_policy.hpp>
#endif

namespace raft {

/**
 * @brief A container policy for host mdarray.
 */
template <typename ElementType, typename Allocator = std::allocator<ElementType>>
class host_vector_policy {
 public:
  using element_type          = ElementType;
  using container_type        = std::vector<element_type, Allocator>;
  using allocator_type        = typename container_type::allocator_type;
  using pointer               = typename container_type::pointer;
  using const_pointer         = typename container_type::const_pointer;
  using reference             = element_type&;
  using const_reference       = element_type const&;
  using accessor_policy       = std::experimental::default_accessor<element_type>;
  using const_accessor_policy = std::experimental::default_accessor<element_type const>;

 public:
  auto create(raft::resources const&, size_t n) -> container_type { return container_type(n); }

  constexpr host_vector_policy() noexcept(std::is_nothrow_default_constructible_v<ElementType>) =
    default;

  [[nodiscard]] constexpr auto access(container_type& c, size_t n) const noexcept -> reference
  {
    return c[n];
  }
  [[nodiscard]] constexpr auto access(container_type const& c, size_t n) const noexcept
    -> const_reference
  {
    return c[n];
  }

  [[nodiscard]] auto make_accessor_policy() noexcept { return accessor_policy{}; }
  [[nodiscard]] auto make_accessor_policy() const noexcept { return const_accessor_policy{}; }
};

#ifndef RAFT_DISABLE_CUDA
/**
 * @brief A container policy for pinned mdarray.
 */
template <typename ElementType>
struct pinned_vector_policy {
  using element_type = ElementType;
  using allocator_type =
    thrust::mr::stateless_resource_allocator<element_type,
                                             thrust::cuda::universal_host_pinned_memory_resource>;
  using container_type        = thrust::host_vector<element_type, allocator_type>;
  using pointer               = typename container_type::pointer;
  using const_pointer         = typename container_type::const_pointer;
  using reference             = element_type&;
  using const_reference       = element_type const&;
  using accessor_policy       = std::experimental::default_accessor<element_type>;
  using const_accessor_policy = std::experimental::default_accessor<element_type const>;

  auto create(raft::resources const&, size_t n) -> container_type
  {
    return container_type(n, allocator_);
  }

  constexpr pinned_vector_policy() noexcept(std::is_nothrow_default_constructible_v<ElementType>)
    : mr_{}, allocator_{&mr_}
  {
  }

  [[nodiscard]] constexpr auto access(container_type& c, size_t n) const noexcept -> reference
  {
    return c[n];
  }
  [[nodiscard]] constexpr auto access(container_type const& c, size_t n) const noexcept
    -> const_reference
  {
    return c[n];
  }

  [[nodiscard]] auto make_accessor_policy() noexcept { return accessor_policy{}; }
  [[nodiscard]] auto make_accessor_policy() const noexcept { return const_accessor_policy{}; }

 private:
  thrust::system::cuda::universal_host_pinned_memory_resource mr_;
  allocator_type allocator_;
};
#else
template <typename ElementType>
using pinned_vector_policy = detail::fail_container_policy<ElementType>;
#endif
}  // namespace raft
