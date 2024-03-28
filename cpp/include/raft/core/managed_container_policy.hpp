/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/detail/span.hpp>  // dynamic_extent
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/managed_memory_resource.hpp>

namespace raft {
/**
 * @brief A container policy for managed mdarray.
 */
template <typename ElementType>
class managed_uvector_policy {
 public:
  using element_type    = ElementType;
  using container_type  = device_uvector<element_type>;
  using pointer         = typename container_type::pointer;
  using const_pointer   = typename container_type::const_pointer;
  using reference       = device_reference<element_type>;
  using const_reference = device_reference<element_type const>;

  using accessor_policy       = std::experimental::default_accessor<element_type>;
  using const_accessor_policy = std::experimental::default_accessor<element_type const>;

  auto create(raft::resources const& res, size_t n) -> container_type
  {
    return container_type(n, resource::get_cuda_stream(res), mr_);
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
  static auto* get_default_memory_resource()
  {
    auto static result = rmm::mr::managed_memory_resource{};
    return &result;
  }
  rmm::mr::managed_memory_resource* mr_{get_default_memory_resource()};
};

}  // namespace raft
#else
#include <raft/core/detail/fail_container_policy.hpp>
namespace raft {

// Provide placeholders that will allow CPU-GPU interoperable codebases to
// compile in non-CUDA mode but which will throw exceptions at runtime on any
// attempt to touch device data

template <typename ElementType>
using managed_uvector_policy = detail::fail_container_policy<ElementType>;

}  // namespace raft
#endif
