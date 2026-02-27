/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/detail/span.hpp>  // dynamic_extent
#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/util/cudart_utils.hpp>

namespace raft {
/**
 * @brief A container policy for managed mdarray.
 */
template <typename ElementType>
struct managed_uvector_policy {
  using element_type    = ElementType;
  using container_type  = device_uvector<element_type>;
  using pointer         = typename container_type::pointer;
  using const_pointer   = typename container_type::const_pointer;
  using reference       = device_reference<element_type>;
  using const_reference = device_reference<element_type const>;

  using accessor_policy       = cuda::std::default_accessor<element_type>;
  using const_accessor_policy = cuda::std::default_accessor<element_type const>;

  auto create(raft::resources const& res, size_t n) -> container_type
  {
    return container_type(
      n, resource::get_cuda_stream(res), raft::resource::get_managed_memory_resource(res));
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
