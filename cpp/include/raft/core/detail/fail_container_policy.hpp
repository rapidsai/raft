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
#include <raft/core/error.hpp>
#include <raft/core/logger-macros.hpp>
#include <raft/core/resources.hpp>
#include <raft/thirdparty/mdspan/include/experimental/mdspan>

#include <stddef.h>

namespace raft {
namespace detail {

template <typename T>
struct fail_reference {
  using value_type    = typename std::remove_cv_t<T>;
  using pointer       = T*;
  using const_pointer = T const*;

  fail_reference() = default;
  template <typename StreamViewType>
  fail_reference(T* ptr, StreamViewType stream)
  {
    throw non_cuda_build_error{"Attempted to construct reference to device data in non-CUDA build"};
  }

  operator value_type() const  // NOLINT
  {
    throw non_cuda_build_error{"Attempted to dereference device data in non-CUDA build"};
    return value_type{};
  }
  auto operator=(T const& other) -> fail_reference&
  {
    throw non_cuda_build_error{"Attempted to assign to device data in non-CUDA build"};
    return *this;
  }
};

/** A placeholder container which throws an exception on use
 *
 * This placeholder is used in non-CUDA builds for container types that would
 * otherwise be provided with CUDA code. Attempting to construct a non-empty
 * container of this type throws an exception indicating that there was an
 * attempt to use the device from a non-CUDA build. An example of when this
 * might happen is if a downstream application attempts to allocate a device
 * mdarray using a library built with non-CUDA RAFT.
 */
template <typename T>
struct fail_container {
  using value_type = T;
  using size_type  = std::size_t;

  using reference       = fail_reference<T>;
  using const_reference = fail_reference<T const>;

  using pointer       = value_type*;
  using const_pointer = value_type const*;

  using iterator       = pointer;
  using const_iterator = const_pointer;

  explicit fail_container(size_t n = size_t{})
  {
    if (n != size_t{}) {
      throw non_cuda_build_error{"Attempted to allocate device container in non-CUDA build"};
    }
  }

  template <typename Index>
  auto operator[](Index i) noexcept -> reference
  {
    RAFT_LOG_ERROR("Attempted to access device data in non-CUDA build");
    return reference{};
  }

  template <typename Index>
  auto operator[](Index i) const noexcept -> const_reference
  {
    RAFT_LOG_ERROR("Attempted to access device data in non-CUDA build");
    return const_reference{};
  }
  void resize(size_t n)
  {
    if (n != size_t{}) {
      throw non_cuda_build_error{"Attempted to allocate device container in non-CUDA build"};
    }
  }

  [[nodiscard]] auto data() noexcept -> pointer { return nullptr; }
  [[nodiscard]] auto data() const noexcept -> const_pointer { return nullptr; }
};

/** A placeholder container policy which throws an exception on use
 *
 * This placeholder is used in non-CUDA builds for container types that would
 * otherwise be provided with CUDA code. Attempting to construct a non-empty
 * container of this type throws an exception indicating that there was an
 * attempt to use the device from a non-CUDA build. An example of when this
 * might happen is if a downstream application attempts to allocate a device
 * mdarray using a library built with non-CUDA RAFT.
 */
template <typename ElementType>
struct fail_container_policy {
  using element_type    = ElementType;
  using container_type  = fail_container<element_type>;
  using pointer         = typename container_type::pointer;
  using const_pointer   = typename container_type::const_pointer;
  using reference       = typename container_type::reference;
  using const_reference = typename container_type::const_reference;

  using accessor_policy       = std::experimental::default_accessor<element_type>;
  using const_accessor_policy = std::experimental::default_accessor<element_type const>;

  auto create(raft::resources const& res, size_t n) -> container_type { return container_type(n); }

  fail_container_policy() = default;

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

}  // namespace detail
}  // namespace raft
