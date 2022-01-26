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
#include <experimental/mdspan>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/device_ptr.h>

namespace raft::detail {
/**
 * @brief A simplified version of thrust::device_reference with support for CUDA stream.
 */
template <typename T>
class device_reference {
 public:
  using value_type    = typename std::remove_cv_t<T>;
  using pointer       = thrust::device_ptr<T>;
  using const_pointer = thrust::device_ptr<T const>;

 private:
  std::conditional_t<std::is_const<T>::value, const_pointer, pointer> ptr_;
  rmm::cuda_stream_view stream_;

 public:
  device_reference(thrust::device_ptr<T> ptr, rmm::cuda_stream_view stream)
    : ptr_{ptr}, stream_{stream}
  {
  }

  operator value_type() const  // NOLINT
  {
    auto* raw = ptr_.get();
    value_type v{};
    update_host(&v, raw, 1, stream_);
    return v;
  }
  auto operator=(T const& other) -> device_reference&
  {
    auto* raw = ptr_.get();
    update_device(raw, &other, 1, stream_);
    return *this;
  }
};

/**
 * @brief A thin wrapper over rmm::device_uvector for implementing the mdarray container policy.
 *
 */
template <typename T>
class device_uvector {
  rmm::device_uvector<T> data_;

 public:
  using value_type = T;
  using size_type  = std::size_t;

  using reference       = device_reference<T>;
  using const_reference = device_reference<T const>;

  using pointer       = value_type*;
  using const_pointer = value_type const*;

  using iterator       = pointer;
  using const_iterator = const_pointer;

 public:
  ~device_uvector()                         = default;
  device_uvector(device_uvector&&) noexcept = default;
  device_uvector(device_uvector const& that) : data_{that.data_, that.data_.stream()} {}

  auto operator=(device_uvector<T> const& that) -> device_uvector<T>&
  {
    data_ = rmm::device_uvector<T>{that.data_, that.data_.stream()};
    return *this;
  }
  auto operator=(device_uvector<T>&& that) noexcept -> device_uvector<T>& = default;

  /**
   * @brief Default ctor is deleted as it doesn't accept stream.
   */
  device_uvector() = delete;
  /**
   * @brief Ctor that accepts a size, stream and an optional mr.
   */
  explicit device_uvector(
    std::size_t size,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
    : data_{size, stream, mr}
  {
  }
  /**
   * @brief Index operator that returns a proxy to the actual data.
   */
  template <typename Index>
  auto operator[](Index i) noexcept -> reference
  {
    return device_reference<T>{thrust::device_ptr<T>{data_.data() + i}, data_.stream()};
  }
  /**
   * @brief Index operator that returns a proxy to the actual data.
   */
  template <typename Index>
  auto operator[](Index i) const noexcept
  {
    return device_reference<T const>{thrust::device_ptr<T const>{data_.data() + i}, data_.stream()};
  }

  [[nodiscard]] auto data() noexcept -> pointer { return data_.data(); }
  [[nodiscard]] auto data() const noexcept -> const_pointer { return data_.data(); }
};

/**
 * @brief A container policy for device mdarray.
 */
template <typename ElementType>
class device_uvector_policy {
  rmm::cuda_stream_view stream_;

 public:
  using element_type   = ElementType;
  using container_type = device_uvector<element_type>;
  // FIXME(jiamingy): allocator type is not supported by rmm::device_uvector
  using pointer         = typename container_type::pointer;
  using const_pointer   = typename container_type::const_pointer;
  using reference       = device_reference<element_type>;
  using const_reference = device_reference<element_type const>;

  using accessor_policy       = std::experimental::default_accessor<element_type>;
  using const_accessor_policy = std::experimental::default_accessor<element_type const>;

 public:
  auto create(size_t n) -> container_type { return container_type(n, stream_); }

  device_uvector_policy() = delete;
  explicit device_uvector_policy(rmm::cuda_stream_view stream) noexcept(
    std::is_nothrow_copy_constructible_v<rmm::cuda_stream_view>)
    : stream_{stream}
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
};

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
  auto create(size_t n) -> container_type { return container_type(n); }

  constexpr host_vector_policy() noexcept(std::is_nothrow_default_constructible_v<ElementType>) =
    default;
  explicit constexpr host_vector_policy(rmm::cuda_stream_view) noexcept(
    std::is_nothrow_default_constructible_v<ElementType>)
    : host_vector_policy()
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
};

/**
 * @brief A mixin to distinguish host and device memory.
 */
template <typename AccessorPolicy, bool is_host>
struct accessor_mixin : public AccessorPolicy {
  using accessor_type = AccessorPolicy;
  using is_host_type  = std::conditional_t<is_host, std::true_type, std::false_type>;
  // make sure the explicit ctor can fall through
  using AccessorPolicy::AccessorPolicy;
  accessor_mixin(AccessorPolicy const& that) : AccessorPolicy{that} {}  // NOLINT
};

template <typename AccessorPolicy>
using host_accessor = accessor_mixin<AccessorPolicy, true>;

template <typename AccessorPolicy>
using device_accessor = accessor_mixin<AccessorPolicy, false>;

namespace stdex = std::experimental;

using vector_extent_t = stdex::extents<stdex::dynamic_extent>;
using matrix_extent_t = stdex::extents<stdex::dynamic_extent, stdex::dynamic_extent>;
}  // namespace raft::detail
