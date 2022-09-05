/*
 * Copyright (2019) Sandia Corporation
 *
 * The source code is licensed under the 3-clause BSD license found in the LICENSE file
 * thirdparty/LICENSES/mdarray.license
 */

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
#include <raft/core/mdspan.hpp>
#include <raft/cudart_utils.h>
#include <raft/detail/span.hpp>  // dynamic_extent

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

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
  explicit device_uvector(std::size_t size, rmm::cuda_stream_view stream) : data_{size, stream} {}
  /**
   * @brief Ctor that accepts a size, stream and a memory resource.
   */
  explicit device_uvector(std::size_t size,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr)
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
  rmm::mr::device_memory_resource* mr_;

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
  auto create(size_t n) -> container_type
  {
    return mr_ ? container_type(n, stream_, mr_) : container_type(n, stream_);
  }

  device_uvector_policy() = delete;
  explicit device_uvector_policy(
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource* mr =
      nullptr) noexcept(std::is_nothrow_copy_constructible_v<rmm::cuda_stream_view>)
    : stream_{stream}, mr_(mr)
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
template <typename AccessorPolicy, bool is_host, bool is_device>
struct accessor_mixin : public AccessorPolicy {
  using accessor_type   = AccessorPolicy;
  using is_host_type    = std::conditional_t<is_host, std::true_type, std::false_type>;
  using is_device_type  = std::conditional_t<is_device, std::true_type, std::false_type>;
  using is_managed_type = std::conditional_t<is_device && is_host, std::true_type, std::false_type>;
  static constexpr bool is_host_accessible    = is_host;
  static constexpr bool is_device_accessible  = is_device;
  static constexpr bool is_managed_accessible = is_device && is_host;
  // make sure the explicit ctor can fall through
  using AccessorPolicy::AccessorPolicy;
  using offset_policy = accessor_mixin;
  accessor_mixin(AccessorPolicy const& that) : AccessorPolicy{that} {}  // NOLINT
};

template <typename AccessorPolicy>
using host_accessor = accessor_mixin<AccessorPolicy, true, false>;

template <typename AccessorPolicy>
using device_accessor = accessor_mixin<AccessorPolicy, false, true>;

template <typename AccessorPolicy>
using managed_accessor = accessor_mixin<AccessorPolicy, true, true>;

namespace stdex = std::experimental;

template <typename IndexType>
using vector_extent = stdex::extents<IndexType, dynamic_extent>;

template <typename IndexType>
using matrix_extent = stdex::extents<IndexType, dynamic_extent, dynamic_extent>;

template <typename IndexType = std::uint32_t>
using scalar_extent = stdex::extents<IndexType, 1>;

template <typename T>
MDSPAN_INLINE_FUNCTION auto native_popc(T v) -> int32_t
{
  int c = 0;
  for (; v != 0; v &= v - 1) {
    c++;
  }
  return c;
}

MDSPAN_INLINE_FUNCTION auto popc(uint32_t v) -> int32_t
{
#if defined(__CUDA_ARCH__)
  return __popc(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcount(v);
#else
  return native_popc(v);
#endif  // compiler
}

MDSPAN_INLINE_FUNCTION auto popc(uint64_t v) -> int32_t
{
#if defined(__CUDA_ARCH__)
  return __popcll(v);
#elif defined(__GNUC__) || defined(__clang__)
  return __builtin_popcountll(v);
#else
  return native_popc(v);
#endif  // compiler
}

template <class T, std::size_t N, std::size_t... Idx>
MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N], std::index_sequence<Idx...>)
{
  return std::make_tuple(arr[Idx]...);
}

template <class T, std::size_t N>
MDSPAN_INLINE_FUNCTION constexpr auto arr_to_tup(T (&arr)[N])
{
  return arr_to_tup(arr, std::make_index_sequence<N>{});
}

// uint division optimization inspired by the CIndexer in cupy.  Division operation is
// slow on both CPU and GPU, especially 64 bit integer.  So here we first try to avoid 64
// bit when the index is smaller, then try to avoid division when it's exp of 2.
template <typename I, typename IndexType, size_t... Extents>
MDSPAN_INLINE_FUNCTION auto unravel_index_impl(I idx, stdex::extents<IndexType, Extents...> shape)
{
  constexpr auto kRank = static_cast<int32_t>(shape.rank());
  std::size_t index[shape.rank()]{0};  // NOLINT
  static_assert(std::is_signed<decltype(kRank)>::value,
                "Don't change the type without changing the for loop.");
  for (int32_t dim = kRank; --dim > 0;) {
    auto s = static_cast<std::remove_const_t<std::remove_reference_t<I>>>(shape.extent(dim));
    if (s & (s - 1)) {
      auto t     = idx / s;
      index[dim] = idx - t * s;
      idx        = t;
    } else {  // exp of 2
      index[dim] = idx & (s - 1);
      idx >>= popc(s - 1);
    }
  }
  index[0] = idx;
  return arr_to_tup(index);
}

/**
 * Ensure all types listed in the parameter pack `Extents` are integral types.
 * Usage:
 *   put it as the last nameless template parameter of a function:
 *     `typename = ensure_integral_extents<Extents...>`
 */
template <typename... Extents>
using ensure_integral_extents = std::enable_if_t<std::conjunction_v<std::is_integral<Extents>...>>;

}  // namespace raft::detail
