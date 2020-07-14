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
#include <experimental/mdspan>
#include <raft/cudart_utils.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/device_ptr.h>

namespace raft {
namespace detail {
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
}  // namespace detail

namespace stdex = std::experimental;

/**
 * @brief stdex::mdspan with device tag to avoid accessing incorrect memory location.
 */
template <class ElementType,
          class Extents,
          class LayoutPolicy   = stdex::layout_right,
          class AccessorPolicy = stdex::default_accessor<ElementType>>
using device_mdspan =
  stdex::mdspan<ElementType, Extents, LayoutPolicy, detail::device_accessor<AccessorPolicy>>;

/**
 * @brief stdex::mdspan with host tag to avoid accessing incorrect memory location.
 */
template <class ElementType,
          class Extents,
          class LayoutPolicy   = stdex::layout_right,
          class AccessorPolicy = stdex::default_accessor<ElementType>>
using host_mdspan =
  stdex::mdspan<ElementType, Extents, LayoutPolicy, detail::host_accessor<AccessorPolicy>>;

/**
 * @brief Modified from the c++ mdarray proposal
 *
 *   https://isocpp.org/files/papers/D1684R0.html
 *
 * mdarray is a container type for mdspan with similar template arguments.  However there
 * are some inconsistencies in between them.  We have made some modificiations to fit our
 * needs, which are listed below.
 *
 * - Layout policy is different, the mdarray in raft uses `stdex::extent` directly just
 *   like `mdspan`, while the `mdarray` in the reference implementation uses varidic
 *   template.
 *
 * - Most of the constructors from the reference implementation is removed to make sure
 *   CUDA stream is honorred.
 *
 * - unique_size is not implemented, which is still working in progress in the proposal
 *
 * - For container policy, we adopt the alternative approach documented in the proposal
 *   [sec 2.4.3], which requires an additional make_accessor method for it to be used in
 *   mdspan.  The container policy reference implementation has multiple `access` methods
 *   that accommodate needs for both mdarray and mdspan.  This is more difficult for us
 *   since the policy might contain states that are unwanted inside a CUDA kernel.  Also,
 *   on host we return a proxy to the actual value as `device_ref` so different access
 *   methods will have different return type, which is less desirable.
 *
 * - For the above reasons, copying from other mdarray with different policy type is also
 *   removed.
 */
template <class ElementType, class Extents, class LayoutPolicy, class ContainerPolicy>
class mdarray {
  static_assert(!std::is_const<ElementType>::value,
                "Element type for container must not be const.");

 public:
  using extents_type = Extents;
  using layout_type  = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;

  using value_type      = std::remove_cv_t<element_type>;
  using index_type      = std::size_t;
  using difference_type = std::ptrdiff_t;
  // Naming: ref impl: container_policy_type, proposal: container_policy
  using container_policy_type = ContainerPolicy;
  using container_type        = typename container_policy_type::container_type;

  using pointer         = typename container_policy_type::pointer;
  using const_pointer   = typename container_policy_type::const_pointer;
  using reference       = typename container_policy_type::reference;
  using const_reference = typename container_policy_type::const_reference;

 private:
  template <typename E,
            typename ViewAccessorPolicy =
              std::conditional_t<std::is_const_v<E>,
                                 typename container_policy_type::const_accessor_policy,
                                 typename container_policy_type::accessor_policy>>
  using view_type_impl =
    std::conditional_t<container_policy_type::is_host_type::value,
                       host_mdspan<E, extents_type, layout_type, ViewAccessorPolicy>,
                       device_mdspan<E, extents_type, layout_type, ViewAccessorPolicy>>;

 public:
  /**
   * \brief the mdspan type returned by view method.
   */
  using view_type       = view_type_impl<element_type>;
  using const_view_type = view_type_impl<element_type const>;

 public:
  constexpr mdarray() noexcept(std::is_nothrow_default_constructible_v<container_type>)
    : cp_{rmm::cuda_stream_default}, c_{cp_.create(0)} {};
  constexpr mdarray(mdarray const&) noexcept(std::is_nothrow_copy_constructible_v<container_type>) =
    default;
  constexpr mdarray(mdarray&&) noexcept(std::is_nothrow_move_constructible<container_type>::value) =
    default;

  constexpr auto operator                                               =(mdarray const&) noexcept(
    std::is_nothrow_copy_assignable<container_type>::value) -> mdarray& = default;
  constexpr auto operator                                               =(mdarray&&) noexcept(
    std::is_nothrow_move_assignable<container_type>::value) -> mdarray& = default;

  ~mdarray() noexcept(std::is_nothrow_destructible<container_type>::value) = default;

#ifndef RAFT_MDARRAY_CTOR_CONSTEXPR
#if !(__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 2)
// 11.0:
// Error: Internal Compiler Error (codegen): "there was an error in verifying the lgenfe output!"
//
// 11.2:
// Call parameter type does not match function signature!
// i8** null
// i8*  %call14 = call i32 null(void (i8*)* null, i8* null, i8** null), !dbg !1060
// <unnamed>: parse Invalid record (Producer: 'LLVM7.0.1' Reader: 'LLVM 7.0.1')
#define RAFT_MDARRAY_CTOR_CONSTEXPR constexpr
#else
#define RAFT_MDARRAY_CTOR_CONSTEXPR
#endif  // !(__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 2)
#endif  // RAFT_MDARRAY_CTOR_CONSTEXPR

  /**
   * @brief The only constructor that can create storage, this is to make sure CUDA stream is being
   * used.
   */
  RAFT_MDARRAY_CTOR_CONSTEXPR mdarray(mapping_type const& m, container_policy_type const& cp)
    : cp_(cp), map_(m), c_(cp_.create(map_.required_span_size()))
  {
  }
  RAFT_MDARRAY_CTOR_CONSTEXPR mdarray(mapping_type const& m, container_policy_type& cp)
    : cp_(cp), map_(m), c_(cp_.create(map_.required_span_size()))
  {
  }

#undef RAFT_MDARRAY_CTOR_CONSTEXPR

  /**
   * @brief Get a mdspan that can be passed down to CUDA kernels.
   */
  auto view() noexcept { return view_type(c_.data(), map_, cp_.make_accessor_policy()); }
  /**
   * @brief Get a mdspan that can be passed down to CUDA kernels.
   */
  auto view() const noexcept
  {
    return const_view_type(c_.data(), map_, cp_.make_accessor_policy());
  }

  [[nodiscard]] constexpr auto size() const noexcept -> index_type { return this->view().size(); }

  [[nodiscard]] auto data() noexcept -> pointer { return c_.data(); }
  [[nodiscard]] constexpr auto data() const noexcept -> const_pointer { return c_.data(); }

  /**
   * @brief Indexing operator, use it sparingly since it triggers a device<->host copy.
   */
  template <typename... IndexType>
  auto operator()(IndexType&&... indices)
    -> std::enable_if_t<sizeof...(IndexType) == extents_type::rank() &&
                          (std::is_convertible_v<IndexType, index_type> && ...) &&
                          std::is_constructible_v<extents_type, IndexType...> &&
                          std::is_constructible_v<mapping_type, extents_type>,
                        /* device policy is not default constructible due to requirement for CUDA
                           stream. */
                        /* std::is_default_constructible_v<container_policy_type> */
                        reference>
  {
    return cp_.access(c_, map_(std::forward<IndexType>(indices)...));
  }

  /**
   * @brief Indexing operator, use it sparingly since it triggers a device<->host copy.
   */
  template <typename... IndexType>
  auto operator()(IndexType&&... indices) const
    -> std::enable_if_t<sizeof...(IndexType) == extents_type::rank() &&
                          (std::is_convertible_v<IndexType, index_type> && ...) &&
                          std::is_constructible_v<extents_type, IndexType...> &&
                          std::is_constructible<mapping_type, extents_type>::value,
                        /* device policy is not default constructible due to requirement for CUDA
                           stream. */
                        /* std::is_default_constructible_v<container_policy_type> */
                        const_reference>
  {
    return cp_.access(c_, map_(std::forward<IndexType>(indices)...));
  }

  // basic_mdarray observers of the domain multidimensional index space (also in basic_mdspan)
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto rank() noexcept -> index_type
  {
    return extents_type::rank();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto rank_dynamic() noexcept -> index_type
  {
    return extents_type::rank_dynamic();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto static_extent(size_t r) noexcept
    -> index_type
  {
    return extents_type::static_extent(r);
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto extents() const noexcept -> extents_type
  {
    return map_.extents();
  }
  /**
   * @brief the extent of rank r
   */
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto extent(size_t r) const noexcept -> index_type
  {
    return map_.extents().extent(r);
  }
  // mapping
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto mapping() const noexcept -> mapping_type
  {
    return map_;
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto is_unique() const noexcept -> bool
  {
    return map_.is_unique();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto is_contiguous() const noexcept -> bool
  {
    return map_.is_contiguous();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto is_strided() const noexcept -> bool
  {
    return map_.is_strided();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto stride(size_t r) const -> index_type
  {
    return map_.stride(r);
  }

  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto is_always_unique() noexcept -> bool
  {
    return mapping_type::is_always_unique();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto is_always_contiguous() noexcept -> bool
  {
    return mapping_type::is_always_contiguous();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto is_always_strided() noexcept -> bool
  {
    return mapping_type::is_always_strided();
  }

 private:
  template <typename, typename, typename, typename>
  friend class mdarray;

 private:
  container_policy_type cp_;
  mapping_type map_;
  container_type c_;
};

template <class ElementType,
          class Extents,
          class LayoutPolicy    = stdex::layout_right,
          class ContainerPolicy = detail::host_vector_policy<ElementType>>
using host_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, detail::host_accessor<ContainerPolicy>>;

template <class ElementType,
          class Extents,
          class LayoutPolicy    = stdex::layout_right,
          class ContainerPolicy = detail::device_uvector_policy<ElementType>>
using device_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, detail::device_accessor<ContainerPolicy>>;
}  // namespace raft
