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

#include <stddef.h>

#include <raft/core/handle.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/detail/mdarray.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft {
/**
 * @brief Dimensions extents for raft::host_mdspan or raft::device_mdspan
 */
template <typename IndexType, size_t... ExtentsPack>
using extents = std::experimental::extents<IndexType, ExtentsPack...>;

/**
 * @defgroup C-Contiguous layout for mdarray and mdspan. Implies row-major and contiguous memory.
 * @{
 */
using detail::stdex::layout_right;
using layout_c_contiguous = layout_right;
using row_major           = layout_right;
/** @} */

/**
 * @defgroup F-Contiguous layout for mdarray and mdspan. Implies column-major and contiguous memory.
 * @{
 */
using detail::stdex::layout_left;
using layout_f_contiguous = layout_left;
using col_major           = layout_left;
/** @} */

/**
 * @brief Strided layout for non-contiguous memory.
 */
using detail::stdex::layout_stride;

/**
 * @defgroup Common mdarray/mdspan extent types. The rank is known at compile time, each dimension
 * is known at run time (dynamic_extent in each dimension).
 * @{
 */
using detail::matrix_extent;
using detail::scalar_extent;
using detail::vector_extent;

template <typename IndexType>
using extent_1d = vector_extent<IndexType>;

template <typename IndexType>
using extent_2d = matrix_extent<IndexType>;

template <typename IndexType>
using extent_3d = detail::stdex::extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_4d =
  detail::stdex::extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_5d = detail::stdex::extents<IndexType,
                                         dynamic_extent,
                                         dynamic_extent,
                                         dynamic_extent,
                                         dynamic_extent,
                                         dynamic_extent>;
/** @} */

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = detail::stdex::default_accessor<ElementType>>
using mdspan = detail::stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>;

namespace detail {
/**
 * @\brief Template checks and helpers to determine if type T is an std::mdspan
 *         or a derived type
 */

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void __takes_an_mdspan_ptr(mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>*);

template <typename T, typename = void>
struct is_mdspan : std::false_type {
};
template <typename T>
struct is_mdspan<T, std::void_t<decltype(__takes_an_mdspan_ptr(std::declval<T*>()))>>
  : std::true_type {
};

template <typename T>
using is_mdspan_t = is_mdspan<std::remove_const_t<T>>;

template <typename T>
inline constexpr bool is_mdspan_v = is_mdspan_t<T>::value;
}  // namespace detail

/**
 * @\brief Boolean to determine if variadic template types Tn are either
 *          raft::host_mdspan/raft::device_mdspan or their derived types
 */
template <typename... Tn>
inline constexpr bool is_mdspan_v = std::conjunction_v<detail::is_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_mdspan = std::enable_if_t<is_mdspan_v<Tn...>>;

/**
 * @brief stdex::mdspan with device tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = detail::stdex::default_accessor<ElementType>>
using device_mdspan =
  mdspan<ElementType, Extents, LayoutPolicy, detail::device_accessor<AccessorPolicy>>;

/**
 * @brief stdex::mdspan with host tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = detail::stdex::default_accessor<ElementType>>
using host_mdspan =
  mdspan<ElementType, Extents, LayoutPolicy, detail::host_accessor<AccessorPolicy>>;

template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = detail::stdex::default_accessor<ElementType>>
using managed_mdspan =
  mdspan<ElementType, Extents, LayoutPolicy, detail::managed_accessor<AccessorPolicy>>;

namespace detail {
template <typename T, bool B>
struct is_device_mdspan : std::false_type {
};
template <typename T>
struct is_device_mdspan<T, true> : std::bool_constant<T::accessor_type::is_device_accessible> {
};

/**
 * @\brief Boolean to determine if template type T is either raft::device_mdspan or a derived type
 */
template <typename T>
using is_device_mdspan_t = is_device_mdspan<T, is_mdspan_v<T>>;

template <typename T, bool B>
struct is_host_mdspan : std::false_type {
};
template <typename T>
struct is_host_mdspan<T, true> : std::bool_constant<T::accessor_type::is_host_accessible> {
};

/**
 * @\brief Boolean to determine if template type T is either raft::host_mdspan or a derived type
 */
template <typename T>
using is_host_mdspan_t = is_host_mdspan<T, is_mdspan_v<T>>;

template <typename T, bool B>
struct is_managed_mdspan : std::false_type {
};
template <typename T>
struct is_managed_mdspan<T, true> : std::bool_constant<T::accessor_type::is_managed_accessible> {
};

/**
 * @\brief Boolean to determine if template type T is either raft::managed_mdspan or a derived type
 */
template <typename T>
using is_managed_mdspan_t = is_managed_mdspan<T, is_mdspan_v<T>>;
}  // namespace detail

/**
 * @\brief Boolean to determine if variadic template types Tn are either raft::device_mdspan or a
 * derived type
 */
template <typename... Tn>
inline constexpr bool is_device_mdspan_v = std::conjunction_v<detail::is_device_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_device_mdspan = std::enable_if_t<is_device_mdspan_v<Tn...>>;

/**
 * @\brief Boolean to determine if variadic template types Tn are either raft::host_mdspan or a
 * derived type
 */
template <typename... Tn>
inline constexpr bool is_host_mdspan_v = std::conjunction_v<detail::is_host_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_host_mdspan = std::enable_if_t<is_host_mdspan_v<Tn...>>;

/**
 * @\brief Boolean to determine if variadic template types Tn are either raft::managed_mdspan or a
 * derived type
 */
template <typename... Tn>
inline constexpr bool is_managed_mdspan_v = std::conjunction_v<detail::is_managed_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_managed_mdspan = std::enable_if_t<is_managed_mdspan_v<Tn...>>;

/**
 * @brief Interface to implement an owning multi-dimensional array
 *
 * raft::array_interace is an interface to owning container types for mdspan.
 * Check implementation of raft::mdarray which implements raft::array_interface
 * using Curiously Recurring Template Pattern.
 * This interface calls into method `view()` whose implementation is provided by
 * the implementing class. `view()` must return an object of type raft::host_mdspan
 * or raft::device_mdspan or any types derived from the them.
 */
template <typename Base>
class array_interface {
  /**
   * @brief Get a mdspan that can be passed down to CUDA kernels.
   */
  auto view() noexcept { return static_cast<Base*>(this)->view(); }
  /**
   * @brief Get a mdspan that can be passed down to CUDA kernels.
   */
  auto view() const noexcept { return static_cast<Base*>(this)->view(); }
};

namespace detail {
template <typename T, typename = void>
struct is_array_interface : std::false_type {
};
template <typename T>
struct is_array_interface<T, std::void_t<decltype(std::declval<T>().view())>>
  : std::bool_constant<is_mdspan_v<decltype(std::declval<T>().view())>> {
};

template <typename T>
using is_array_interface_t = is_array_interface<std::remove_const_t<T>>;

/**
 * @\brief Boolean to determine if template type T is raft::array_interface or derived type
 *         or any type that has a member function `view()` that returns either
 *         raft::host_mdspan or raft::device_mdspan
 */
template <typename T>
inline constexpr bool is_array_interface_v = is_array_interface<std::remove_const_t<T>>::value;
}  // namespace detail

template <typename...>
struct is_array_interface : std::true_type {
};
template <typename T1>
struct is_array_interface<T1> : detail::is_array_interface_t<T1> {
};
template <typename T1, typename... Tn>
struct is_array_interface<T1, Tn...> : std::conditional_t<detail::is_array_interface_v<T1>,
                                                          is_array_interface<Tn...>,
                                                          std::false_type> {
};
/**
 * @\brief Boolean to determine if variadic template types Tn are raft::array_interface
 *         or derived type or any type that has a member function `view()` that returns either
 *         raft::host_mdspan or raft::device_mdspan
 */
template <typename... Tn>
inline constexpr bool is_array_interface_v = is_array_interface<Tn...>::value;

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
template <typename ElementType, typename Extents, typename LayoutPolicy, typename ContainerPolicy>
class mdarray
  : public array_interface<mdarray<ElementType, Extents, LayoutPolicy, ContainerPolicy>> {
  static_assert(!std::is_const<ElementType>::value,
                "Element type for container must not be const.");

 public:
  using extents_type = Extents;
  using layout_type  = LayoutPolicy;
  using mapping_type = typename layout_type::template mapping<extents_type>;
  using element_type = ElementType;

  using value_type      = std::remove_cv_t<element_type>;
  using index_type      = typename extents_type::index_type;
  using difference_type = std::ptrdiff_t;
  using rank_type       = typename extents_type::rank_type;

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
    std::conditional_t<container_policy_type::is_host_accessible,
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

  [[nodiscard]] constexpr auto size() const noexcept -> std::size_t { return this->view().size(); }

  [[nodiscard]] auto data_handle() noexcept -> pointer { return c_.data(); }
  [[nodiscard]] constexpr auto data_handle() const noexcept -> const_pointer { return c_.data(); }

  /**
   * @brief Indexing operator, use it sparingly since it triggers a device<->host copy.
   */
  template <typename... IndexType>
  auto operator()(IndexType&&... indices)
    -> std::enable_if_t<sizeof...(IndexType) == extents_type::rank() &&
                          (std::is_convertible_v<IndexType, index_type> && ...) &&
                          std::is_constructible_v<extents_type, IndexType...>,
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
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto rank() noexcept -> rank_type
  {
    return extents_type::rank();
  }
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto rank_dynamic() noexcept -> rank_type
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
  [[nodiscard]] MDSPAN_INLINE_FUNCTION constexpr auto is_exhaustive() const noexcept -> bool
  {
    return map_.is_exhaustive();
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
  [[nodiscard]] MDSPAN_INLINE_FUNCTION static constexpr auto is_always_exhaustive() noexcept -> bool
  {
    return mapping_type::is_always_exhaustive();
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

/**
 * @brief mdarray with host container policy
 * @tparam ElementType the data type of the elements
 * @tparam Extents defines the shape
 * @tparam LayoutPolicy policy for indexing strides and layout ordering
 * @tparam ContainerPolicy storage and accessor policy
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = detail::host_vector_policy<ElementType>>
using host_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, detail::host_accessor<ContainerPolicy>>;

/**
 * @brief mdarray with device container policy
 * @tparam ElementType the data type of the elements
 * @tparam Extents defines the shape
 * @tparam LayoutPolicy policy for indexing strides and layout ordering
 * @tparam ContainerPolicy storage and accessor policy
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy    = layout_c_contiguous,
          typename ContainerPolicy = detail::device_uvector_policy<ElementType>>
using device_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, detail::device_accessor<ContainerPolicy>>;

/**
 * @brief Shorthand for 0-dim host mdarray (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using host_scalar = host_mdarray<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 0-dim host mdarray (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using device_scalar = device_mdarray<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 1-dim host mdarray.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_vector = host_mdarray<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for 1-dim device mdarray.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_vector = device_mdarray<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous host matrix.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_matrix = host_mdarray<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous device matrix.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_matrix = device_mdarray<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for 0-dim host mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using host_scalar_view = host_mdspan<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 0-dim host mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using device_scalar_view = device_mdspan<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 1-dim host mdspan.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_vector_view = host_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for 1-dim device mdspan.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_vector_view = device_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous host matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using host_matrix_view = host_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous device matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using device_matrix_view = device_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Create a raft::mdspan
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @tparam is_host_accessible whether the data is accessible on host
 * @tparam is_device_accessible whether the data is accessible on device
 * @param ptr Pointer to the data
 * @param exts dimensionality of the array (series of integers)
 * @return raft::mdspan
 */
template <typename ElementType,
          typename IndexType        = std::uint32_t,
          typename LayoutPolicy     = layout_c_contiguous,
          bool is_host_accessible   = false,
          bool is_device_accessible = true,
          size_t... Extents>
auto make_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
{
  using accessor_type = detail::accessor_mixin<std::experimental::default_accessor<ElementType>,
                                               is_host_accessible,
                                               is_device_accessible>;

  return mdspan<ElementType, decltype(exts), LayoutPolicy, accessor_type>{ptr, exts};
}

/**
 * @brief Create a raft::managed_mdspan
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param ptr Pointer to the data
 * @param exts dimensionality of the array (series of integers)
 * @return raft::managed_mdspan
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_managed_mdspan(ElementType* ptr, extents<IndexType, Extents...> exts)
{
  return make_mdspan<ElementType, IndexType, LayoutPolicy, true, true>(ptr, exts);
}

/**
 * @brief Create a 0-dim (scalar) mdspan instance for host value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on device to wrap
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto make_host_scalar_view(ElementType* ptr)
{
  scalar_extent<IndexType> extents;
  return host_scalar_view<ElementType, IndexType>{ptr, extents};
}

/**
 * @brief Create a 0-dim (scalar) mdspan instance for device value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on device to wrap
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto make_device_scalar_view(ElementType* ptr)
{
  scalar_extent<IndexType> extents;
  return device_scalar_view<ElementType, IndexType>{ptr, extents};
}

/**
 * @brief Create a 2-dim c-contiguous mdspan instance for host pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on host to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_host_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  matrix_extent<IndexType> extents{n_rows, n_cols};
  return host_matrix_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
}
/**
 * @brief Create a 2-dim c-contiguous mdspan instance for device pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on device to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_device_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  matrix_extent<IndexType> extents{n_rows, n_cols};
  return device_matrix_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 1-dim mdspan instance for host pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on host to wrap
 * @param[in] n number of elements in pointer
 * @return raft::host_vector_view
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_host_vector_view(ElementType* ptr, IndexType n)
{
  vector_extent<IndexType> extents{n};
  return host_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 1-dim mdspan instance for device pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on device to wrap
 * @param[in] n number of elements in pointer
 * @return raft::device_vector_view
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_device_vector_view(ElementType* ptr, IndexType n)
{
  vector_extent<IndexType> extents{n};
  return device_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a host mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param exts dimensionality of the array (series of integers)
 * @return raft::host_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_host_mdarray(extents<IndexType, Extents...> exts)
{
  using mdarray_t = host_mdarray<ElementType, decltype(exts), LayoutPolicy>;

  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy;

  return mdarray_t{layout, policy};
}

/**
 * @brief Create a device mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param handle raft::handle_t
 * @param exts dimensionality of the array (series of integers)
 * @return raft::device_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_device_mdarray(const raft::handle_t& handle, extents<IndexType, Extents...> exts)
{
  using mdarray_t = device_mdarray<ElementType, decltype(exts), LayoutPolicy>;

  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy{handle.get_stream()};

  return mdarray_t{layout, policy};
}

/**
 * @brief Create a device mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param handle raft::handle_t
 * @param mr rmm memory resource used for allocating the memory for the array
 * @param exts dimensionality of the array (series of integers)
 * @return raft::device_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_device_mdarray(const raft::handle_t& handle,
                         rmm::mr::device_memory_resource* mr,
                         extents<IndexType, Extents...> exts)
{
  using mdarray_t = device_mdarray<ElementType, decltype(exts), LayoutPolicy>;

  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy{handle.get_stream(), mr};

  return mdarray_t{layout, policy};
}

/**
 * @brief Create raft::extents to specify dimensionality
 *
 * @tparam IndexType The type of each dimension of the extents
 * @tparam Extents Dimensions (a series of integers)
 * @param exts The desired dimensions
 * @return raft::extents
 */
template <typename IndexType,
          typename... Extents,
          typename = detail::ensure_integral_extents<Extents...>>
auto make_extents(Extents... exts)
{
  return extents<IndexType, ((void)exts, dynamic_extent)...>{exts...};
}

/**
 * @brief Create a 2-dim c-contiguous host mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] n_rows number or rows in matrix
 * @param[in] n_cols number of columns in matrix
 * @return raft::host_matrix
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_host_matrix(IndexType n_rows, IndexType n_cols)
{
  return make_host_mdarray<ElementType, IndexType, LayoutPolicy>(
    make_extents<IndexType>(n_rows, n_cols));
}

/**
 * @brief Create a 2-dim c-contiguous device mdarray.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] n_rows number or rows in matrix
 * @param[in] n_cols number of columns in matrix
 * @return raft::device_matrix
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_device_matrix(raft::handle_t const& handle, IndexType n_rows, IndexType n_cols)
{
  return make_device_mdarray<ElementType, IndexType, LayoutPolicy>(
    handle.get_stream(), make_extents<IndexType>(n_rows, n_cols));
}

/**
 * @brief Create a host scalar from v.
 *
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 * @param[in] v scalar type to wrap
 * @return raft::host_scalar
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto make_host_scalar(ElementType const& v)
{
  // FIXME(jiamingy): We can optimize this by using std::array as container policy, which
  // requires some more compile time dispatching. This is enabled in the ref impl but
  // hasn't been ported here yet.
  scalar_extent<IndexType> extents;
  using policy_t = typename host_scalar<ElementType>::container_policy_type;
  policy_t policy;
  auto scalar = host_scalar<ElementType>{extents, policy};
  scalar(0)   = v;
  return scalar;
}

/**
 * @brief Create a device scalar from v.
 *
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 * @param[in] handle raft handle for managing expensive cuda resources
 * @param[in] v scalar to wrap on device
 * @return raft::device_scalar
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto make_device_scalar(raft::handle_t const& handle, ElementType const& v)
{
  scalar_extent<IndexType> extents;
  using policy_t = typename device_scalar<ElementType>::container_policy_type;
  policy_t policy{handle.get_stream()};
  auto scalar = device_scalar<ElementType>{extents, policy};
  scalar(0)   = v;
  return scalar;
}

/**
 * @brief Create a 1-dim host mdarray.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] n number of elements in vector
 * @return raft::host_vector
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_host_vector(IndexType n)
{
  return make_host_mdarray<ElementType, IndexType, LayoutPolicy>(make_extents<IndexType>(n));
}

/**
 * @brief Create a 1-dim device mdarray.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] handle raft handle for managing expensive cuda resources
 * @param[in] n number of elements in vector
 * @return raft::device_vector
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_device_vector(raft::handle_t const& handle, IndexType n)
{
  return make_device_mdarray<ElementType, IndexType, LayoutPolicy>(handle.get_stream(),
                                                                   make_extents<IndexType>(n));
}

/**
 * @brief Flatten raft::host_mdspan or raft::device_mdspan into a 1-dim array view
 *
 * @tparam mdspan_type Expected type raft::host_mdspan or raft::device_mdspan
 * @param mds raft::host_mdspan or raft::device_mdspan object
 * @return raft::host_mdspan or raft::device_mdspan with vector_extent
 *         depending on AccessoryPolicy
 */
template <typename mdspan_type, typename = enable_if_mdspan<mdspan_type>>
auto flatten(mdspan_type mds)
{
  RAFT_EXPECTS(mds.is_exhaustive(), "Input must be contiguous.");

  vector_extent<typename mdspan_type::size_type> ext{mds.size()};

  return detail::stdex::mdspan<typename mdspan_type::element_type,
                               decltype(ext),
                               typename mdspan_type::layout_type,
                               typename mdspan_type::accessor_type>(mds.data_handle(), ext);
}

/**
 * @brief Flatten object implementing raft::array_interface into a 1-dim array view
 *
 * @tparam array_interface_type Expected type implementing raft::array_interface
 * @param mda raft::array_interace implementing object
 * @return Either raft::host_mdspan or raft::device_mdspan with vector_extent
 *         depending on the underlying ContainerPolicy
 */
template <typename array_interface_type,
          std::enable_if_t<is_array_interface_v<array_interface_type>>* = nullptr>
auto flatten(const array_interface_type& mda)
{
  return flatten(mda.view());
}

/**
 * @brief Reshape raft::host_mdspan or raft::device_mdspan
 *
 * @tparam mdspan_type Expected type raft::host_mdspan or raft::device_mdspan
 * @tparam IndexType the index type of the extents
 * @tparam Extents raft::extents for dimensions
 * @param mds raft::host_mdspan or raft::device_mdspan object
 * @param new_shape Desired new shape of the input
 * @return raft::host_mdspan or raft::device_mdspan, depending on AccessorPolicy
 */
template <typename mdspan_type,
          typename IndexType = std::uint32_t,
          size_t... Extents,
          typename = enable_if_mdspan<mdspan_type>>
auto reshape(mdspan_type mds, extents<IndexType, Extents...> new_shape)
{
  RAFT_EXPECTS(mds.is_exhaustive(), "Input must be contiguous.");

  size_t new_size = 1;
  for (size_t i = 0; i < new_shape.rank(); ++i) {
    new_size *= new_shape.extent(i);
  }
  RAFT_EXPECTS(new_size == mds.size(), "Cannot reshape array with size mismatch");

  return detail::stdex::mdspan<typename mdspan_type::element_type,
                               decltype(new_shape),
                               typename mdspan_type::layout_type,
                               typename mdspan_type::accessor_type>(mds.data_handle(), new_shape);
}

/**
 * @brief Reshape object implementing raft::array_interface
 *
 * @tparam array_interface_type Expected type implementing raft::array_interface
 * @tparam Extents raft::extents for dimensions
 * @tparam IndexType the index type of the extents
 * @param mda raft::array_interace implementing object
 * @param new_shape Desired new shape of the input
 * @return raft::host_mdspan or raft::device_mdspan, depending on the underlying
 *         ContainerPolicy
 */
template <typename array_interface_type,
          typename IndexType = std::uint32_t,
          size_t... Extents,
          std::enable_if_t<is_array_interface_v<array_interface_type>>* = nullptr>
auto reshape(const array_interface_type& mda, extents<IndexType, Extents...> new_shape)
{
  return reshape(mda.view(), new_shape);
}

/**
 * \brief Turns linear index into coordinate.  Similar to numpy unravel_index.
 *
 * \code
 *   auto m = make_host_matrix<float>(7, 6);
 *   auto m_v = m.view();
 *   auto coord = unravel_index(2, m.extents(), typename decltype(m)::layout_type{});
 *   std::apply(m_v, coord) = 2;
 * \endcode
 *
 * \param idx    The linear index.
 * \param shape  The shape of the array to use.
 * \param layout Must be `layout_c_contiguous` (row-major) in current implementation.
 *
 * \return A std::tuple that represents the coordinate.
 */
template <typename Idx, typename IndexType, typename LayoutPolicy, size_t... Exts>
MDSPAN_INLINE_FUNCTION auto unravel_index(Idx idx,
                                          extents<IndexType, Exts...> shape,
                                          LayoutPolicy const& layout)
{
  static_assert(std::is_same_v<std::remove_cv_t<std::remove_reference_t<decltype(layout)>>,
                               layout_c_contiguous>,
                "Only C layout is supported.");
  static_assert(std::is_integral_v<Idx>, "Index must be integral.");
  auto constexpr kIs64 = sizeof(std::remove_cv_t<std::remove_reference_t<Idx>>) == sizeof(uint64_t);
  if (kIs64 && static_cast<uint64_t>(idx) > std::numeric_limits<uint32_t>::max()) {
    return detail::unravel_index_impl<uint64_t>(static_cast<uint64_t>(idx), shape);
  } else {
    return detail::unravel_index_impl<uint32_t>(static_cast<uint32_t>(idx), shape);
  }
}
}  // namespace raft
