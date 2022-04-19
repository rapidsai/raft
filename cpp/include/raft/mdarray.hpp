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
#include <experimental/mdspan>
#include <raft/detail/mdarray.hpp>
#include <raft/handle.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace raft {

template <size_t... ExtentsPack>
using extents = std::experimental::extents<ExtentsPack...>;

/**
 * @\brief C-Contiguous layout for mdarray and mdspan. Implies row-major and contiguous memory.
 */
using layout_c_contiguous = detail::stdex::layout_right;

/**
 * @\brief F-Contiguous layout for mdarray and mdspan. Implies column-major and contiguous memory.
 */
using layout_f_contiguous = detail::stdex::layout_left;

template <typename T>
struct __is_mdspan : std::false_type {
};

template <typename... Args>
struct __is_mdspan<detail::stdex::mdspan<Args...>> : std::true_type {
};

template <typename ElementType, typename Extents, typename LayoutPolicy, typename AccessorPolicy>
void __takes_an_mdspan_ptr(
  detail::stdex::mdspan<ElementType, Extents, LayoutPolicy, AccessorPolicy>*);

template <typename T, typename = void>
struct __is_derived_mdspan : std::false_type {
};

template <typename T>
struct __is_derived_mdspan<T, std::void_t<decltype(__takes_an_mdspan_ptr(std::declval<T*>()))>>
  : std::true_type {
};

template <typename T>
inline constexpr bool is_mdspan_v = std::disjunction_v<__is_mdspan<std::remove_const_t<T>>,
                                                       __is_derived_mdspan<std::remove_const_t<T>>>;

template <typename T, typename U = void>
using is_mdspan_t = std::enable_if_t<is_mdspan_v<T>, U>;

/**
 * @brief stdex::mdspan with device tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = detail::stdex::default_accessor<ElementType>>
using device_mdspan = detail::stdex::
  mdspan<ElementType, Extents, LayoutPolicy, detail::device_accessor<AccessorPolicy>>;

template <typename T, bool B>
struct __is_device_mdspan : std::false_type {
};

template <typename T>
struct __is_device_mdspan<T, true> : std::bool_constant<not T::accessor_type::is_host_type::value> {
};

template <typename T>
inline constexpr bool is_device_mdspan_v = __is_device_mdspan<T, is_mdspan_v<T>>::value;

template <typename T, typename U = void>
using is_device_mdspan_t = std::enable_if_t<is_device_mdspan_v<T>, U>;

/**
 * @brief stdex::mdspan with host tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = detail::stdex::default_accessor<ElementType>>
using host_mdspan =
  detail::stdex::mdspan<ElementType, Extents, LayoutPolicy, detail::host_accessor<AccessorPolicy>>;

template <typename T, bool B>
struct __is_host_mdspan : std::false_type {
};

template <typename T>
struct __is_host_mdspan<T, true> : T::accessor_type::is_host_type {
};

template <typename T>
inline constexpr bool is_host_mdspan_v = __is_host_mdspan<T, is_mdspan_v<T>>::value;

template <typename T, typename U = void>
using is_host_mdspan_t = std::enable_if_t<is_host_mdspan_v<T>, U>;

// template <typename T>
// inline constexpr bool is_host_or_device_mdspan_v = std::conjunction_v<__is_device_mdspan<T>,
// __is_host_mdspan<T>>;

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

  /**
   * @brief Get an implicitly constructed mdspan that can be passed down to CUDA kernels.
   */
  operator view_type() noexcept { return view_type(c_.data(), map_, cp_.make_accessor_policy()); }
  /**
   * @brief Get an implicitly constructed mdspan that can be passed down to CUDA kernels.
   */
  operator const_view_type() const noexcept
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

template <typename T>
struct __is_mdarray : std::false_type {
};

template <typename... Args>
struct __is_mdarray<mdarray<Args...>> : std::true_type {
};

template <typename T>
inline constexpr bool is_mdarray_v = __is_mdarray<std::remove_const_t<T>>::value;

template <typename T, typename U = void>
using is_mdarray_t = std::enable_if_t<is_mdspan_v<T>, U>;

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
 */
template <typename ElementType>
using host_scalar = host_mdarray<ElementType, detail::scalar_extent>;

/**
 * @brief Shorthand for 0-dim host mdarray (scalar).
 * @tparam ElementType the data type of the scalar element
 */
template <typename ElementType>
using device_scalar = device_mdarray<ElementType, detail::scalar_extent>;

/**
 * @brief Shorthand for 1-dim host mdarray.
 * @tparam ElementType the data type of the vector elements
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
using host_vector = host_mdarray<ElementType, detail::vector_extent, LayoutPolicy>;

/**
 * @brief Shorthand for 1-dim device mdarray.
 * @tparam ElementType the data type of the vector elements
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
using device_vector = device_mdarray<ElementType, detail::vector_extent, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous host matrix.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
using host_matrix = host_mdarray<ElementType, detail::matrix_extent, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous device matrix.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
using device_matrix = device_mdarray<ElementType, detail::matrix_extent, LayoutPolicy>;

/**
 * @brief Shorthand for 0-dim host mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 */
template <typename ElementType>
using host_scalar_view = host_mdspan<ElementType, detail::scalar_extent>;

/**
 * @brief Shorthand for 0-dim host mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 */
template <typename ElementType>
using device_scalar_view = device_mdspan<ElementType, detail::scalar_extent>;

/**
 * @brief Shorthand for 1-dim host mdspan.
 * @tparam ElementType the data type of the vector elements
 */
template <typename ElementType, typename LayoutType = layout_c_contiguous>
using host_vector_view = host_mdspan<ElementType, detail::vector_extent, LayoutType>;

/**
 * @brief Shorthand for 1-dim device mdspan.
 * @tparam ElementType the data type of the vector elements
 */
template <typename ElementType, typename LayoutType = layout_c_contiguous>
using device_vector_view = device_mdspan<ElementType, detail::vector_extent, LayoutType>;

/**
 * @brief Shorthand for c-contiguous host matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 *
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
using host_matrix_view = host_mdspan<ElementType, detail::matrix_extent, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous device matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 *
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
using device_matrix_view = device_mdspan<ElementType, detail::matrix_extent, LayoutPolicy>;

/**
 * @brief Create a 0-dim (scalar) mdspan instance for host value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @param[in] ptr on device to wrap
 */
template <typename ElementType>
auto make_host_scalar_view(ElementType* ptr)
{
  detail::scalar_extent extents;
  return host_scalar_view<ElementType>{ptr, extents};
}

/**
 * @brief Create a 0-dim (scalar) mdspan instance for device value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @param[in] ptr on device to wrap
 */
template <typename ElementType>
auto make_device_scalar_view(ElementType* ptr)
{
  detail::scalar_extent extents;
  return device_scalar_view<ElementType>{ptr, extents};
}

/**
 * @brief Create a 2-dim c-contiguous mdspan instance for host pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on host to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_host_matrix_view(ElementType* ptr, size_t n_rows, size_t n_cols)
{
  detail::matrix_extent extents{n_rows, n_cols};
  return host_matrix_view<ElementType, LayoutPolicy>{ptr, extents};
}
/**
 * @brief Create a 2-dim c-contiguous mdspan instance for device pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr on device to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_device_matrix_view(ElementType* ptr, size_t n_rows, size_t n_cols)
{
  detail::matrix_extent extents{n_rows, n_cols};
  return device_matrix_view<ElementType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 1-dim mdspan instance for host pointer.
 * @tparam ElementType the data type of the vector elements
 * @param[in] ptr on host to wrap
 * @param[in] n number of elements in pointer
 * @return raft::host_vector_view
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_host_vector_view(ElementType* ptr, size_t n)
{
  detail::vector_extent extents{n};
  return host_vector_view<ElementType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 1-dim mdspan instance for device pointer.
 * @tparam ElementType the data type of the vector elements
 * @param[in] ptr on device to wrap
 * @param[in] n number of elements in pointer
 * @return raft::device_vector_view
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_device_vector_view(ElementType* ptr, size_t n)
{
  detail::vector_extent extents{n};
  return device_vector_view<ElementType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 2-dim c-contiguous host mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] n_rows number or rows in matrix
 * @param[in] n_cols number of columns in matrix
 * @return raft::host_matrix
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_host_matrix(size_t n_rows, size_t n_cols)
{
  detail::matrix_extent extents{n_rows, n_cols};
  using policy_t = typename host_matrix<ElementType>::container_policy_type;
  policy_t policy;
  return host_matrix<ElementType, LayoutPolicy>{extents, policy};
}

/**
 * @brief Create a 2-dim c-contiguous device mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] n_rows number or rows in matrix
 * @param[in] n_cols number of columns in matrix
 * @param[in] stream cuda stream for ordering events
 * @return raft::device_matrix
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_device_matrix(size_t n_rows, size_t n_cols, rmm::cuda_stream_view stream)
{
  detail::matrix_extent extents{n_rows, n_cols};
  using policy_t = typename device_matrix<ElementType>::container_policy_type;
  policy_t policy{stream};
  return device_matrix<ElementType, LayoutPolicy>{extents, policy};
}

/**
 * @brief Create a 2-dim c-contiguous device mdarray.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] handle raft handle for managing expensive resources
 * @param[in] n_rows number or rows in matrix
 * @param[in] n_cols number of columns in matrix
 * @return raft::device_matrix
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_device_matrix(raft::handle_t const& handle, size_t n_rows, size_t n_cols)
{
  return make_device_matrix<ElementType, LayoutPolicy>(n_rows, n_cols, handle.get_stream());
}

/**
 * @brief Create a host scalar from v.
 *
 * @tparam ElementType the data type of the scalar element
 * @param[in] v scalar type to wrap
 * @return raft::host_scalar
 */
template <typename ElementType>
auto make_host_scalar(ElementType const& v)
{
  // FIXME(jiamingy): We can optimize this by using std::array as container policy, which
  // requires some more compile time dispatching. This is enabled in the ref impl but
  // hasn't been ported here yet.
  detail::scalar_extent extents;
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
 * @param[in] v scalar type to wrap on device
 * @param[in] stream the cuda stream for ordering events
 * @return raft::device_scalar
 */
template <typename ElementType>
auto make_device_scalar(ElementType const& v, rmm::cuda_stream_view stream)
{
  detail::scalar_extent extents;
  using policy_t = typename device_scalar<ElementType>::container_policy_type;
  policy_t policy{stream};
  auto scalar = device_scalar<ElementType>{extents, policy};
  scalar(0)   = v;
  return scalar;
}

/**
 * @brief Create a device scalar from v.
 *
 * @tparam ElementType the data type of the scalar element
 * @param[in] handle raft handle for managing expensive cuda resources
 * @param[in] v scalar to wrap on device
 * @return raft::device_scalar
 */
template <typename ElementType>
auto make_device_scalar(raft::handle_t const& handle, ElementType const& v)
{
  return make_device_scalar<ElementType>(v, handle.get_stream());
}

/**
 * @brief Create a 1-dim host mdarray.
 * @tparam ElementType the data type of the vector elements
 * @param[in] n number of elements in vector
 * @return raft::host_vector
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_host_vector(size_t n)
{
  detail::vector_extent extents{n};
  using policy_t = typename host_vector<ElementType>::container_policy_type;
  policy_t policy;
  return host_vector<ElementType, LayoutPolicy>{extents, policy};
}

/**
 * @brief Create a 1-dim device mdarray.
 * @tparam ElementType the data type of the vector elements
 * @param[in] n number of elements in vector
 * @param[in] stream the cuda stream for ordering events
 * @return raft::device_vector
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_device_vector(size_t n, rmm::cuda_stream_view stream)
{
  detail::vector_extent extents{n};
  using policy_t = typename device_vector<ElementType>::container_policy_type;
  policy_t policy{stream};
  return device_vector<ElementType, LayoutPolicy>{extents, policy};
}

/**
 * @brief Create a 1-dim device mdarray.
 * @tparam ElementType the data type of the vector elements
 * @param[in] handle raft handle for managing expensive cuda resources
 * @param[in] n number of elements in vector
 * @return raft::device_vector
 */
template <typename ElementType, typename LayoutPolicy = layout_c_contiguous>
auto make_device_vector(raft::handle_t const& handle, size_t n)
{
  return make_device_vector<ElementType, LayoutPolicy>(n, handle.get_stream());
}

template <typename host_mdspan_type,
          std::enable_if_t<is_host_mdspan_v<host_mdspan_type>>* = nullptr>
auto flatten(host_mdspan_type h_mds)
{
  RAFT_EXPECTS(h_mds.is_contiguous(), "Input must be contiguous.");

  return make_host_vector_view<typename host_mdspan_type::element_type,
                               typename host_mdspan_type::layout_type>(h_mds.data(), h_mds.size());
}

template <typename device_mdspan_type,
          std::enable_if_t<is_device_mdspan_v<device_mdspan_type>>* = nullptr>
auto flatten(device_mdspan_type d_mds)
{
  RAFT_EXPECTS(d_mds.is_contiguous(), "Input must be contiguous.");
  return make_device_vector_view<typename device_mdspan_type::element_type,
                                 typename device_mdspan_type::layout_type>(d_mds.data(),
                                                                           d_mds.size());
}

template <typename mdarray_type, std::enable_if_t<is_mdarray_v<mdarray_type>>* = nullptr>
auto flatten(const mdarray_type& mda)
{
  return flatten(mda.view());
}

template <typename ElementType, typename LayoutType>
constexpr auto flatten(host_vector_view<ElementType, LayoutType> h_vv)
{
  return h_vv;
}

template <typename ElementType, typename LayoutType>
auto flatten(const host_vector<ElementType, LayoutType>& h_v)
{
  return flatten(h_v.view());
}

template <typename ElementType>
constexpr auto flatten(host_scalar_view<ElementType> h_sv)
{
  return h_sv;
}

template <typename ElementType>
auto flatten(const host_scalar<ElementType>& h_s)
{
  return flatten(h_s.view());
}

template <typename ElementType, typename LayoutType>
constexpr auto flatten(device_vector_view<ElementType, LayoutType> d_vv)
{
  return d_vv;
}

template <typename ElementType, typename LayoutType>
auto flatten(const device_vector<ElementType, LayoutType>& d_v)
{
  return flatten(d_v.view());
}

template <typename ElementType>
constexpr auto flatten(device_scalar_view<ElementType> d_sv)
{
  return d_sv;
}

template <typename ElementType>
auto flatten(const device_scalar<ElementType>& d_s)
{
  return flatten(d_s.view());
}

template <typename host_mdspan_type,
          size_t... Extents,
          std::enable_if_t<is_host_mdspan_v<host_mdspan_type>>* = nullptr>
auto reshape(host_mdspan_type h_mds, std::experimental::extents<Extents...> new_shape)
{
  RAFT_EXPECTS(h_mds.is_contiguous(), "Input must be contiguous.");

  // if (new_shape == h_mds.extents()) {
  //   return h_mds;
  // } else if (new_shape.rank() == 1) {
  //   auto new_size = new_shape.extent(0);
  //   RAFT_EXPECTS(new_size <= h_mds.size(),
  //                "Cannot reshape array of size %ul into %ul",
  //                h_mds.size(),
  //                new_size());

  //   if (new_size == 1) {
  //     return make_host_scalar_view<typename host_mdspan_type::element_type>(h_mds.data());
  //   } else {
  //     return make_host_vector_view<typename host_mdspan_type::element_type,
  //                                  typename host_mdspan_type::layout_type>(h_mds.data(),
  //                                  new_size);
  //   }
  // } else if (new_shape.rank() == 2) {
  //   auto new_size = new_shape.extent(0) * new_shape.extent(1);
  //   RAFT_EXPECTS(new_size == h_mds.size(), "Cannot reshape array with size mismatch");

  //   return make_host_matrix_view<typename host_mdspan_type::element_type,
  //                                typename host_mdspan_type::layout_type>(
  //     h_mds.data(), new_shape.extent(0), new_shape.extent(1));
  // } else {
  size_t new_size = 1;
  for (size_t i = 0; i < new_shape.rank(); ++i) {
    new_size *= new_shape.extent(i);
  }
  RAFT_EXPECTS(new_size <= h_mds.size(), "Cannot reshape array with size mismatch");

  return detail::stdex::mdspan<typename host_mdspan_type::element_type,
                               decltype(new_shape),
                               typename host_mdspan_type::layout_type,
                               typename host_mdspan_type::accessor_type>(h_mds.data(), new_shape);
  // }
}

// template <typename device_mdspan_type,
//           std::enable_if_t<is_device_mdspan_v<device_mdspan_type>>* = nullptr>
// auto reshape(device_mdspan_type d_mds)
// {
//   RAFT_EXPECTS(d_mds.is_contiguous(), "Input must be contiguous.");
//   return make_device_vector_view<typename device_mdspan_type::element_type,
//                                  typename device_mdspan_type::layout_type>(d_mds.data(),
//                                                                            d_mds.size());
// }

template <typename mdarray_type,
          size_t... Extents,
          std::enable_if_t<is_mdarray_v<mdarray_type>>* = nullptr>
auto reshape(const mdarray_type& mda, extents<Extents...> new_shape)
{
  return reshape(mda.view(), new_shape);
}

}  // namespace raft
