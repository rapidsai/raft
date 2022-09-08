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

#include <raft/core/detail/host_device_accessor.hpp>

namespace raft {

template <typename AccessorPolicy>
using host_accessor = detail::host_device_accessor<AccessorPolicy, true, false>;

/**
 * @brief std::experimental::mdspan with host tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = std::experimental::default_accessor<ElementType>>
using host_mdspan = mdspan<ElementType, Extents, LayoutPolicy, host_accessor<AccessorPolicy>>;

namespace detail {

template <typename T, bool B>
struct is_host_accessible_mdspan : std::false_type {
};
template <typename T>
struct is_host_accessible_mdspan<T, true>
  : std::bool_constant<T::accessor_type::is_host_accessible> {
};

/**
 * @\brief Boolean to determine if template type T is either raft::host_mdspan or a derived type
 */
template <typename T>
using is_host_accessible_mdspan_t = is_host_accessible_mdspan<T, is_mdspan_v<T>>;

}  // namespace detail

/**
 * @\brief Boolean to determine if variadic template types Tn are either raft::host_mdspan or a
 * derived type
 */
template <typename... Tn>
inline constexpr bool is_host_accessible_mdspan_v =
  std::conjunction_v<detail::is_host_accessible_mdspan_t<Tn>...>;

template <typename... Tn>
using enable_if_host_mdspan = std::enable_if_t<is_host_accessible_mdspan_v<Tn...>>;

/**
 * @brief Shorthand for 0-dim host mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using host_scalar_view = host_mdspan<ElementType, scalar_extent<IndexType>>;

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
  return host_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, n};
}
}  // end namespace raft