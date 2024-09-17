/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_container_policy.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/core/resources.hpp>

#include <rmm/resource_ref.hpp>

#include <cstdint>

namespace raft {

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
          typename ContainerPolicy = device_uvector_policy<ElementType>>
using device_mdarray =
  mdarray<ElementType, Extents, LayoutPolicy, device_accessor<ContainerPolicy>>;

/**
 * @brief Shorthand for 0-dim host mdarray (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using device_scalar = device_mdarray<ElementType, scalar_extent<IndexType>>;

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
 * @brief Create a device mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param handle raft::resources
 * @param exts dimensionality of the array (series of integers)
 * @return raft::device_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_device_mdarray(raft::resources const& handle, extents<IndexType, Extents...> exts)
{
  using mdarray_t = device_mdarray<ElementType, decltype(exts), LayoutPolicy>;

  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy{};

  return mdarray_t{handle, layout, policy};
}

/**
 * @brief Create a device mdarray.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param handle raft::resources
 * @param mr rmm memory resource used for allocating the memory for the array
 * @param exts dimensionality of the array (series of integers)
 * @return raft::device_mdarray
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous,
          size_t... Extents>
auto make_device_mdarray(raft::resources const& handle,
                         rmm::device_async_resource_ref mr,
                         extents<IndexType, Extents...> exts)
{
  using mdarray_t = device_mdarray<ElementType, decltype(exts), LayoutPolicy>;

  typename mdarray_t::mapping_type layout{exts};
  typename mdarray_t::container_policy_type policy{mr};

  return mdarray_t{handle, layout, policy};
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
auto make_device_matrix(raft::resources const& handle, IndexType n_rows, IndexType n_cols)
{
  return make_device_mdarray<ElementType, IndexType, LayoutPolicy>(
    handle, make_extents<IndexType>(n_rows, n_cols));
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
auto make_device_scalar(raft::resources const& handle, ElementType const& v)
{
  scalar_extent<IndexType> extents;
  using policy_t = typename device_scalar<ElementType>::container_policy_type;
  policy_t policy{};
  auto scalar = device_scalar<ElementType>{handle, extents, policy};
  scalar(0)   = v;
  return scalar;
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
auto make_device_vector(raft::resources const& handle, IndexType n)
{
  return make_device_mdarray<ElementType, IndexType, LayoutPolicy>(handle,
                                                                   make_extents<IndexType>(n));
}

}  // end namespace raft
