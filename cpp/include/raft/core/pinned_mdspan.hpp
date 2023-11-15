/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <raft/core/mdspan.hpp>
#include <raft/core/memory_type.hpp>

#include <raft/core/host_device_accessor.hpp>

namespace raft {

template <typename AccessorPolicy>
using pinned_accessor = host_device_accessor<AccessorPolicy, memory_type::pinned>;

/**
 * @brief std::experimental::mdspan with pinned tag to avoid accessing incorrect memory location.
 */
template <typename ElementType,
          typename Extents,
          typename LayoutPolicy   = layout_c_contiguous,
          typename AccessorPolicy = std::experimental::default_accessor<ElementType>>
using pinned_mdspan = mdspan<ElementType, Extents, LayoutPolicy, pinned_accessor<AccessorPolicy>>;

/**
 * @brief Shorthand for 0-dim pinned mdspan (scalar).
 * @tparam ElementType the data type of the scalar element
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType, typename IndexType = std::uint32_t>
using pinned_scalar_view = pinned_mdspan<ElementType, scalar_extent<IndexType>>;

/**
 * @brief Shorthand for 1-dim pinned mdspan.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using pinned_vector_view = pinned_mdspan<ElementType, vector_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for c-contiguous pinned matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
using pinned_matrix_view = pinned_mdspan<ElementType, matrix_extent<IndexType>, LayoutPolicy>;

/**
 * @brief Shorthand for 128 byte aligned pinned matrix view.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_right_padded<ElementType>,
          typename              = enable_if_layout_padded<ElementType, LayoutPolicy>>
using pinned_aligned_matrix_view =
  pinned_mdspan<ElementType,
                matrix_extent<IndexType>,
                LayoutPolicy,
                std::experimental::aligned_accessor<ElementType, detail::alignment::value>>;

/**
 * @brief Create a 2-dim 128 byte aligned mdspan instance for pinned pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam LayoutPolicy must be of type layout_{left/right}_padded
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on pinned to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_right_padded<ElementType>>
auto make_pinned_aligned_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  using data_handle_type =
    typename std::experimental::aligned_accessor<ElementType,
                                                 detail::alignment::value>::data_handle_type;

  static_assert(std::is_same<LayoutPolicy, layout_left_padded<ElementType>>::value ||
                std::is_same<LayoutPolicy, layout_right_padded<ElementType>>::value);
  assert(reinterpret_cast<std::uintptr_t>(ptr) ==
         std::experimental::details::alignTo(reinterpret_cast<std::uintptr_t>(ptr),
                                             detail::alignment::value));
  data_handle_type aligned_pointer = ptr;

  matrix_extent<IndexType> extents{n_rows, n_cols};
  return pinned_aligned_matrix_view<ElementType, IndexType, LayoutPolicy>{aligned_pointer, extents};
}

/**
 * @brief Create a 0-dim (scalar) mdspan instance for pinned value.
 *
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr on device to wrap
 */
template <typename ElementType, typename IndexType = std::uint32_t>
auto make_pinned_scalar_view(ElementType* ptr)
{
  scalar_extent<IndexType> extents;
  return pinned_scalar_view<ElementType, IndexType>{ptr, extents};
}

/**
 * @brief Create a 2-dim c-contiguous mdspan instance for pinned pointer. It's
 *        expected that the given layout policy match the layout of the underlying
 *        pointer.
 * @tparam ElementType the data type of the matrix elements
 * @tparam IndexType the index type of the extents
 * @tparam LayoutPolicy policy for strides and layout ordering
 * @param[in] ptr to pinned data to wrap
 * @param[in] n_rows number of rows in pointer
 * @param[in] n_cols number of columns in pointer
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_pinned_matrix_view(ElementType* ptr, IndexType n_rows, IndexType n_cols)
{
  matrix_extent<IndexType> extents{n_rows, n_cols};
  return pinned_matrix_view<ElementType, IndexType, LayoutPolicy>{ptr, extents};
}

/**
 * @brief Create a 1-dim mdspan instance for pinned pointer.
 * @tparam ElementType the data type of the vector elements
 * @tparam IndexType the index type of the extents
 * @param[in] ptr to pinned data to wrap
 * @param[in] n number of elements in pointer
 * @return raft::pinned_vector_view
 */
template <typename ElementType,
          typename IndexType    = std::uint32_t,
          typename LayoutPolicy = layout_c_contiguous>
auto make_pinned_vector_view(ElementType* ptr, IndexType n)
{
  return pinned_vector_view<ElementType, IndexType, LayoutPolicy>{ptr, n};
}
}  // end namespace raft
