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

#include <raft/thirdparty/mdspan/include/experimental/mdspan>

namespace raft {

using std::experimental::dynamic_extent;
using std::experimental::extents;

/**
 * @defgroup mdspan_layout C- and F-contiguous mdspan layouts
 * @{
 */
using std::experimental::layout_right;
using layout_c_contiguous = layout_right;
using row_major           = layout_right;

using std::experimental::layout_left;
using layout_f_contiguous = layout_left;
using col_major           = layout_left;
/** @} */

template <typename IndexType>
using vector_extent = std::experimental::extents<IndexType, dynamic_extent>;

template <typename IndexType>
using matrix_extent = std::experimental::extents<IndexType, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using scalar_extent = std::experimental::extents<IndexType, 1>;

/**
 * @brief Strided layout for non-contiguous memory.
 */
using std::experimental::layout_stride;

template <typename IndexType>
using extent_1d = vector_extent<IndexType>;

template <typename IndexType>
using extent_2d = matrix_extent<IndexType>;

template <typename IndexType>
using extent_3d =
  std::experimental::extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_4d = std::experimental::
  extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_5d = std::experimental::extents<IndexType,
                                             dynamic_extent,
                                             dynamic_extent,
                                             dynamic_extent,
                                             dynamic_extent,
                                             dynamic_extent>;

}  // namespace raft
