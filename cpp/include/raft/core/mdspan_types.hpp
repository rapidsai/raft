/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda/std/mdspan>

namespace raft {

using cuda::std::dynamic_extent;
using cuda::std::extents;

/**
 * @defgroup mdspan_layout C- and F-contiguous mdspan layouts
 * @{
 */
using cuda::std::layout_right;
using layout_c_contiguous = layout_right;
using row_major           = layout_right;

using cuda::std::layout_left;
using layout_f_contiguous = layout_left;
using col_major           = layout_left;
/** @} */

template <typename IndexType>
using vector_extent = cuda::std::extents<IndexType, dynamic_extent>;

template <typename IndexType>
using matrix_extent = cuda::std::extents<IndexType, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using scalar_extent = cuda::std::extents<IndexType, 1>;

/**
 * @brief Strided layout for non-contiguous memory.
 */
using cuda::std::layout_stride;

template <typename IndexType>
using extent_1d = vector_extent<IndexType>;

template <typename IndexType>
using extent_2d = matrix_extent<IndexType>;

template <typename IndexType>
using extent_3d = cuda::std::extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_4d =
  cuda::std::extents<IndexType, dynamic_extent, dynamic_extent, dynamic_extent, dynamic_extent>;

template <typename IndexType>
using extent_5d = cuda::std::extents<IndexType,
                                     dynamic_extent,
                                     dynamic_extent,
                                     dynamic_extent,
                                     dynamic_extent,
                                     dynamic_extent>;

}  // namespace raft
