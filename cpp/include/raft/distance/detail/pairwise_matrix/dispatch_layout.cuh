/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>                              // RAFT_EXPECTS
#include <raft/distance/detail/pairwise_matrix/params.cuh>  // pairwise_matrix_params

#include <algorithm>    // std::min
#include <cstdint>      // size_t
#include <type_traits>  // std::integral_constant
namespace raft::distance::detail {

/**
 * @brief: Computes minimal common alignment of the rows in a 2D array in bytes
 *
 * The 2D matrix `x` is assumed to be row-major. This function computes the
 * minimal alignment in bytes of the first elements of each row.
 * Output can be 16, 8, 4, 2, 1.
 *
 * @param x        Base pointer of row-major input matrix
 * @param stride   Stride in number of element between consecutive rows.
 */
template <typename DataT>
size_t alignment_of_2d_array(const DataT* x, size_t stride)
{
  auto base           = reinterpret_cast<uintptr_t>(x);
  size_t stride_bytes = sizeof(DataT) * stride;

  for (int align = 16; align >= 0; align /= 2) {
    bool base_aligned   = base % align == 0;
    bool stride_aligned = stride_bytes % align == 0;
    if (base_aligned && stride_aligned) { return align; }
  }
  return 1;
}

/**
 * @brief: Computes the vec_len parameter kernel policy parameter
 *
 * @param params  Kernel parameters
 */
template <typename IdxT, typename DataT, typename OutT, typename FinOpT>
int determine_vec_len(pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params)
{
  size_t align_x        = alignment_of_2d_array(params.x, params.ldx);
  size_t align_y        = alignment_of_2d_array(params.y, params.ldy);
  size_t byte_alignment = min(align_x, align_y);

  // Since alignment is in bytes, it could be smaller than sizeof(DataT).
  // Handle this (unlikely) case here.
  RAFT_EXPECTS(sizeof(DataT) <= byte_alignment,
               "Input matrix must be aligned to size of elements.");

  // Compute number of elements that can be loaded in one instruction
  // without causing misalignent errors.
  int vec_len_aligned = (byte_alignment % sizeof(DataT) == 0) ? byte_alignment / sizeof(DataT) : 1;

  // In the future, pairwise_matrix might support `int8_t` input. In that case,
  // byte_alignment / sizeof(DataT) might exceed 4. We maximize at 4 here, to
  // prevent adding more cases in dispatch_layout below (which are expensive to
  // compile).
  vec_len_aligned = std::min(vec_len_aligned, 4);

  return vec_len_aligned;
}

template <int n>
using vec_len_constant = std::integral_constant<int, n>;

/**
 * @brief: Converts run-time arguments to compile-time arguments
 *
 * Converts run-time arguments row_major and vec_len to compile-time arguments
 * and dispatches a lambda f with these compile-time arguments.
 *
 * This is equivalent to copying and pasting the lambda function `f` in each of
 * the switch case statements.
 *
 * @tparam F         Type of lambda f.
 * @param row_major  Boolean indicating whether input arrays have row-major layout.
 * @param vec_len    Integer value 1, 2, or 4 specifying the Veclen template parameter of
 *                   the KernelPolicy.
 * @param f          Lambda that takes two std::integral_constant parameters representing
 *                   row_major and vec_len.
 */
template <typename F>
auto dispatch_layout(bool row_major, int vec_len, F&& f)
{
  if (row_major) {
    switch (vec_len) {
      case 4: return f(std::true_type(), vec_len_constant<4>());
      case 2: return f(std::true_type(), vec_len_constant<2>());
      default: return f(std::true_type(), vec_len_constant<1>());
    }
  } else {
    switch (vec_len) {
      case 4: return f(std::false_type(), vec_len_constant<4>());
      case 2: return f(std::false_type(), vec_len_constant<2>());
      default: return f(std::false_type(), vec_len_constant<1>());
    }
  }
}

};  // namespace raft::distance::detail
