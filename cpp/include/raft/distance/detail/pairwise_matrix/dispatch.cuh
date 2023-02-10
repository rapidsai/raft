/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "kernel_sm60.cuh"
#include <cstdio>
#include <raft/distance/detail/pairwise_distance_cutlass_base.cuh>
#include <raft/linalg/contractions.cuh>
#include <utility>

namespace raft::distance::detail {

/**
 * @brief: Computes minimal alignment of row starting elements in 2D array
 *
 * The 2D matrix x is assumed to be row-major. This function computes the
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

template <int n>
using vec_len_constant = std::integral_constant<int, n>;

template <typename F>
inline void dispatch(bool row_major, int vec_len, F&& f) {
  if (row_major) {
    switch (vec_len) {
      case 4: f(std::bool_constant<true>(), vec_len_constant<4>()); break;
      case 2: f(std::bool_constant<true>(), vec_len_constant<2>()); break;
      default: f(std::bool_constant<true>(), vec_len_constant<1>()); break;
    }
  } else {
    switch (vec_len) {
      case 4: f(std::bool_constant<false>(), vec_len_constant<4>()); break;
      case 2: f(std::bool_constant<false>(), vec_len_constant<2>()); break;
      default: f(std::bool_constant<false>(), vec_len_constant<1>()); break;
    }
  }
}

template <typename opT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_matrix_dispatch(opT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const DataT* x_norm,
                              const DataT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major)
{
  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  IdxT ldx, ldy, ld_out;
  if (is_row_major) {
    ldx = k, ldy = k, ld_out = n;
  } else {
    // Flip x, y, and m, n.
    std::swap<const DataT*>(x, y);
    std::swap<const DataT*>(x_norm, y_norm);
    std::swap(m, n);
    ldx = m, ldy = n, ld_out = n;
  }

  size_t align_x = alignment_of_2d_array(x, ldx);
  size_t align_y = alignment_of_2d_array(y, ldy);
  size_t byte_alignment = min(align_x, align_y);

  // Since alignment is in bytes, it could be smaller than sizeof(DataT).
  // Handle this (unlikely) case here.
  RAFT_EXPECTS(sizeof(DataT) <= byte_alignment, "Input matrix must be aligned to size of elements.");

  // Compute number of elements that can be loaded in one instruction
  // without causing misalignent errors.
  int vec_len_aligned = (byte_alignment % sizeof(DataT) == 0) ? byte_alignment / sizeof(DataT) : 1;

  dispatch(
    is_row_major,
    vec_len_aligned,
    [&](auto row_major, auto vec_len_aligned) {
      // row_major and vec_len are std::integral_constants of type bool and int
      // respectively.

      // To keep compile times in check, we only specialize on veclen > 1 when
      // the inner loop is relatively cheap (< 5 flops).
      constexpr int vec_len_op = distance_op.expensive_inner_loop ? 1 : vec_len_aligned();

      // Prevent double, vec_len=4 combination (this is not supported)
      constexpr int vec_len = std::min(vec_len_op, static_cast<int>(16 / sizeof(DataT)));

      typedef typename raft::linalg::Policy4x4<DataT, vec_len>::Policy RowPolicy;
      typedef typename raft::linalg::Policy4x4<DataT, vec_len>::ColPolicy ColPolicy;
      typedef typename std::conditional<row_major(), RowPolicy, ColPolicy>::type Policy;

      // Create compile-time template parameter
      using KP_T = kernel_params_T<DataT, AccT, OutT, IdxT, Policy, opT, FinOpT, row_major()>;

      return pairwise_matrix<KP_T>(
        distance_op, fin_op, x, y, x_norm, y_norm, m, n, k, ldx, ldy, ld_out, out, stream);
    });
}

template <typename opT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_matrix_cutlass_dispatch(opT cutlass_op,
                                      IdxT m,
                                      IdxT n,
                                      IdxT k,
                                      const DataT* x,
                                      const DataT* y,
                                      const DataT* x_norm,
                                      const DataT* y_norm,
                                      OutT* out,
                                      FinOpT fin_op,
                                      cudaStream_t stream,
                                      bool is_row_major)
{
  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  IdxT ldx, ldy, ld_out;
  if (is_row_major) {
    ldx = k, ldy = k, ld_out = n;
  } else {
    std::swap<const DataT*>(x, y);
    std::swap<const DataT*>(x_norm, y_norm);
    std::swap(m, n);
    ldx = m, ldy = n, ld_out = n;
  }

  size_t align_x = alignment_of_2d_array(x, ldx);
  size_t align_y = alignment_of_2d_array(y, ldy);
  size_t byte_alignment = min(align_x, align_y);

  // Since alignment is in bytes, it could be smaller than sizeof(DataT).
  // Handle this (unlikely) case here.
  RAFT_EXPECTS(sizeof(DataT) <= byte_alignment, "Input matrix must be aligned to size of elements.");

  // Compute number of elements that can be loaded in one instruction
  // without causing misalignent errors.
  int vec_len_aligned = (byte_alignment % sizeof(DataT) == 0) ? byte_alignment / sizeof(DataT) : 1;

  dispatch(
    is_row_major,
    vec_len_aligned,
    [&](auto row_major, auto vec_len_aligned) {
      // row_major and vec_len are std::integral_constants of type bool and int
      // respectively.

      // Prevent double, vec_len=4 combination (this is not supported)
      constexpr int vec_len = std::min(vec_len_aligned, static_cast<int>(16 / sizeof(DataT)));

      cutlassDistanceKernel<DataT, AccT, OutT, IdxT, vec_len, FinOpT, opT, row_major()>(
        x, y, x_norm, y_norm, m, n, k, ldx, ldy, ld_out, out, fin_op, cutlass_op, stream);
  });
}

};  // namespace raft::distance::detail
