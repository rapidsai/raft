/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/detail/shift.cuh>

namespace raft::matrix {

/**
 * @brief col_shift: in-place shifts all columns by k columns to the right and fills the first k
 * columns in with "val"
 * @param[in] handle: raft handle
 * @param[in out] in_out: input matrix of size (n_rows, n_cols)
 * @param[in] val: value to fill in the first column (same for all rows)
 * @param[in] k: shift size
 */
template <typename math_t, typename matrix_idx_t>
void col_right_shift(raft::resources const& handle,
                     raft::device_matrix_view<math_t, matrix_idx_t, row_major> in_out,
                     math_t val,
                     size_t k)
{
  RAFT_EXPECTS(in_out.extent(1) > k,
               "Shift size k should be smaller than the number of columns in matrix.");
  detail::col_right_shift(handle, in_out, val, k);
}

/**
 * @brief col_shift: in-place shifts all columns by k columns to the right and replaces the first
 * n_rows x k part of the in_out matrix with "values" matrix
 * @param[in] handle: raft handle
 * @param[in out] in_out: input matrix of size (n_rows, n_cols)
 * @param[in] values: value matrix to fill in the first
 */
template <typename math_t, typename matrix_idx_t>
void col_right_shift(raft::resources const& handle,
                     raft::device_matrix_view<math_t, matrix_idx_t, row_major> in_out,
                     raft::device_matrix_view<const math_t, matrix_idx_t> values)
{
  RAFT_EXPECTS(in_out.extent(0) == values.extent(0),
               "in_out matrix and the values matrix should haver the same number of rows");
  RAFT_EXPECTS(in_out.extent(1) > values.extent(1),
               "number of columns in in_out should be > number of columns in values");
  detail::col_right_shift(handle, in_out, values);
}

/**
 * @brief col_shift: in-place shifts all columns by k columns to the right and fills the first k
 * columns with its row id
 * @param[in] handle: raft handle
 * @param[in out] in_out: input matrix of size (n_rows, n_cols)
 * @param[in] k: shift size
 */
template <typename math_t, typename matrix_idx_t>
void col_right_shift_self(raft::resources const& handle,
                          raft::device_matrix_view<math_t, matrix_idx_t, row_major> in_out,
                          size_t k)
{
  RAFT_EXPECTS(in_out.extent(1) > k,
               "Shift size k should be smaller than the number of columns in matrix.");
  detail::col_right_shift_self(handle, in_out, k);
}

}  // namespace raft::matrix
