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
 * @brief In-place operation. Shifts rows or columns to shift_direction by k, and fills the empty
 * values with "val" Example 1) if we have a row-major 3x4 matrix in_out = [[1,2,3,4], [5,6,7,8],
 * [9,10,11,12]], val=100, k=2, shift_direction = ShiftDirection::TOWARDS_END and shift_type =
 * ShiftType::COL, then we end up with [[100,100,1,2], [100,100,5,6], [100,100,9,10]]. Example 2) if
 * we have a row-major 3x4 matrix in_out = [[1,2,3,4], [5,6,7,8], [9,10,11,12]], val=100, k=1,
 * shift_direction = ShiftDirection::TOWARDS_ZERO and shift_type = ShiftType::ROW, then we end up
 * with [[5,6,7,8], [9,10,11,12], [100,100,100,100]]
 * @param[in] handle: raft handle
 * @param[in out] in_out: input matrix of size (n_rows, n_cols)
 * @param[in] val: value to fill in the first k columns (same for all rows)
 * @param[in] k: shift size
 * @param[in] shift_direction: ShiftDirection::TOWARDS_ZERO shifts towards the 0th row/col
 * direction, and ShiftDirection::TOWARDS_END shifts towards the (nrow-1)th row/col direction
 * @param[in] shift_type: ShiftType::ROW shifts rows and ShiftType::COL shift columns
 */
template <typename ValueT, typename IdxT>
void shift(raft::resources const& handle,
           raft::device_matrix_view<ValueT, IdxT, row_major> in_out,
           ValueT val,
           size_t k,
           ShiftDirection shift_direction = ShiftDirection::TOWARDS_END,
           ShiftType shift_type           = ShiftType::COL)
{
  if (shift_type == ShiftType::COL) {
    RAFT_EXPECTS(static_cast<size_t>(in_out.extent(1)) > k,
                 "Shift size k should be smaller than the number of columns in matrix.");
  } else {
    RAFT_EXPECTS(static_cast<size_t>(in_out.extent(0)) > k,
                 "Shift size k should be smaller than the number of rows in matrix.");
  }
  detail::shift(handle, in_out, val, k, shift_direction, shift_type);
}

/**
 * @brief col_shift: in-place shifts all columns by k columns to the right and replaces the first
 * n_rows x k part of the in_out matrix with the "values" matrix
 * @param[in] handle: raft handle
 * @param[in out] in_out: input matrix of size (n_rows, n_cols)
 * @param[in] values: value matrix of size (n_rows x k) to fill in the first k columns
 * @param[in] shift_direction: ShiftDirection::TOWARDS_ZERO shifts towards the 0th row/col
 * direction, and ShiftDirection::TOWARDS_END shifts towards the (nrow-1)th row/col direction
 * @param[in] shift_type: ShiftType::ROW shifts rows and ShiftType::COL shift columns
 */
template <typename ValueT, typename IdxT>
void shift(raft::resources const& handle,
           raft::device_matrix_view<ValueT, IdxT, row_major> in_out,
           raft::device_matrix_view<const ValueT, IdxT> values,
           ShiftDirection shift_direction = ShiftDirection::TOWARDS_END,
           ShiftType shift_type           = ShiftType::COL)
{
  if (shift_type == ShiftType::COL) {
    RAFT_EXPECTS(in_out.extent(0) == values.extent(0),
                 "in_out matrix and the values matrix should haver the same number of rows when "
                 "using shift_type=ShiftType::COL");
    RAFT_EXPECTS(in_out.extent(1) > values.extent(1),
                 "number of columns in in_out should be > number of columns in values when using "
                 "shift_type=ShiftType::COL");
  } else {
    RAFT_EXPECTS(in_out.extent(1) == values.extent(1),
                 "in_out matrix and the values matrix should haver the same number of cols when "
                 "using shift_type=ShiftType::ROW");
    RAFT_EXPECTS(in_out.extent(0) > values.extent(0),
                 "number of rows in in_out should be > number of rows in values when using "
                 "shift_type=ShiftType::ROW");
  }

  detail::shift(handle, in_out, values, shift_direction, shift_type);
}

/**
 * @brief col_shift: in-place shifts all columns by k columns to the right and fills the first k
 * columns with its row id
 * @param[in] handle: raft handle
 * @param[in out] in_out: input matrix of size (n_rows, n_cols)
 * @param[in] k: shift size
 * @param[in] shift_direction: ShiftDirection::TOWARDS_ZERO shifts towards the 0th row/col
 * direction, and ShiftDirection::TOWARDS_END shifts towards the (nrow-1)th row/col direction
 * @param[in] shift_type: ShiftType::ROW shifts rows and ShiftType::COL shift columns
 */
template <typename math_t, typename matrix_idx_t>
void shift_self(raft::resources const& handle,
                raft::device_matrix_view<math_t, matrix_idx_t, row_major> in_out,
                size_t k,
                ShiftDirection shift_direction = ShiftDirection::TOWARDS_END,
                ShiftType shift_type           = ShiftType::COL)
{
  if (shift_type == ShiftType::COL) {
    RAFT_EXPECTS(static_cast<size_t>(in_out.extent(1)) > k,
                 "Shift size k should be smaller than the number of columns in matrix.");
  } else {
    RAFT_EXPECTS(static_cast<size_t>(in_out.extent(0)) > k,
                 "Shift size k should be smaller than the number of rows in matrix.");
  }
  detail::shift_self(handle, in_out, k, shift_direction, shift_type);
}

}  // namespace raft::matrix
