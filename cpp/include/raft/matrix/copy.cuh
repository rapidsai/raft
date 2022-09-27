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

#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/detail/matrix.cuh>

namespace raft::matrix {

/**
 * @brief Copy selected rows of the input matrix into contiguous space.
 *
 * On exit out[i + k*n_rows] = in[indices[i] + k*n_rows],
 * where i = 0..n_rows_indices-1, and k = 0..n_cols-1.
 *
 * @param[in] handle raft handle
 * @param[in] in input matrix
 * @param[out] out output matrix
 * @param[in] indices of the rows to be copied
 */
template <typename m_t, typename idx_array_t>
void copy_rows(const raft::handle_t& handle,
               raft::device_matrix_view<const m_t> in,
               raft::device_matrix_view<m_t> out,
               raft::device_vector_view<idx_array_t> indices)
{
  RAFT_EXPECTS(in.extent(1) == out.extent(1),
               "Input and output matrices must have same number of columns");
  RAFT_EXPECTS(indices.extent(0) == out.extent(0),
               "Number of rows in output matrix must equal number of indices");
  bool in_rowmajor  = raft::is_row_major(in);
  bool out_rowmajor = raft::is_row_major(out);

  RAFT_EXPECTS(in_rowmajor == out_rowmajor,
               "Input and output matrices must have same layout (row- or column-major)");

  detail::copyRows(in.data_handle(),
                   in.extent(0),
                   in.extent(1),
                   out.data_handle(),
                   indices.data_handle(),
                   indices.extent(0),
                   handle.get_stream());
}

/**
 * @brief copy matrix operation for column major matrices.
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix
 */
template <typename m_t, typename matrix_idx_t>
void copy(const raft::handle_t& handle,
          raft::device_matrix_view<const m_t, matrix_idx_t, col_major> in,
          raft::device_matrix_view<m_t, matrix_idx_t, col_major> out)
{
  RAFT_EXPECTS(in.extent(0) == out.extent(0) && in.extent(1) == out.extent(1),
               "Input and output matrix shapes must match.");

  raft::copy_async(
    out.data_handle(), in.data_handle(), in.extent(0) * out.extent(1), handle.get_stream());
}

/**
 * @brief copy matrix operation for column major matrices. First n_rows and
 * n_cols of input matrix "in" is copied to "out" matrix.
 * @param in: input matrix
 * @param in_n_rows: number of rows of input matrix
 * @param out: output matrix
 * @param out_n_rows: number of rows of output matrix
 * @param out_n_cols: number of columns of output matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t>
void trunc_zero_origin(
  const raft::handle_t &handle,
  raft::device_matrix_view<const m_t, idx_t, col_major> in,
  raft::device_matrix_view<m_t, idx_t, col_major> out) {

  RAFT_EXPECTS(out.extent(0) <= in.extent(0) &&
               out.extent(1) <= in.extent(1),
               "Output matrix must have less or equal number of rows and columns");

  detail::truncZeroOrigin<m_t, idx_t>(in.data_handle(), in.extent(0), out.data_handle(), out.extent(0), out.extent(1), handle.get_stream());
}

}  // namespace raft::matrix
