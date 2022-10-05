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
#include <raft/matrix/matrix.cuh>

namespace raft::matrix {

/**
 * @brief Slice a matrix (in-place)
 * @param handle: raft handle
 * @param in: input matrix (column-major)
 * @param out: output matrix (column-major)
 * @param x1, y1: coordinate of the top-left point of the wanted area (0-based)
 * @param x2, y2: coordinate of the bottom-right point of the wanted area
 * (1-based)
 * example: Slice the 2nd and 3rd columns of a 4x3 matrix: slice_matrix(M_d, 4,
 * 3, 0, 1, 4, 3);
 */
template <typename m_t, typename idx_t>
void slice(const raft::handle_t& handle,
           raft::device_matrix_view<m_t, idx_t, col_major> in,
           raft::device_matrix_view<m_t, idx_t, col_major> out,
           idx_t x1,
           idx_t y1,
           idx_t x2,
           idx_t y2)
{
  RAFT_EXPECTS(x2 > x1, "x2 must be > x1");
  RAFT_EXPECTS(y2 > y1, "y2 must be > y1");
  RAFT_EXPECTS(x1 >= 0, "x1 must be >= 0");
  RAFT_EXPECTS(x2 <= in.extents(0), "x2 must be <= number of rows in the input matrix")
  RAFT_EXPECTS(y1 >= 0, "y1 must be >= 0");
  RAFT_EXPECTS(y2 <= in.extents(1), "y2 must be <= number of columns in the input matrix");

  detail::sliceMatrix(in.data_handle(),
                      in.extent(0),
                      in.extent(1),
                      out.data_handle(),
                      x1,
                      y1,
                      x2,
                      y2,
                      handle.get_stream());
}
}  // namespace raft::matrix