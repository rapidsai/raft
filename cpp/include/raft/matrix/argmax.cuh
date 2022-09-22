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
 * @brief Argmax: find the row idx with maximum value for each column
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param out: output vector of size n_cols
 * @param stream: cuda stream
 */
template <typename math_t, typename matrix_idx_t>
void argmax(const raft::handle_t& handle,
            raft::device_matrix_view<const math_t, matrix_idx_t, col_major> in,
            raft::device_vector_view<math_t> out)
{
  RAFT_EXPECTS(out.extent(1) == in.extent(1),
               "Size of output vector must equal number of columns in input matrix.");
  detail::argmax(
    in.data_handle(), in.extent(0), in.extent(1), out.data_handle(), handle.get_stream());
}
}  // namespace raft::matrix
