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
#include <raft/matrix/matrix.cuh>
#include <raft/matrix/detail/matrix.cuh>

namespace raft::matrix {

/**
 * @brief Initialize a diagonal matrix with a vector
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols
 */
template <typename m_t, typename idx_t = int>
void initialize_diagonal(
        const raft::handle_t &handle,
        raft::device_vector_view<m_t> vec,
        raft::device_matrix_view<m_t, idx_t, col_major> matrix) {
    detail::initializeDiagonalMatrix(vec.data_handle(),
                                     matrix.data_handle(),
                                     matrix.extent(0),
                                     matrix.extent(1),
                                     handle.get_stream());
}

/**
 * @brief Take reciprocal of elements on diagonal of square matrix (in-place)
 * @param in: square input matrix with size len x len
 */
template <typename m_t, typename idx_t = int>
void invert_diagonal(const raft::handle_t &handle,
                     raft::device_matrix_view<m_t, idx_t, col_major> in)
{
    RAFT_EXPECTS(in.extent(0) == in.extent(1), "Matrix must be square.");
    detail::getDiagonalInverseMatrix(in.data_handle(), in.extent(0), handle.get_stream());
}
}
