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

#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/detail/matrix.cuh>

namespace raft::matrix {

/**
 * @defgroup matrix_diagonal Matrix diagonal operations
 * @{
 */

/**
 * @brief Initialize a diagonal matrix with a vector
 * @param[in] handle: raft handle
 * @param[in] vec: vector of length k = min(n_rows, n_cols)
 * @param[out] matrix: matrix of size n_rows x n_cols
 */
template <typename m_t, typename idx_t, typename layout>
void set_diagonal(raft::device_resources const& handle,
                  raft::device_vector_view<const m_t, idx_t> vec,
                  raft::device_matrix_view<m_t, idx_t, layout> matrix)
{
  RAFT_EXPECTS(vec.extent(0) == std::min(matrix.extent(0), matrix.extent(1)),
               "Diagonal vector must be min(matrix.n_rows, matrix.n_cols)");

  detail::initializeDiagonalMatrix(vec.data_handle(),
                                   matrix.data_handle(),
                                   matrix.extent(0),
                                   matrix.extent(1),
                                   handle.get_stream());
}

/**
 * @brief Initialize a diagonal matrix with a vector
 * @param handle: raft handle
 * @param[in] matrix: matrix of size n_rows x n_cols
 * @param[out] vec: vector of length k = min(n_rows, n_cols)
 */
template <typename m_t, typename idx_t, typename layout>
void get_diagonal(raft::device_resources const& handle,
                  raft::device_matrix_view<const m_t, idx_t, layout> matrix,
                  raft::device_vector_view<m_t, idx_t> vec)
{
  RAFT_EXPECTS(vec.extent(0) == std::min(matrix.extent(0), matrix.extent(1)),
               "Diagonal vector must be min(matrix.n_rows, matrix.n_cols)");
  detail::getDiagonalMatrix(vec.data_handle(),
                            matrix.data_handle(),
                            matrix.extent(0),
                            matrix.extent(1),
                            handle.get_stream());
}

/**
 * @brief Take reciprocal of elements on diagonal of square matrix (in-place)
 * @param handle raft handle
 * @param[inout] inout: square input matrix with size len x len
 */
template <typename m_t, typename idx_t, typename layout>
void invert_diagonal(raft::device_resources const& handle,
                     raft::device_matrix_view<m_t, idx_t, layout> inout)
{
  // TODO: Use get_diagonal for this to support rectangular
  RAFT_EXPECTS(inout.extent(0) == inout.extent(1), "Matrix must be square.");
  detail::getDiagonalInverseMatrix(inout.data_handle(), inout.extent(0), handle.get_stream());
}

/** @} */  // end of group matrix_diagonal

}  // namespace raft::matrix
