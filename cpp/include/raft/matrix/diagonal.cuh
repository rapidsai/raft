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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/detail/matrix.cuh>
#include <raft/matrix/init.cuh>
#include <raft/util/input_validation.hpp>

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
void set_diagonal(raft::resources const& handle,
                  raft::device_vector_view<const m_t, idx_t> vec,
                  raft::device_matrix_view<m_t, idx_t, layout> matrix)
{
  RAFT_EXPECTS(vec.extent(0) == std::min(matrix.extent(0), matrix.extent(1)),
               "Diagonal vector must be min(matrix.n_rows, matrix.n_cols)");
  constexpr auto is_row_major = std::is_same_v<layout, layout_c_contiguous>;

  detail::initializeDiagonalMatrix(vec.data_handle(),
                                   matrix.data_handle(),
                                   matrix.extent(0),
                                   matrix.extent(1),
                                   is_row_major,
                                   resource::get_cuda_stream(handle));
}

/**
 * @brief Initialize a diagonal matrix with a vector
 * @param handle: raft handle
 * @param[in] matrix: matrix of size n_rows x n_cols
 * @param[out] vec: vector of length k = min(n_rows, n_cols)
 */
template <typename m_t, typename idx_t, typename layout>
void get_diagonal(raft::resources const& handle,
                  raft::device_matrix_view<const m_t, idx_t, layout> matrix,
                  raft::device_vector_view<m_t, idx_t> vec)
{
  RAFT_EXPECTS(vec.extent(0) == std::min(matrix.extent(0), matrix.extent(1)),
               "Diagonal vector must be min(matrix.n_rows, matrix.n_cols)");
  constexpr auto is_row_major = std::is_same_v<layout, layout_c_contiguous>;
  detail::getDiagonalMatrix(vec.data_handle(),
                            matrix.data_handle(),
                            matrix.extent(0),
                            matrix.extent(1),
                            is_row_major,
                            resource::get_cuda_stream(handle));
}

/**
 * @brief Take reciprocal of elements on diagonal of square matrix (in-place)
 * @param handle raft handle
 * @param[inout] inout: square input matrix with size len x len
 */
template <typename m_t, typename idx_t, typename layout>
void invert_diagonal(raft::resources const& handle,
                     raft::device_matrix_view<m_t, idx_t, layout> inout)
{
  // TODO: Use get_diagonal for this to support rectangular
  RAFT_EXPECTS(inout.extent(0) == inout.extent(1), "Matrix must be square.");
  detail::getDiagonalInverseMatrix(
    inout.data_handle(), inout.extent(0), resource::get_cuda_stream(handle));
}

/**
 * @brief create an identity matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam idx_t indexing type used for the output
 * @tparam layout_t layout of the matrix data (must be row or col major)
 * @param[in] handle: raft handle
 * @param[out] out: output matrix
 */
template <typename math_t, typename idx_t, typename layout_t>
void eye(const raft::resources& handle, raft::device_matrix_view<math_t, idx_t, layout_t> out)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");

  auto diag = raft::make_device_vector<math_t, idx_t>(handle, min(out.extent(0), out.extent(1)));
  RAFT_CUDA_TRY(cudaMemsetAsync(
    out.data_handle(), 0, out.size() * sizeof(math_t), resource::get_cuda_stream(handle)));
  raft::matrix::fill(handle, diag.view(), math_t(1));
  set_diagonal(handle, raft::make_const_mdspan(diag.view()), out);
}

/** @} */  // end of group matrix_diagonal

}  // namespace raft::matrix
