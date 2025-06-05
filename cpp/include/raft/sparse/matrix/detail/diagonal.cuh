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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/matrix/init.cuh>
#include <raft/util/input_validation.hpp>

namespace raft::sparse::matrix::detail {

/**
 * @brief Get the diagonal vector from a CSR matrix
 *
 * This function extracts the diagonal elements from a CSR matrix and stores them in a vector.
 * The diagonal elements are the elements where the row index and column index are the same.
 *
 *
 * @tparam T The data type of the matrix elements
 * @tparam IndexType The data type of the indices
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  csr_matrix_view View of the input CSR matrix from which to extract the diagonal
 * @param[out] diagonal        View of the output vector where diagonal elements will be stored
 *
 */
template <typename T, typename IndexType>
void diagonal(raft::resources const& res,
              raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix_view,
              raft::device_vector_view<T, IndexType> diagonal)
{
  auto structure = csr_matrix_view.structure_view();
  auto n_rows    = structure.get_n_rows();

  auto values      = csr_matrix_view.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();
  auto diag_ptr    = diagonal.data_handle();

  auto policy = raft::resource::get_thrust_policy(res);

  thrust::for_each(policy,
                   thrust::counting_iterator<IndexType>(0),
                   thrust::counting_iterator<IndexType>(n_rows),
                   [values, col_indices, row_offsets, diag_ptr] __device__(IndexType row) {
                     // For each row, find diagonal element (if it exists)
                     for (auto j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
                       if (col_indices[j] == row) {
                         diag_ptr[row] = values[j];
                         break;
                       }
                     }
                   });
}

/**
 * @brief Scale a CSR matrix by its diagonal elements
 *
 * This function scales each element of the CSR matrix by the corresponding diagonal element.
 * The diagonal elements are assumed to be stored in the diagonal vector.
 *
 *
 * @tparam T The data type of the matrix elements
 * @tparam IndexType The data type of the indices
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  csr_matrix_view View of the input CSR matrix to scale
 * @param[in]  diagonal        View of the input vector containing diagonal elements
 *
 *
 */
template <typename T, typename IndexType>
void scale_by_diagonal_symmetric(
  raft::resources const& res,
  const raft::device_vector_view<T, IndexType> diagonal,
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix)
{
  auto structure = csr_matrix.structure_view();
  auto nnz       = structure.get_nnz();

  auto values      = csr_matrix.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();
  auto diag_ptr    = diagonal.data_handle();

  auto policy = raft::resource::get_thrust_policy(res);

  // For each row
  thrust::for_each(policy,
                   thrust::counting_iterator<IndexType>(0),
                   thrust::counting_iterator<IndexType>(structure.get_n_rows()),
                   [values, col_indices, row_offsets, diag_ptr] __device__(IndexType row) {
                     T row_scale = 1.0f / diag_ptr[row];  // Scale factor for this row

                     // For each element in this row
                     for (auto j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
                       IndexType col = col_indices[j];
                       T col_scale   = 1.0f / diag_ptr[col];  // Scale factor for the column

                       // Scale by both row and column diagonal elements
                       values[j] = row_scale * values[j] * col_scale;
                     }
                   });
}

/**
 * @brief Set the diagonal elements of a CSR matrix to a scalar value
 *
 * This function sets the diagonal elements of a CSR matrix to a scalar value.
 * The diagonal elements are the elements where the row index and column index are the same.
 *
 *
 * @tparam T The data type of the matrix elements
 * @tparam IndexType The data type of the indices
 *
 * @param[in]  res             RAFT resources for managing device memory and streams
 * @param[in]  csr_matrix_view View of the input CSR matrix to modify
 * @param[in]  scalar          The scalar value to set the diagonal elements to
 *
 */
template <typename T, typename IndexType>
void set_diagonal(raft::resources const& res,
                  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix,
                  T scalar)
{
  auto structure = csr_matrix.structure_view();
  auto n_rows    = structure.get_n_rows();

  auto values      = csr_matrix.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();

  auto policy = raft::resource::get_thrust_policy(res);

  thrust::for_each(policy,
                   thrust::counting_iterator<IndexType>(0),
                   thrust::counting_iterator<IndexType>(n_rows),
                   [values, col_indices, row_offsets, scalar] __device__(IndexType row) {
                     // For each row, find diagonal element (if it exists)
                     for (auto j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
                       if (col_indices[j] == row) {
                         values[j] = scalar;
                         break;
                       }
                     }
                   });
}

}  // namespace raft::sparse::matrix::detail
