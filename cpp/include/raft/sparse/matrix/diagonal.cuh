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
#include <raft/matrix/detail/matrix.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/matrix/detail/diagonal.cuh>
#include <raft/util/input_validation.hpp>

namespace raft::sparse::matrix {

/**
 * @brief Get the diagonal vector from a CSR matrix
 *
 * This function extracts the diagonal elements from a CSR matrix and stores them in a vector.
 * The diagonal elements are the elements where the row index and column index are the same.
 **/
template <typename T, typename IndexType>
void get_diagonal_vector_from_csr(
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix_view,
  raft::device_vector_view<T, IndexType> diagonal,
  raft::resources const& res)
{
  detail::get_diagonal_vector_from_csr(csr_matrix_view, diagonal, res);
}

/**
 * @brief Scale a CSR matrix by its diagonal elements
 *
 * This function scales each element of the CSR matrix by the corresponding diagonal element.
 * The diagonal elements are assumed to be stored in the diagonal vector.
 **/
template <typename T, typename IndexType>
void scale_csr_by_diagonal_symmetric(
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix,
  const raft::device_vector_view<T, IndexType> diagonal,  // Vector of scaling factors
  raft::resources const& res)
{
  detail::scale_csr_by_diagonal_symmetric(csr_matrix, diagonal, res);
}

/**
 * @brief Set the diagonal elements of a CSR matrix to ones
 *
 * This function sets the diagonal elements of a CSR matrix to ones.
 * The diagonal elements are the elements where the row index and column index are the same.
 **/
// TODO: allow any scalar value to be set
template <typename T, typename IndexType>
void set_csr_diagonal_to_ones_thrust(
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix,
  raft::resources const& res)
{
  detail::set_csr_diagonal_to_ones_thrust(csr_matrix, res);
}

}  // namespace raft::sparse::matrix
