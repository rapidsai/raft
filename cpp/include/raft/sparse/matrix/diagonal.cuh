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
#include <raft/matrix/init.cuh>
#include <raft/sparse/matrix/detail/diagonal.cuh>
#include <raft/util/input_validation.hpp>

namespace raft::sparse::matrix {

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
  detail::diagonal(res, csr_matrix_view, diagonal);
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
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix_view)
{
  detail::scale_by_diagonal_symmetric(res, diagonal, csr_matrix_view);
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
                  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix_view,
                  T scalar)
{
  detail::set_diagonal(res, csr_matrix_view, scalar);
}

}  // namespace raft::sparse::matrix
