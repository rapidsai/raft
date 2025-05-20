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
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/detail/laplacian.cuh>
#include <raft/sparse/matrix/diagonal.cuh>

namespace raft::sparse::linalg {

/** Given a CSR adjacency matrix, return the graph Laplacian
 *
 * Note that for non-symmetric matrices, the out-degree Laplacian is returned.
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto compute_graph_laplacian(
  raft::resources const& res,
  device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType> input)
{
  return detail::compute_graph_laplacian(res, input);
}

/**
 * @brief Given a CSR adjacency matrix, return the normalized graph Laplacian
 *
 * Return the normalized Laplacian matrix, which is defined as D^(-1/2) * L * D^(-1/2),
 * where D is the diagonal degree matrix and L is the graph Laplacian.
 * Also returns the scaled diagonal degree matrix.
 *
 *
 * @tparam ElementType The data type of the matrix elements
 * @tparam IndptrType The data type of the row pointers
 * @tparam IndicesType The data type of the column indices
 * @tparam NZType The data type for representing nonzero counts
 *
 * @param[in] res RAFT resources for managing device memory and streams
 * @param[in] input View of the input CSR adjacency matrix
 * @param[out] diagonal_out View of the output vector where the scaled diagonal degree
 *                           matrix D^(-1/2) will be stored (must be pre-allocated with
 *                           size at least n_rows)
 *
 * @return A CSR matrix containing the normalized graph Laplacian
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto laplacian_normalized(
  raft::resources const& res,
  device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType> input,
  device_vector_view<ElementType, IndptrType> diagonal_out)
{
  return detail::laplacian_normalized(res, input, diagonal_out);
}

}  // namespace raft::sparse::linalg
