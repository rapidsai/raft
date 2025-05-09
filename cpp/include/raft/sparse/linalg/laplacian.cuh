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

/** Given a CSR adjacency matrix, return the normalized graph Laplacian
 *
 * Return the normalized Laplacian matrix, which is defined as D^(-1/2) * L * D^(-1/2),
 * where D is the diagonal degree matrix of the adjacency matrix.
 * Also returns the scaled diagonal degree matrix.
 */
template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto compute_graph_laplacian_normalized(
  raft::resources const& res,
  device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType> input,
  device_vector_view<ElementType, IndptrType> diagonal_out)
{
  return detail::compute_graph_laplacian_normalized(res, input, diagonal_out);
}

}  // namespace raft::sparse::linalg
