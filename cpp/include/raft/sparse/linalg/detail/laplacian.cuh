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
#include <raft/core/detail/macros.hpp>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/diagonal.cuh>

#include <type_traits>

namespace raft::sparse::linalg::detail {

/* Compute the graph Laplacian of an adjacency matrix
 *
 * This kernel implements the necessary logic for computing a graph
 * Laplacian for an adjacency matrix in CSR format. A custom kernel is
 * required because cusparse does not conveniently implement matrix subtraction with 64-bit
 * indices. The custom kernel also allows the computation to be completed
 * with no extra allocations or compute.
 */
template <typename ElementType, typename IndptrType, typename IndicesType>
RAFT_KERNEL compute_graph_laplacian_kernel(ElementType* output_values,
                                           IndicesType* output_indices,
                                           IndptrType* output_indptr,
                                           IndptrType dim,
                                           ElementType const* adj_values,
                                           IndicesType const* adj_indices,
                                           IndptrType const* adj_indptr)
{
  /* The graph Laplacian L of an adjacency matrix A is given by:
   * L = D - A
   * where D is the degree matrix of A. The degree matrix is itself defined
   * as the sum of each row of A and represents the degree of the node
   * indicated by the index of the row. */

  for (auto row = threadIdx.x + blockIdx.x * blockDim.x; row < dim; row += blockDim.x * gridDim.x) {
    auto row_begin = adj_indptr[row];
    auto row_end   = adj_indptr[row + 1];
    // All output indexes will need to be offset by the row, since every row will
    // gain exactly one new non-zero element. degree_output_index is the index
    // where we will store the degree of each row
    auto degree_output_index = row_begin + row;
    auto degree_value        = ElementType{};
    // value_index indicates the index of the current value in the original
    // adjacency matrix
    for (auto value_index = row_begin; value_index < row_end; ++value_index) {
      auto col_index         = adj_indices[value_index];
      auto is_lower_diagonal = col_index < row;
      auto output_index      = value_index + row + !is_lower_diagonal;
      auto input_value       = col_index == row ? 0 : adj_values[value_index];
      degree_value += input_value;
      output_values[output_index]  = ElementType{-1} * input_value;
      output_indices[output_index] = col_index;
      // Increment the index where we will store the degree for every non-zero
      // element before we reach the diagonal
      degree_output_index += is_lower_diagonal;
    }
    output_values[degree_output_index]  = degree_value;
    output_indices[degree_output_index] = row;
    output_indptr[row]                  = row_begin + row;
    output_indptr[row + 1]              = row_end + row + 1;
  }
}

template <typename ElementType, typename IndptrType, typename IndicesType, typename NZType>
auto compute_graph_laplacian(
  raft::resources const& res,
  device_csr_matrix_view<ElementType, IndptrType, IndicesType, NZType> input)
{
  auto input_structure = input.structure_view();
  auto dim             = input_structure.get_n_rows();
  RAFT_EXPECTS(dim == input_structure.get_n_cols(),
               "The graph Laplacian can only be computed on a square adjacency matrix");
  auto result = make_device_csr_matrix<std::remove_const_t<ElementType>,
                                       std::remove_const_t<IndptrType>,
                                       std::remove_const_t<IndicesType>,
                                       std::remove_const_t<NZType>>(
    res,
    dim,
    dim,
    /* The nnz for the result will be the dimension of the (square) input matrix plus the number of
     * non-zero elements in the original matrix, since we introduce non-zero elements along the
     * diagonal to represent the degree of each node. */
    input_structure.get_nnz() + dim);
  auto result_structure                         = result.structure_view();
  auto static constexpr const threads_per_block = 256;
  auto blocks = std::min(int((dim + threads_per_block - 1) / threads_per_block), 65535);
  auto stream = resource::get_cuda_stream(res);
  detail::compute_graph_laplacian_kernel<<<blocks, threads_per_block, 0, stream>>>(
    result.get_elements().data(),
    result_structure.get_indices().data(),
    result_structure.get_indptr().data(),
    dim,
    input.get_elements().data(),
    input_structure.get_indices().data(),
    input_structure.get_indptr().data());
  return result;
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
  auto laplacian           = detail::compute_graph_laplacian(res, input);
  auto laplacian_structure = laplacian.structure_view();

  auto diagonal =
    raft::make_device_vector<ElementType, IndptrType>(res, laplacian_structure.get_n_rows());
  raft::sparse::matrix::diagonal(res, laplacian.view(), diagonal.view());

  raft::linalg::unary_op(
    res, raft::make_const_mdspan(diagonal.view()), diagonal.view(), raft::sqrt_op());

  raft::sparse::matrix::scale_by_diagonal_symmetric(res, diagonal.view(), laplacian.view());
  raft::sparse::matrix::set_diagonal(res, laplacian.view(), static_cast<ElementType>(1.0));

  auto stream = resource::get_cuda_stream(res);

  raft::copy(diagonal_out.data_handle(), diagonal.data_handle(), diagonal.size(), stream);
  return laplacian;
}

}  // namespace raft::sparse::linalg::detail
