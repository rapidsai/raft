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

#include <type_traits>

namespace raft {
namespace sparse {
namespace linalg {

/** Given a CSR adjacency matrix, return the graph Laplacian
 *
 * Note that for non-symmetric matrices, the out-degree Laplacian is returned.
 */
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
  detail::compute_graph_laplacian_kernel<<<threads_per_block, blocks, 0, stream>>>(
    result.get_elements().data(),
    result_structure.get_indices().data(),
    result_structure.get_indptr().data(),
    dim,
    input.get_elements().data(),
    input_structure.get_indices().data(),
    input_structure.get_indptr().data());
  return result;
}

}  // namespace linalg
}  // namespace sparse
}  // namespace raft
