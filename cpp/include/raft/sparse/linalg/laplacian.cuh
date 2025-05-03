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
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/detail/laplacian.cuh>
#include <raft/sparse/matrix/diagonal.cuh>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>


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
  return detail::compute_graph_laplacian(res, input);
}

template <typename ElementType>
struct SqrtFunctor {
  __device__ ElementType operator()(ElementType x) {
    return std::sqrt(x);
  }
};

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
  auto laplacian = detail::compute_graph_laplacian(res, input);
  auto laplacian_structure = laplacian.structure_view();

  auto diagonal = raft::make_device_vector<ElementType, IndptrType>(res, laplacian_structure.get_n_rows());
  raft::sparse::matrix::get_diagonal_vector_from_csr(laplacian.view(), diagonal.view(), res);

  thrust::transform(thrust::device,
                    diagonal.data_handle(),
                    diagonal.data_handle() + diagonal.size(),
                    diagonal.data_handle(),  // in-place
                    SqrtFunctor<ElementType>());

  raft::sparse::matrix::scale_csr_by_diagonal_symmetric(laplacian.view(), diagonal.view(), res);
  raft::sparse::matrix::set_csr_diagonal_to_ones_thrust(laplacian.view(), res);

  auto stream = resource::get_cuda_stream(res);

  raft::copy(diagonal_out.data_handle(), diagonal.data_handle(), diagonal.size(), stream);
  return laplacian;
}

}  // namespace linalg
}  // namespace sparse
}  // namespace raft
