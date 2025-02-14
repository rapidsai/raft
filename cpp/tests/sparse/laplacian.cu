/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace raft {
namespace sparse {
namespace linalg {

TEST(Raft, ComputeGraphLaplacian)
{
  // The following adjacency matrix will be used to allow for manual
  // verification of results:
  // [[0 1 1 1]
  //  [1 0 0 1]
  //  [1 0 0 0]
  //  [1 1 0 0]]

  auto data    = std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1};
  auto indices = std::vector<int>{1, 2, 3, 0, 3, 0, 0, 1};
  auto indptr  = std::vector<int>{0, 3, 5, 6, 8};

  auto res = raft::resources{};
  auto adjacency_matrix =
    make_device_csr_matrix<float>(res, int(indptr.size() - 1), int(indptr.size() - 1), data.size());
  auto adjacency_structure = adjacency_matrix.structure_view();
  raft::copy(adjacency_matrix.get_elements().data(),
             &(data[0]),
             data.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure.get_indices().data(),
             &(indices[0]),
             indices.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(adjacency_structure.get_indptr().data(),
             &(indptr[0]),
             indptr.size(),
             raft::resource::get_cuda_stream(res));
  auto laplacian           = compute_graph_laplacian(res, adjacency_matrix.view());
  auto laplacian_structure = laplacian.structure_view();
  auto laplacian_data      = std::vector<float>(laplacian_structure.get_nnz());
  auto laplacian_indices   = std::vector<int>(laplacian_structure.get_nnz());
  auto laplacian_indptr    = std::vector<int>(laplacian_structure.get_n_rows() + 1);
  raft::copy(&(laplacian_data[0]),
             laplacian.get_elements().data(),
             laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(laplacian_indices[0]),
             laplacian_structure.get_indices().data(),
             laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(laplacian_indptr[0]),
             laplacian_structure.get_indptr().data(),
             laplacian_structure.get_n_rows() + 1,
             raft::resource::get_cuda_stream(res));
  auto expected_data    = std::vector<float>{3, -1, -1, -1, -1, 2, -1, -1, 1, -1, -1, 2};
  auto expected_indices = std::vector<int>{0, 1, 2, 3, 0, 1, 3, 0, 2, 0, 1, 3};
  auto expected_indptr  = std::vector<int>{0, 4, 7, 9, 12};
  raft::resource::sync_stream(res);

  EXPECT_EQ(expected_data, laplacian_data);
  EXPECT_EQ(expected_indices, laplacian_indices);
  EXPECT_EQ(expected_indptr, laplacian_indptr);
}

}  // namespace linalg
}  // namespace sparse
}  // namespace raft
