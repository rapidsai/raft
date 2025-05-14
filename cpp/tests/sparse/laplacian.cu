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

namespace raft::sparse::linalg {

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

TEST(Raft, ComputeGraphLaplacianNormalized)
{
  // Using the same adjacency matrix as in the ComputeGraphLaplacian test:
  // [[0 1 1 1]
  //  [1 0 0 1]
  //  [1 0 0 0]
  //  [1 1 0 0]]

  auto data    = std::vector<float>{1, 1, 1, 1, 1, 1, 1, 1};
  auto indices = std::vector<int>{1, 2, 3, 0, 3, 0, 0, 1};
  auto indptr  = std::vector<int>{0, 3, 5, 6, 8};

  auto res              = raft::resources{};
  auto adjacency_matrix = make_device_csr_matrix<float, int, int, int>(
    res, (indptr.size() - 1), (indptr.size() - 1), data.size());
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

  // Create diagonal output vector
  auto diagonal_out = raft::make_device_vector<float, int>(res, adjacency_structure.get_n_rows());

  // Compute normalized Laplacian
  auto normalized_laplacian =
    laplacian_normalized(res, adjacency_matrix.view(), diagonal_out.view());
  auto normalized_laplacian_structure = normalized_laplacian.structure_view();

  // Copy results back to host
  auto normalized_laplacian_data    = std::vector<float>(normalized_laplacian_structure.get_nnz());
  auto normalized_laplacian_indices = std::vector<int>(normalized_laplacian_structure.get_nnz());
  auto normalized_laplacian_indptr =
    std::vector<int>(normalized_laplacian_structure.get_n_rows() + 1);
  auto diagonal_data = std::vector<float>(adjacency_structure.get_n_rows());

  raft::copy(&(normalized_laplacian_data[0]),
             normalized_laplacian.get_elements().data(),
             normalized_laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(normalized_laplacian_indices[0]),
             normalized_laplacian_structure.get_indices().data(),
             normalized_laplacian_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(normalized_laplacian_indptr[0]),
             normalized_laplacian_structure.get_indptr().data(),
             normalized_laplacian_structure.get_n_rows() + 1,
             raft::resource::get_cuda_stream(res));
  raft::copy(&(diagonal_data[0]),
             diagonal_out.data_handle(),
             diagonal_out.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // For the given adjacency matrix, the diagonal degree matrix D has values [3, 2, 1, 2]
  // The square root of these values is [√3, √2, 1, √2]
  // The normalized Laplacian should have values close to:
  // [1, -1/√(3*2), -1/√(3*1), -1/√(3*2),
  //  -1/√(2*3), 1, -1/√(2*2),
  //  -1/√(1*3), 1,
  //  -1/√(2*3), -1/√(2*2), 1]

  // Expected diagonal values (sqrt of diagonal degree matrix)
  auto expected_diagonal =
    std::vector<float>{std::sqrt(3.0f), std::sqrt(2.0f), 1.0f, std::sqrt(2.0f)};

  // Expected normalized Laplacian values
  auto expected_data = std::vector<float>{1.0f,
                                          -1.0f / std::sqrt(3.0f * 2.0f),
                                          -1.0f / std::sqrt(3.0f * 1.0f),
                                          -1.0f / std::sqrt(3.0f * 2.0f),
                                          -1.0f / std::sqrt(2.0f * 3.0f),
                                          1.0f,
                                          -1.0f / std::sqrt(2.0f * 2.0f),
                                          -1.0f / std::sqrt(1.0f * 3.0f),
                                          1.0f,
                                          -1.0f / std::sqrt(2.0f * 3.0f),
                                          -1.0f / std::sqrt(2.0f * 2.0f),
                                          1.0f};

  // Same indices and indptr as non-normalized Laplacian
  auto expected_indices = std::vector<int>{0, 1, 2, 3, 0, 1, 3, 0, 2, 0, 1, 3};
  auto expected_indptr  = std::vector<int>{0, 4, 7, 9, 12};

  // Compare results with expected values with a small tolerance for floating point differences
  const float tol = 1e-6f;
  ASSERT_EQ(expected_diagonal.size(), diagonal_data.size());
  for (size_t i = 0; i < expected_diagonal.size(); ++i) {
    EXPECT_NEAR(expected_diagonal[i], diagonal_data[i], tol);
  }

  ASSERT_EQ(expected_data.size(), normalized_laplacian_data.size());
  for (size_t i = 0; i < expected_data.size(); ++i) {
    EXPECT_NEAR(expected_data[i], normalized_laplacian_data[i], tol);
  }

  EXPECT_EQ(expected_indices, normalized_laplacian_indices);
  EXPECT_EQ(expected_indptr, normalized_laplacian_indptr);
}

}  // namespace raft::sparse::linalg
