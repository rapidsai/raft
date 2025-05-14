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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/diagonal.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <vector>

namespace raft::sparse::matrix {

// Helper function to create a test CSR matrix
auto create_test_csr_matrix(raft::resources& res)
{
  // Create a test matrix:
  // [[2 1 0 3]
  //  [0 5 0 2]
  //  [0 0 1 0]
  //  [4 0 0 6]]

  auto data    = std::vector<float>{2, 1, 3, 5, 2, 1, 4, 6};
  auto indices = std::vector<int>{0, 1, 3, 1, 3, 2, 0, 3};
  auto indptr  = std::vector<int>{0, 3, 5, 6, 8};

  auto matrix    = make_device_csr_matrix<float, int, int, int>(res, 4, 4, data.size());
  auto structure = matrix.structure_view();

  raft::copy(
    matrix.get_elements().data(), &(data[0]), data.size(), raft::resource::get_cuda_stream(res));
  raft::copy(structure.get_indices().data(),
             &(indices[0]),
             indices.size(),
             raft::resource::get_cuda_stream(res));
  raft::copy(structure.get_indptr().data(),
             &(indptr[0]),
             indptr.size(),
             raft::resource::get_cuda_stream(res));

  return matrix;
}

TEST(SparseMatrixDiagonal, GetDiagonalVectorFromCSR)
{
  auto res    = raft::resources{};
  auto matrix = create_test_csr_matrix(res);

  // Create diagonal output vector
  auto diagonal_vec = raft::make_device_vector<float, int>(res, 4);

  // Initialize diagonal with zeros first
  auto zeros = std::vector<float>{0, 0, 0, 0};
  raft::copy(
    diagonal_vec.data_handle(), &(zeros[0]), zeros.size(), raft::resource::get_cuda_stream(res));

  // Get diagonal
  diagonal(res, matrix.view(), diagonal_vec.view());

  // Copy result back to host
  auto diagonal_host = std::vector<float>(4);
  raft::copy(&(diagonal_host[0]),
             diagonal_vec.data_handle(),
             diagonal_vec.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Expected diagonal: [2, 5, 1, 6]
  auto expected_diagonal = std::vector<float>{2, 5, 1, 6};
  EXPECT_EQ(expected_diagonal, diagonal_host);
}

TEST(SparseMatrixDiagonal, ScaleCSRByDiagonalSymmetric)
{
  auto res    = raft::resources{};
  auto matrix = create_test_csr_matrix(res);

  // Create diagonal with values [2, 4, 2, 4]
  auto diagonal_data = std::vector<float>{2, 4, 2, 4};
  auto diagonal_vec  = raft::make_device_vector<float, int>(res, 4);
  raft::copy(diagonal_vec.data_handle(),
             &(diagonal_data[0]),
             diagonal_data.size(),
             raft::resource::get_cuda_stream(res));

  // Scale matrix by diagonal
  scale_by_diagonal_symmetric(res, diagonal_vec.view(), matrix.view());

  // Copy result back to host
  auto matrix_structure = matrix.structure_view();
  auto matrix_data_host = std::vector<float>(matrix_structure.get_nnz());
  raft::copy(&(matrix_data_host[0]),
             matrix.get_elements().data(),
             matrix_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Expected scaled matrix:
  // Original: [[2 1 0 3]
  //            [0 5 0 2]
  //            [0 0 1 0]
  //            [4 0 0 6]]
  // Diagonal: [2, 4, 2, 4]
  // Scale formula: value / (diagonal[row] * diagonal[col])
  // Result:
  // [2/(2*2)=0.5, 1/(2*4)=0.125, 3/(2*4)=0.375,
  //  5/(4*4)=0.3125, 2/(4*4)=0.125,
  //  1/(2*2)=0.25,
  //  4/(4*2)=0.5, 6/(4*4)=0.375]

  auto expected_data =
    std::vector<float>{0.5f, 0.125f, 0.375f, 0.3125f, 0.125f, 0.25f, 0.5f, 0.375f};

  const float tol = 1e-5f;
  ASSERT_EQ(expected_data.size(), matrix_data_host.size());
  for (size_t i = 0; i < expected_data.size(); ++i) {
    EXPECT_NEAR(expected_data[i], matrix_data_host[i], tol);
  }
}

TEST(SparseMatrixDiagonal, SetCSRDiagonalToOnes)
{
  auto res    = raft::resources{};
  auto matrix = create_test_csr_matrix(res);

  // Set diagonal to ones
  set_diagonal(res, matrix.view(), 1.0f);

  // Copy result back to host
  auto matrix_structure = matrix.structure_view();
  auto matrix_data_host = std::vector<float>(matrix_structure.get_nnz());
  raft::copy(&(matrix_data_host[0]),
             matrix.get_elements().data(),
             matrix_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Original data:
  // [2, 1, 3, 5, 2, 1, 4, 6]
  // Diagonal elements are at positions 0, 3, 5, 7
  // Expected data after setting diagonal to ones:
  // [1, 1, 3, 1, 2, 1, 4, 1]

  auto expected_data = std::vector<float>{1, 1, 3, 1, 2, 1, 4, 1};
  EXPECT_EQ(expected_data, matrix_data_host);
}

TEST(SparseMatrixDiagonal, CompleteWorkflow)
{
  // Test combining all three operations in a workflow
  auto res    = raft::resources{};
  auto matrix = create_test_csr_matrix(res);

  // 1. Get diagonal
  auto diagonal_vec = raft::make_device_vector<float, int>(res, 4);
  diagonal(res, matrix.view(), diagonal_vec.view());

  // 2. Scale matrix by diagonal
  scale_by_diagonal_symmetric(res, diagonal_vec.view(), matrix.view());

  // 3. Set diagonal to ones
  set_diagonal(res, matrix.view(), 1.0f);

  // Copy results back to host
  auto matrix_structure = matrix.structure_view();
  auto matrix_data_host = std::vector<float>(matrix_structure.get_nnz());
  auto diagonal_host    = std::vector<float>(4);

  raft::copy(&(matrix_data_host[0]),
             matrix.get_elements().data(),
             matrix_structure.get_nnz(),
             raft::resource::get_cuda_stream(res));
  raft::copy(&(diagonal_host[0]),
             diagonal_vec.data_handle(),
             diagonal_vec.size(),
             raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);

  // Original diagonal: [2, 5, 1, 6]
  auto expected_diagonal = std::vector<float>{2, 5, 1, 6};
  EXPECT_EQ(expected_diagonal, diagonal_host);

  // Expected scaled matrix after entire workflow:
  // 1. Original: [[2 1 0 3]
  //               [0 5 0 2]
  //               [0 0 1 0]
  //               [4 0 0 6]]
  // 2. Scale by diagonal [2, 5, 1, 6]:
  //    Result after scaling:
  //    [2/(2*2)=0.5, 1/(2*5)=0.1, 3/(2*6)=0.25,
  //     5/(5*5)=0.2, 2/(5*6)=0.0667,
  //     1/(1*1)=1.0,
  //     4/(6*2)=0.3333, 6/(6*6)=0.1667]
  // 3. Set diagonal to ones:
  //    [1, 0.1, 0.25, 1, 0.0667, 1, 0.3333, 1]

  auto expected_data = std::vector<float>{1.0f, 0.1f, 0.25f, 1.0f, 0.0667f, 1.0f, 0.3333f, 1.0f};

  const float tol = 1e-4f;
  ASSERT_EQ(expected_data.size(), matrix_data_host.size());
  for (size_t i = 0; i < expected_data.size(); ++i) {
    EXPECT_NEAR(expected_data[i], matrix_data_host[i], tol);
  }
}

}  // namespace raft::sparse::matrix
