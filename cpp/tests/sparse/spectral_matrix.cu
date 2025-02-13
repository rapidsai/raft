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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resources.hpp>
#include <raft/spectral/laplacian.cuh>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

namespace raft {
namespace spectral {
namespace matrix {
namespace {
template <typename index_type, typename value_type>
struct csr_view_t {
  index_type* offsets;
  index_type* indices;
  value_type* edge_data;
  index_type number_of_vertices;
  index_type number_of_edges;
};
}  // namespace
TEST(Raft, SpectralMatrices)
{
  using index_type = int;
  using value_type = double;
  using nnz_type   = uint64_t;

  raft::resources h;
  ASSERT_EQ(0, raft::resource::get_device_id(h));

  csr_view_t<index_type, value_type> csr_v{nullptr, nullptr, nullptr, 0, 0};

  int const sz = 10;
  vector_t<index_type> d_v{h, sz};

  index_type* ro{nullptr};
  index_type* ci{nullptr};
  value_type* vs{nullptr};
  nnz_type nnz     = 0;
  index_type nrows = 0;
  sparse_matrix_t<index_type, value_type, nnz_type> sm1{h, ro, ci, vs, nrows, nnz};
  sparse_matrix_t<index_type, value_type, nnz_type> sm2{h, csr_v};
  ASSERT_EQ(nullptr, sm1.row_offsets_);
  ASSERT_EQ(nullptr, sm2.row_offsets_);

  auto stream = resource::get_cuda_stream(h);

  auto cnstr_lm1 = [&h, ro, ci, vs, nrows, nnz](void) {
    laplacian_matrix_t<index_type, value_type, nnz_type> lm1{h, ro, ci, vs, nrows, nnz};
  };
  EXPECT_ANY_THROW(cnstr_lm1());  // because of nullptr ptr args

  auto cnstr_lm2 = [&h, &sm2](void) {
    laplacian_matrix_t<index_type, value_type, nnz_type> lm2{h, sm2};
  };
  EXPECT_ANY_THROW(cnstr_lm2());  // because of nullptr ptr args

  auto cnstr_mm1 = [&h, ro, ci, vs, nrows, nnz](void) {
    modularity_matrix_t<index_type, value_type, nnz_type> mm1{h, ro, ci, vs, nrows, nnz};
  };
  EXPECT_ANY_THROW(cnstr_mm1());  // because of nullptr ptr args

  auto cnstr_mm2 = [&h, &sm2](void) {
    modularity_matrix_t<index_type, value_type, nnz_type> mm2{h, sm2};
  };
  EXPECT_ANY_THROW(cnstr_mm2());  // because of nullptr ptr args
}

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

}  // namespace matrix
}  // namespace spectral
}  // namespace raft
