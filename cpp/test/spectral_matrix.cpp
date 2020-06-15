/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/handle.hpp>

#include <raft/spectral/matrix_wrappers.hpp>

namespace raft {

TEST(Raft, SpectralMatrices) {
  using namespace matrix;
  using index_type = int;
  using value_type = double;

  handle_t h;
  ASSERT_EQ(0, h.get_num_internal_streams());
  ASSERT_EQ(0, h.get_device());

  int const sz = 10;
  vector_t<index_type> d_v{h, sz};

  GraphCSRView<index_type, index_type, value_type> empty_graph;

  index_type* ro{nullptr};
  index_type* ci{nullptr};
  value_type* vs{nullptr};
  index_type nnz = 0;
  index_type nrows = 0;
  sparse_matrix_t<index_type, value_type> sm1{ro, ci, vs, nrows, nnz};
  sparse_matrix_t<index_type, value_type> sm2{empty_graph};
  ASSERT_EQ(nullptr, sm1.row_offsets_);
  ASSERT_EQ(nullptr, sm2.row_offsets_);

  laplacian_matrix_t<index_type, value_type> lm1{h, ro, ci, vs, nrows, nnz};
  laplacian_matrix_t<index_type, value_type> lm2{h, empty_graph};
  ASSERT_EQ(nullptr, lm1.diagonal_.raw());
  ASSERT_EQ(nullptr, lm2.diagonal_.raw());

  modularity_matrix_t<index_type, value_type> mm1{h, ro, ci, vs, nrows, nnz};
  modularity_matrix_t<index_type, value_type> mm2{h, empty_graph};
  ASSERT_EQ(nullptr, mm1.diagonal_.raw());
  ASSERT_EQ(nullptr, mm2.diagonal_.raw());
}

}  // namespace raft
