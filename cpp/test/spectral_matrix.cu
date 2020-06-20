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
  sparse_matrix_t<index_type, value_type> sm1{h, ro, ci, vs, nrows, nnz};
  sparse_matrix_t<index_type, value_type> sm2{h, empty_graph};
  ASSERT_EQ(nullptr, sm1.row_offsets_);
  ASSERT_EQ(nullptr, sm2.row_offsets_);

  auto stream = h.get_stream();
  auto t_exe_pol = thrust::cuda::par.on(stream);

  auto cnstr_lm1 = [&h, t_exe_pol, ro, ci, vs, nrows, nnz](void) {
    laplacian_matrix_t<index_type, value_type> lm1{h,  t_exe_pol, ro, ci,
                                                   vs, nrows,     nnz};
  };
  EXPECT_ANY_THROW(cnstr_lm1());  // because of nullptr ptr args

  auto cnstr_lm2 = [&h, t_exe_pol, &empty_graph](void) {
    laplacian_matrix_t<index_type, value_type> lm2{h, t_exe_pol, empty_graph};
  };
  EXPECT_ANY_THROW(cnstr_lm2());  // because of nullptr ptr args

  auto cnstr_mm1 = [&h, t_exe_pol, ro, ci, vs, nrows, nnz](void) {
    modularity_matrix_t<index_type, value_type> mm1{h,  t_exe_pol, ro, ci,
                                                    vs, nrows,     nnz};
  };
  EXPECT_ANY_THROW(cnstr_mm1());  // because of nullptr ptr args

  auto cnstr_mm2 = [&h, t_exe_pol, &empty_graph](void) {
    modularity_matrix_t<index_type, value_type> mm2{h, t_exe_pol, empty_graph};
  };
  EXPECT_ANY_THROW(cnstr_mm2());  // because of nullptr ptr args
}

}  // namespace raft
