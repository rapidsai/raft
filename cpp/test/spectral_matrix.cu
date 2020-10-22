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
namespace {
template <typename IndexType, typename ValueType>
struct csr_view_t {
  IndexType* offsets;
  IndexType* indices;
  ValueType* edge_data;
  IndexType number_of_vertices;
  IndexType number_of_edges;
};
}  // namespace
TEST(Raft, SpectralMatrices) {  // NOLINT
  // for unit-test files, that too inside a method, it should be ok to use a
  // `using namespace ...` statement
  using namespace matrix;  // NOLINT
  using index_t = int;
  using value_t = double;

  handle_t h;
  ASSERT_EQ(0, h.get_num_internal_streams());
  ASSERT_EQ(0, h.get_device());

  csr_view_t<index_t, value_t> csr_v{nullptr, nullptr, nullptr, 0, 0};

  int const sz = 10;
  vector_t<index_t> d_v{h, sz};

  index_t* ro{nullptr};
  index_t* ci{nullptr};
  value_t* vs{nullptr};
  index_t nnz = 0;
  index_t nrows = 0;
  sparse_matrix_t<index_t, value_t> sm1{h, ro, ci, vs, nrows, nnz};
  sparse_matrix_t<index_t, value_t> sm2{h, csr_v};
  ASSERT_EQ(nullptr, sm1.row_offsets);
  ASSERT_EQ(nullptr, sm2.row_offsets);

  auto stream = h.get_stream();
  auto t_exe_pol = thrust::cuda::par.on(stream);

  auto cnstr_lm1 = [&h, t_exe_pol, ro, ci, vs, nrows, nnz]() {
    laplacian_matrix_t<index_t, value_t> lm1{h,  t_exe_pol, ro, ci,
                                                   vs, nrows,     nnz};
  };
  EXPECT_ANY_THROW(cnstr_lm1());  // because of nullptr ptr args

  auto cnstr_lm2 = [&h, t_exe_pol, &sm2]() {
    laplacian_matrix_t<index_t, value_t> lm2{h, t_exe_pol, sm2};
  };
  EXPECT_ANY_THROW(cnstr_lm2());  // because of nullptr ptr args

  auto cnstr_mm1 = [&h, t_exe_pol, ro, ci, vs, nrows, nnz]() {
    modularity_matrix_t<index_t, value_t> mm1{h,  t_exe_pol, ro, ci,
                                                    vs, nrows,     nnz};
  };
  EXPECT_ANY_THROW(cnstr_mm1());  // because of nullptr ptr args

  auto cnstr_mm2 = [&h, t_exe_pol, &sm2]() {
    modularity_matrix_t<index_t, value_t> mm2{h, t_exe_pol, sm2};
  };
  EXPECT_ANY_THROW(cnstr_mm2());  // because of nullptr ptr args
}

}  // namespace raft
