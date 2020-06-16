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

#include <raft/spectral/eigen_solvers.hpp>

namespace raft {

TEST(Raft, SpectralSolvers) {
  using namespace matrix;
  using index_type = int;
  using value_type = double;

  handle_t h;
  ASSERT_EQ(0, h.get_num_internal_streams());
  ASSERT_EQ(0, h.get_device());

  index_type* ro{nullptr};
  index_type* ci{nullptr};
  value_type* vs{nullptr};
  index_type nnz = 0;
  index_type nrows = 0;
  sparse_matrix_t<index_type, value_type> sm1{ro, ci, vs, nrows, nnz};
  ASSERT_EQ(nullptr, sm1.row_offsets_);

  laplacian_matrix_t<index_type, value_type> lm1{h, ro, ci, vs, nrows, nnz};
  ASSERT_EQ(nullptr, lm1.diagonal_.raw());

  index_type neigvs{10};
  index_type maxiter{100};
  index_type restart_iter{10};
  value_type tol{1.0e-10};
  bool reorthog{true};

  index_type iter;
  value_type* eigvals{nullptr};
  value_type* eigvecs{nullptr};
  unsigned long long seed{100110021003};

  eigen_solver_config_t<index_type, value_type> cfg{
    neigvs, maxiter, restart_iter, tol, reorthog, seed};

  lanczos_solver_t<index_type, value_type> eig_solver{cfg};

  eig_solver.solve_smallest_eigenvectors(h, lm1, eigvals, eigvecs);

  eig_solver.solve_largest_eigenvectors(h, lm1, eigvals, eigvecs);
}

}  // namespace raft
