/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resources.hpp>
#include <raft/spectral/eigen_solvers.cuh>

#include <thrust/device_vector.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <iostream>
#include <memory>
#include <type_traits>

namespace raft {
namespace spectral {

TEST(Raft, EigenSolvers)
{
  common::nvtx::range fun_scope("test::EigenSolvers");
  using namespace matrix;
  using index_type = int;
  using value_type = double;
  using nnz_type   = int;

  raft::resources h;
  ASSERT_EQ(0, resource::get_device_id(h));

  index_type* ro{nullptr};
  index_type* ci{nullptr};
  value_type* vs{nullptr};
  index_type nnz   = 0;
  index_type nrows = 0;

  sparse_matrix_t<index_type, value_type, nnz_type> sm1{h, ro, ci, vs, nrows, nnz};
  ASSERT_EQ(nullptr, sm1.row_offsets_);

  index_type neigvs{10};
  index_type maxiter{100};
  index_type restart_iter{10};
  value_type tol{1.0e-10};
  bool reorthog{true};

  // nullptr expected to trigger exceptions:
  //
  value_type* eigvals{nullptr};
  value_type* eigvecs{nullptr};
  std::uint64_t seed{100110021003};

  eigen_solver_config_t<index_type, value_type> cfg{
    neigvs, maxiter, restart_iter, tol, reorthog, seed};

  lanczos_solver_t<index_type, value_type, nnz_type> eig_solver{cfg};

  EXPECT_ANY_THROW(eig_solver.solve_smallest_eigenvectors(h, sm1, eigvals, eigvecs));

  EXPECT_ANY_THROW(eig_solver.solve_largest_eigenvectors(h, sm1, eigvals, eigvecs));
}

}  // namespace spectral
}  // namespace raft
