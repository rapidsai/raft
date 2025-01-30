/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/nvtx.hpp>
#include <raft/core/resource/device_id.hpp>
#include <raft/core/resources.hpp>
#include <raft/spectral/eigen_solvers.cuh>
#include <raft/spectral/partition.cuh>

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

  raft::resources h;
  ASSERT_EQ(0, resource::get_device_id(h));

  index_type* ro{nullptr};
  index_type* ci{nullptr};
  value_type* vs{nullptr};
  index_type nnz   = 0;
  index_type nrows = 0;

  sparse_matrix_t<index_type, value_type> sm1{h, ro, ci, vs, nrows, (uint64_t)nnz};
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

  lanczos_solver_t<index_type, value_type> eig_solver{cfg};

  EXPECT_ANY_THROW(eig_solver.solve_smallest_eigenvectors(h, sm1, eigvals, eigvecs));

  EXPECT_ANY_THROW(eig_solver.solve_largest_eigenvectors(h, sm1, eigvals, eigvecs));
}

TEST(Raft, SpectralSolvers)
{
  common::nvtx::range fun_scope("test::SpectralSolvers");
  using namespace matrix;
  using index_type = int;
  using value_type = double;

  raft::resources h;
  ASSERT_EQ(0, resource::get_device_id(h)

  );

  index_type neigvs{10};
  index_type maxiter{100};
  index_type restart_iter{10};
  value_type tol{1.0e-10};
  bool reorthog{true};

  // nullptr expected to trigger exceptions:
  //
  index_type* clusters{nullptr};
  value_type* eigvals{nullptr};
  value_type* eigvecs{nullptr};

  unsigned long long seed{100110021003};

  eigen_solver_config_t<index_type, value_type> eig_cfg{
    neigvs, maxiter, restart_iter, tol, reorthog, seed};
  lanczos_solver_t<index_type, value_type> eig_solver{eig_cfg};

  index_type k{5};

  cluster_solver_config_t<index_type, value_type> clust_cfg{k, maxiter, tol, seed};
  kmeans_solver_t<index_type, value_type> cluster_solver{clust_cfg};

  sparse_matrix_t<index_type, value_type> sm{h, nullptr, nullptr, nullptr, 0, 0};
  EXPECT_ANY_THROW(
    spectral::partition(h, sm, eig_solver, cluster_solver, clusters, eigvals, eigvecs));

  value_type edgeCut{0};
  value_type cost{0};
  EXPECT_ANY_THROW(spectral::analyzePartition(h, sm, k, clusters, edgeCut, cost));
}

}  // namespace spectral
}  // namespace raft
