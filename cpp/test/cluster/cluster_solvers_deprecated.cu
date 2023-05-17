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

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <raft/spectral/cluster_solvers_deprecated.cuh>

namespace raft {
namespace spectral {

TEST(Raft, ClusterSolvers)
{
  using namespace matrix;
  using index_type = int;
  using value_type = double;

  raft::resources h;

  index_type maxiter{100};
  value_type tol{1.0e-10};
  unsigned long long seed{100110021003};

  auto stream = resource::get_cuda_stream(h);

  index_type n{100};
  index_type d{10};
  index_type k{5};

  // nullptr expected to trigger exceptions:
  //
  value_type* eigvecs{nullptr};
  index_type* codes{nullptr};

  cluster_solver_config_deprecated_t<index_type, value_type> cfg{k, maxiter, tol, seed};
  kmeans_solver_deprecated_t<index_type, value_type> cluster_solver{cfg};

  EXPECT_ANY_THROW(cluster_solver.solve(h, n, d, eigvecs, codes));
}

}  // namespace spectral
}  // namespace raft
