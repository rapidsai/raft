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

#include "../test_utils.cuh"

#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/core/handle.hpp>
#include <raft/spectral/modularity_maximization.cuh>
#include <raft/spectral/partition.cuh>

namespace raft {
namespace cluster {

/**
 * Warning: There appears to be a CUDA 12.2 bug in cusparse that causes an
 * alignment issue. We've fixed the bug in our code through a workaround
 * (see raft/sparse/linalg/spmm.hpp for fix). This test is meant to fail
 * in the case where the fix is accidentally reverted, so that it doesn't
 * break any downstream libraries that depend on RAFT
 */
TEST(Raft, Spectral)
{
  raft::handle_t handle;

  std::vector<int32_t> h_offsets({0, 2, 4, 7, 10, 12, 14});
  std::vector<int32_t> h_indices({1, 2, 0, 2, 0, 1, 3, 2, 4, 5, 3, 5, 3, 4});
  std::vector<float> h_values(
    {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});
  std::vector<int32_t> expected_clustering({1, 1, 1, 0, 0, 0});

  int32_t n_clusters{2};
  int32_t n_eigenvectors{2};
  int32_t evs_max_it{100};
  int32_t kmean_max_it{100};
  int32_t restartIter_lanczos = 15 + n_eigenvectors;
  float evs_tol{0.001};
  float kmean_tol{0.001};
  unsigned long long seed1{1234567};
  unsigned long long seed2{12345678};
  bool reorthog{false};

  rmm::device_uvector<int32_t> offsets(h_offsets.size(), handle.get_stream());
  rmm::device_uvector<int32_t> indices(h_indices.size(), handle.get_stream());
  rmm::device_uvector<float> values(h_indices.size(), handle.get_stream());
  rmm::device_uvector<int32_t> clustering(expected_clustering.size(), handle.get_stream());
  rmm::device_uvector<float> eigenvalues(n_eigenvectors, handle.get_stream());
  rmm::device_uvector<float> eigenvectors(n_eigenvectors * expected_clustering.size(),
                                          handle.get_stream());

  rmm::device_uvector<int32_t> exp_dev(expected_clustering.size(), handle.get_stream());

  raft::update_device(
    exp_dev.data(), expected_clustering.data(), expected_clustering.size(), handle.get_stream());

  raft::update_device(offsets.data(), h_offsets.data(), h_offsets.size(), handle.get_stream());
  raft::update_device(indices.data(), h_indices.data(), h_indices.size(), handle.get_stream());
  raft::update_device(values.data(), h_values.data(), h_values.size(), handle.get_stream());

  raft::spectral::matrix::sparse_matrix_t<int32_t, float> const matrix{
    handle,
    offsets.data(),
    indices.data(),
    values.data(),
    static_cast<int32_t>(offsets.size() - 1),
    static_cast<int32_t>(indices.size())};

  raft::spectral::eigen_solver_config_t<int32_t, float> eig_cfg{
    n_eigenvectors, evs_max_it, restartIter_lanczos, evs_tol, reorthog, seed1};
  raft::spectral::lanczos_solver_t<int32_t, float> eig_solver{eig_cfg};

  raft::spectral::cluster_solver_config_t<int32_t, float> clust_cfg{
    n_clusters, kmean_max_it, kmean_tol, seed2};
  raft::spectral::kmeans_solver_t<int32_t, float> cluster_solver{clust_cfg};

  raft::spectral::partition(handle,
                            matrix,
                            eig_solver,
                            cluster_solver,
                            clustering.data(),
                            eigenvalues.data(),
                            eigenvectors.data());

  ASSERT_TRUE(devArrMatch(expected_clustering.data(),
                          exp_dev.data(),
                          exp_dev.size(),
                          1,
                          raft::Compare<int32_t>(),
                          handle.get_stream()));
}

}  // namespace cluster
}  // namespace raft