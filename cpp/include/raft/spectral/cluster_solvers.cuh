/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#ifndef __CLUSTER_SOLVERS_H
#define __CLUSTER_SOLVERS_H

#pragma once

#include <raft/cluster/kmeans.cuh>
#include <utility>  // for std::pair

namespace raft {
namespace spectral {

using namespace matrix;

// aggregate of control params for Eigen Solver:
//
template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
struct cluster_solver_config_t {
  size_type_t n_clusters;
  size_type_t maxIter;

  value_type_t tol;

  unsigned long long seed{123456};
};

template <typename index_type_t, typename value_type_t, typename size_type_t = index_type_t>
struct kmeans_solver_t {
  explicit kmeans_solver_t(
    cluster_solver_config_t<index_type_t, value_type_t, size_type_t> const& config)
    : config_(config)
  {
  }

  std::pair<value_type_t, index_type_t> solve(handle_t const& handle,
                                              size_type_t n_obs_vecs,
                                              size_type_t dim,
                                              value_type_t const* __restrict__ obs,
                                              index_type_t* __restrict__ codes) const
  {
    RAFT_EXPECTS(obs != nullptr, "Null obs buffer.");
    RAFT_EXPECTS(codes != nullptr, "Null codes buffer.");
    value_type_t residual{};
    index_type_t iters{};
    raft::cluster::KMeansParams km_params;
    km_params.n_clusters = config_.n_clusters;
    km_params.tol        = config_.tol;
    km_params.max_iter   = config_.maxIter;
    km_params.seed       = config_.seed;

    auto observations =
      raft::make_device_matrix<value_type_t>(n_obs_vecs, dim, handle.get_stream());
    auto labels = raft::make_device_vector<index_type_t>(n_obs_vecs, handle.get_stream());
    auto centroids =
      raft::make_device_matrix<value_type_t>(n_obs_vecs, config_.n_clusters, handle.get_stream());
    auto centroidsView = std::make_optional(centroids.view());
    auto weight        = raft::make_device_vector<value_type_t>(n_obs_vecs, handle.get_stream());
    auto sw            = std::make_optional(weight.view());
    thrust::fill(handle.get_thrust_policy(), sw.value().data(), sw.value().data() + n_obs_vecs, 1);
    raft::copy(observations.data(), obs, n_obs_vecs * dim, handle.get_stream());
    raft::cluster::kmeans_fit_predict<value_type_t, index_type_t, raft::layout_c_contiguous>(
      handle, km_params, observations.view(), sw, centroidsView, labels.view(), residual, iters);
    raft::copy(codes, labels.data(), n_obs_vecs, handle.get_stream());
    return std::make_pair(residual, iters);
  }

  auto const& get_config(void) const { return config_; }

 private:
  cluster_solver_config_t<index_type_t, value_type_t, size_type_t> config_;
};

}  // namespace spectral
}  // namespace raft

#endif