/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <raft/core/resource/thrust_policy.hpp>

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

  std::pair<value_type_t, index_type_t> solve(raft::resources const& handle,
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
    km_params.n_clusters     = config_.n_clusters;
    km_params.tol            = config_.tol;
    km_params.max_iter       = config_.maxIter;
    km_params.rng_state.seed = config_.seed;

    auto X      = raft::make_device_matrix_view<const value_type_t>(obs, n_obs_vecs, dim);
    auto labels = raft::make_device_vector_view<index_type_t>(codes, n_obs_vecs);
    auto centroids =
      raft::make_device_matrix<value_type_t, index_type_t>(handle, config_.n_clusters, dim);
    auto weight = raft::make_device_vector<value_type_t, index_type_t>(handle, n_obs_vecs);
    thrust::fill(resource::get_thrust_policy(handle),
                 weight.data_handle(),
                 weight.data_handle() + n_obs_vecs,
                 1);

    auto sw = std::make_optional((raft::device_vector_view<const value_type_t>)weight.view());
    raft::cluster::kmeans_fit_predict<value_type_t, index_type_t>(
      handle,
      km_params,
      X,
      sw,
      centroids.view(),
      labels,
      raft::make_host_scalar_view(&residual),
      raft::make_host_scalar_view(&iters));
    return std::make_pair(residual, iters);
  }

  auto const& get_config(void) const { return config_; }

 private:
  cluster_solver_config_t<index_type_t, value_type_t, size_type_t> config_;
};

}  // namespace spectral
}  // namespace raft

#endif