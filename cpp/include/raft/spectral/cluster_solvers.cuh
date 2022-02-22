/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#pragma once

#include <raft/cluster/kmeans.hpp>
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

    raft::cluster::kmeans(handle,
                          n_obs_vecs,
                          dim,
                          config_.n_clusters,
                          config_.tol,
                          config_.maxIter,
                          obs,
                          codes,
                          residual,
                          iters,
                          config_.seed);
    return std::make_pair(residual, iters);
  }

  auto const& get_config(void) const { return config_; }

 private:
  cluster_solver_config_t<index_type_t, value_type_t, size_type_t> config_;
};

}  // namespace spectral
}  // namespace raft
