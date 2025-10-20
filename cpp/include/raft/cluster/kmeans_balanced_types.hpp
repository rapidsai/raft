/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/random/rng_state.hpp>

namespace raft::cluster::kmeans_balanced {

/**
 * Simple object to specify hyper-parameters to the balanced k-means algorithm.
 *
 * The following metrics are currently supported in k-means balanced:
 *  - InnerProduct
 *  - L2Expanded
 *  - L2SqrtExpanded
 */
struct kmeans_balanced_params : kmeans_base_params {
  /**
   * Number of training iterations
   */
  uint32_t n_iters = 20;
};

}  // namespace raft::cluster::kmeans_balanced

namespace raft::cluster {

using kmeans_balanced::kmeans_balanced_params;

}  // namespace raft::cluster
