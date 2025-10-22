/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
