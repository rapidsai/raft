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
#include <raft/core/logger.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/random/rng_state.hpp>

namespace raft::cluster {

/** Base structure for parameters that are common to all k-means algorithms */
struct kmeans_base_params {
  /**
   * Metric to use for distance computation. The supported metrics can vary per algorithm.
   */
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2Expanded;
};

}  // namespace raft::cluster

namespace raft::cluster::kmeans {

/**
 * Simple object to specify hyper-parameters to the kmeans algorithm.
 */
struct KMeansParams : kmeans_base_params {
  enum InitMethod {

    /**
     * Sample the centroids using the kmeans++ strategy
     */
    KMeansPlusPlus,

    /**
     * Sample the centroids uniformly at random
     */
    Random,

    /**
     * User provides the array of initial centroids
     */
    Array
  };

  /**
   * The number of clusters to form as well as the number of centroids to generate (default:8).
   */
  int n_clusters = 8;

  /**
   * Method for initialization, defaults to k-means++:
   *  - InitMethod::KMeansPlusPlus (k-means++): Use scalable k-means++ algorithm
   * to select the initial cluster centers.
   *  - InitMethod::Random (random): Choose 'n_clusters' observations (rows) at
   * random from the input data for the initial centroids.
   *  - InitMethod::Array (ndarray): Use 'centroids' as initial cluster centers.
   */
  InitMethod init = KMeansPlusPlus;

  /**
   * Maximum number of iterations of the k-means algorithm for a single run.
   */
  int max_iter = 300;

  /**
   * Relative tolerance with regards to inertia to declare convergence.
   */
  double tol = 1e-4;

  /**
   * verbosity level.
   */
  int verbosity = RAFT_LEVEL_INFO;

  /**
   * Seed to the random number generator.
   */
  raft::random::RngState rng_state{0};

  /**
   * Number of instance k-means algorithm will be run with different seeds.
   */
  int n_init = 1;

  /**
   * Oversampling factor for use in the k-means|| algorithm
   */
  double oversampling_factor = 2.0;

  // batch_samples and batch_centroids are used to tile 1NN computation which is
  // useful to optimize/control the memory footprint
  // Default tile is [batch_samples x n_clusters] i.e. when batch_centroids is 0
  // then don't tile the centroids
  int batch_samples = 1 << 15;

  /**
   * if 0 then batch_centroids = n_clusters
   */
  int batch_centroids = 0;  //

  bool inertia_check = false;
};

}  // namespace raft::cluster::kmeans

namespace raft::cluster {

using kmeans::KMeansParams;

}  // namespace raft::cluster
