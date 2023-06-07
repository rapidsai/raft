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
#pragma once

#include <raft/cluster/detail/kmeans_deprecated.cuh>

namespace raft {
namespace cluster {

/**
 *  @brief Find clusters with k-means algorithm.
 *    Initial centroids are chosen with k-means++ algorithm. Empty
 *    clusters are reinitialized by choosing new centroids with
 *    k-means++ algorithm.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param n Number of observation vectors.
 *  @param d Dimension of observation vectors.
 *  @param k Number of clusters.
 *  @param tol Tolerance for convergence. k-means stops when the
 *    change in residual divided by n is less than tol.
 *  @param maxiter Maximum number of k-means iterations.
 *  @param obs (Input, device memory, d*n entries) Observation
 *    matrix. Matrix is stored column-major and each column is an
 *    observation vector. Matrix dimensions are d x n.
 *  @param codes (Output, device memory, n entries) Cluster
 *    assignments.
 *  @param residual On exit, residual sum of squares (sum of squares
 *    of distances between observation vectors and centroids).
 *  @param iters on exit, number of k-means iterations.
 *  @param seed random seed to be used.
 *  @return error flag
 */
template <typename index_type_t, typename value_type_t>
int kmeans(raft::resources const& handle,
           index_type_t n,
           index_type_t d,
           index_type_t k,
           value_type_t tol,
           index_type_t maxiter,
           const value_type_t* __restrict__ obs,
           index_type_t* __restrict__ codes,
           value_type_t& residual,
           index_type_t& iters,
           unsigned long long seed = 123456)
{
  return detail::kmeans<index_type_t, value_type_t>(
    handle, n, d, k, tol, maxiter, obs, codes, residual, iters, seed);
}
}  // namespace cluster
}  // namespace raft
