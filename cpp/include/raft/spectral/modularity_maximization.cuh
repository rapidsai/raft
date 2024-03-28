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
#ifndef __MODULARITY_MAXIMIZATION_H
#define __MODULARITY_MAXIMIZATION_H

#pragma once

#include <raft/spectral/detail/modularity_maximization.hpp>

#include <tuple>

namespace raft {
namespace spectral {

// =========================================================
// Spectral modularity_maximization
// =========================================================

/** Compute partition for a weighted undirected graph. This
 *  partition attempts to minimize the cost function:
 *    Cost = \f$sum_i\f$ (Edges cut by ith partition)/(Vertices in ith partition)
 *
 *  @param handle raft handle for managing expensive resources
 *  @param csr_m Weighted graph in CSR format
 *  @param eigen_solver Eigensolver implementation
 *  @param cluster_solver Cluster solver implementation
 *  @param clusters (Output, device memory, n entries) Partition
 *    assignments.
 *  @param eigVals Output eigenvalue array pointer on device
 *  @param eigVecs Output eigenvector array pointer on device
 *  @return statistics: number of eigensolver iterations, .
 */
template <typename vertex_t, typename weight_t, typename EigenSolver, typename ClusterSolver>
std::tuple<vertex_t, weight_t, vertex_t> modularity_maximization(
  raft::resources const& handle,
  matrix::sparse_matrix_t<vertex_t, weight_t> const& csr_m,
  EigenSolver const& eigen_solver,
  ClusterSolver const& cluster_solver,
  vertex_t* __restrict__ clusters,
  weight_t* eigVals,
  weight_t* eigVecs)
{
  return raft::spectral::detail::
    modularity_maximization<vertex_t, weight_t, EigenSolver, ClusterSolver>(
      handle, csr_m, eigen_solver, cluster_solver, clusters, eigVals, eigVecs);
}
//===================================================
// Analysis of graph partition
// =========================================================

/// Compute modularity
/** This function determines the modularity based on a graph and cluster assignments
 *  @param handle raft handle for managing expensive resources
 *  @param csr_m Weighted graph in CSR format
 *  @param nClusters Number of clusters.
 *  @param clusters (Input, device memory, n entries) Cluster assignments.
 *  @param modularity On exit, modularity
 */
template <typename vertex_t, typename weight_t>
void analyzeModularity(raft::resources const& handle,
                       matrix::sparse_matrix_t<vertex_t, weight_t> const& csr_m,
                       vertex_t nClusters,
                       vertex_t const* __restrict__ clusters,
                       weight_t& modularity)
{
  raft::spectral::detail::analyzeModularity<vertex_t, weight_t>(
    handle, csr_m, nClusters, clusters, modularity);
}

}  // namespace spectral
}  // namespace raft

#endif