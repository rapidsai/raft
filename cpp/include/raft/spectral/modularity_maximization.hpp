/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cmath>
#include <cstdio>

#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <tuple>

#include <raft/spectral/cluster_solvers.hpp>
#include <raft/spectral/eigen_solvers.hpp>
#include <raft/spectral/spectral_util.hpp>

#ifdef COLLECT_TIME_STATISTICS
#include <cuda_profiler_api.h>
#include <stddef.h>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <sys/time.h>
#endif

#ifdef COLLECT_TIME_STATISTICS
static double timer(void) {
  struct timeval tv;
  cudaDeviceSynchronize();
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif

namespace raft {
namespace spectral {

using namespace matrix;
using namespace linalg;

// =========================================================
// Spectral modularity_maximization
// =========================================================

/** Compute partition for a weighted undirected graph. This
 *  partition attempts to minimize the cost function:
 *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
 *
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of partitions.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter_lanczos Maximum number of Lanczos iterations.
 *  @param restartIter_lanczos Maximum size of Lanczos system before
 *    implicit restart.
 *  @param tol_lanczos Convergence tolerance for Lanczos method.
 *  @param maxIter_kmeans Maximum number of k-means iterations.
 *  @param tol_kmeans Convergence tolerance for k-means algorithm.
 *  @param clusters (Output, device memory, n entries) Cluster
 *    assignments.
 *  @param iters_lanczos On exit, number of Lanczos iterations
 *    performed.
 *  @param iters_kmeans On exit, number of k-means iterations
 *    performed.
 *  @return error flag.
 */
template <typename vertex_t, typename weight_t, typename ThrustExePolicy,
          typename EigenSolver, typename ClusterSolver>
std::tuple<vertex_t, weight_t, vertex_t> modularity_maximization(
  handle_t const &handle, ThrustExePolicy thrust_exec_policy,
  sparse_matrix_t<vertex_t, weight_t> const &csr_m,
  EigenSolver const &eigen_solver, ClusterSolver const &cluster_solver,
  vertex_t *__restrict__ clusters, weight_t *eigVals, weight_t *eigVecs) {
  RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");
  RAFT_EXPECTS(eigVals != nullptr, "Null eigVals buffer.");
  RAFT_EXPECTS(eigVecs != nullptr, "Null eigVecs buffer.");

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  std::tuple<vertex_t, weight_t, vertex_t>
    stats;  // # iters eigen solver, cluster solver residual, # iters cluster solver

  vertex_t n = csr_m.nrows;

  // Compute eigenvectors of Modularity Matrix

  // Initialize Modularity Matrix
  modularity_matrix_t<vertex_t, weight_t> B{handle, thrust_exec_policy, csr_m};

  auto eigen_config = eigen_solver.get_config();
  auto nEigVecs = eigen_config.n_eigVecs;

  // Compute eigenvectors corresponding to largest eigenvalues
  std::get<0>(stats) =
    eigen_solver.solve_largest_eigenvectors(handle, B, eigVals, eigVecs);

  // Whiten eigenvector matrix
  transform_eigen_matrix(handle, thrust_exec_policy, n, nEigVecs, eigVecs);

  // notice that at this point the matrix has already been transposed, so we are scaling
  // columns
  scale_obs(nEigVecs, n, eigVecs);
  CHECK_CUDA(stream);

  // Find partition clustering
  auto pair_cluster = cluster_solver.solve(handle, thrust_exec_policy, n,
                                           nEigVecs, eigVecs, clusters);

  std::get<1>(stats) = pair_cluster.first;
  std::get<2>(stats) = pair_cluster.second;

  return stats;
}
//===================================================
// Analysis of graph partition
// =========================================================

/// Compute modularity
/** This function determines the modularity based on a graph and cluster assignments
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of clusters.
 *  @param clusters (Input, device memory, n entries) Cluster assignments.
 *  @param modularity On exit, modularity
 */
template <typename vertex_t, typename weight_t, typename ThrustExePolicy>
void analyzeModularity(handle_t const &handle,
                       ThrustExePolicy thrust_exec_policy,
                       sparse_matrix_t<vertex_t, weight_t> const &csr_m,
                       vertex_t nClusters,
                       vertex_t const *__restrict__ clusters,
                       weight_t &modularity) {
  RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");

  vertex_t i;
  vertex_t n = csr_m.nrows;
  weight_t partModularity, clustersize;

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // Device memory
  vector_t<weight_t> part_i(handle, n);
  vector_t<weight_t> Bx(handle, n);

  // Initialize cuBLAS
  CUBLAS_CHECK(
    cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

  // Initialize Modularity
  modularity_matrix_t<vertex_t, weight_t> B{handle, thrust_exec_policy, csr_m};

  // Initialize output
  modularity = 0;

  // Iterate through partitions
  for (i = 0; i < nClusters; ++i) {
    if (!construct_indicator(handle, thrust_exec_policy, i, n, clustersize,
                             partModularity, clusters, part_i, Bx, B)) {
      WARNING("empty partition");
      continue;
    }

    // Record results
    modularity += partModularity;
  }

  modularity = modularity / B.diagonal.nrm1(thrust_exec_policy);
}

}  // namespace spectral
}  // namespace raft
