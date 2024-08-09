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
#pragma once

#include "raft/util/cudart_utils.hpp"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/spectral/cluster_solvers.cuh>
#include <raft/spectral/detail/spectral_util.cuh>
#include <raft/spectral/eigen_solvers.cuh>
#include <raft/spectral/matrix_wrappers.hpp>

#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <math.h>
#include <stdio.h>

#include <tuple>
#include <vector>

namespace raft {
namespace spectral {
namespace detail {

// =========================================================
// Spectral partitioner
// =========================================================

/// Compute spectral graph partition
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
 *  @param clusters (Output, device memory, n entries) Partition
 *    assignments.
 *  @param iters_lanczos On exit, number of Lanczos iterations
 *    performed.
 *  @param iters_kmeans On exit, number of k-means iterations
 *    performed.
 *  @return statistics: number of eigensolver iterations, .
 */
template <typename vertex_t, typename weight_t, typename EigenSolver, typename ClusterSolver>
std::tuple<vertex_t, weight_t, vertex_t> partition(
  raft::resources const& handle,
  spectral::matrix::sparse_matrix_t<vertex_t, weight_t> const& csr_m,
  EigenSolver const& eigen_solver,
  ClusterSolver const& cluster_solver,
  vertex_t* __restrict__ clusters,
  weight_t* eigVals,
  weight_t* eigVecs)
{
  RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");
  RAFT_EXPECTS(eigVals != nullptr, "Null eigVals buffer.");
  RAFT_EXPECTS(eigVecs != nullptr, "Null eigVecs buffer.");

  auto stream   = resource::get_cuda_stream(handle);
  auto cublas_h = resource::get_cublas_handle(handle);

  std::tuple<vertex_t, weight_t, vertex_t>
    stats;  //{iters_eig_solver,residual_cluster,iters_cluster_solver} // # iters eigen solver,
            // cluster solver residual, # iters cluster solver

  vertex_t n = csr_m.nrows_;

  // -------------------------------------------------------
  // Spectral partitioner
  // -------------------------------------------------------

  // Compute eigenvectors of Laplacian

  // Initialize Laplacian
  /// sparse_matrix_t<vertex_t, weight_t> A{handle, graph};
  spectral::matrix::laplacian_matrix_t<vertex_t, weight_t> L{handle, csr_m};

  // print_device_vector("laplacian dd", L.diagonal_.raw(), L.nrows_, std::cout);
  // print_device_vector("laplacian rows", L.row_offsets_, L.nnz_, std::cout);

  
  auto Svals = raft::make_device_vector<weight_t>(handle, 4);
  auto Scols = raft::make_device_vector<vertex_t>(handle, 4);
  auto Srows = raft::make_device_vector<vertex_t>(handle, 4);
  std::vector<weight_t> Svalsvec = {3, 4, 5, 6};
  std::vector<vertex_t> Scolsvec = {2, 0, 2, 1};
  std::vector<vertex_t> Srowsvec = {0, 1, 3, 4};
  raft::copy(Svals.data_handle(), Svalsvec.data(), 4, stream);
  raft::copy(Scols.data_handle(), Scolsvec.data(), 4, stream);
  raft::copy(Srows.data_handle(), Srowsvec.data(), 4, stream);
  spectral::matrix::sparse_matrix_t<vertex_t, weight_t> S(handle, Srows.data_handle(), Scols.data_handle(), Svals.data_handle(), 3, 3, 4);
  print_device_vector("", S.values_, 4, std::cout);
  print_device_vector("", S.col_indices_, 4, std::cout);
  print_device_vector("", S.row_offsets_, 4, std::cout);
  spectral::matrix::laplacian_matrix_t<vertex_t, weight_t> laplacian(handle, S);
  print_device_vector("", laplacian.values_, 4, std::cout);
  print_device_vector("", laplacian.col_indices_, 4, std::cout);
  print_device_vector("", laplacian.row_offsets_, 4, std::cout);
  print_device_vector("", laplacian.diagonal_.raw(), 3, std::cout);


  auto eigen_config = eigen_solver.get_config();
  auto nEigVecs     = eigen_config.n_eigVecs;

  // Compute smallest eigenvalues and eigenvectors
  std::get<0>(stats) = eigen_solver.solve_smallest_eigenvectors(handle, csr_m, eigVals, eigVecs);

  std::cout << "iters " << std::get<0>(stats) << std::endl;

  std::ofstream out_file("output1.txt"); // Open a file for writing
  
  // Check if the file is open
  if (!out_file.is_open()) {
    std::cerr << "Failed to open output file!" << std::endl;
  }

  print_device_vector("eigenvals", eigVals, nEigVecs, out_file);
  print_device_vector("eigenvecs", eigVecs, n * nEigVecs, out_file);

  // Whiten eigenvector matrix
  // transform_eigen_matrix(handle, n, nEigVecs, eigVecs);

  // Find partition clustering
  auto pair_cluster = cluster_solver.solve(handle, n, nEigVecs, eigVecs, clusters);

  std::get<1>(stats) = pair_cluster.first;
  std::get<2>(stats) = pair_cluster.second;

  return stats;
}

// =========================================================
// Analysis of graph partition
// =========================================================

/// Compute cost function for partition
/** This function determines the edges cut by a partition and a cost
 *  function:
 *    Cost = \sum_i (Edges cut by ith partition)/(Vertices in ith partition)
 *  Graph is assumed to be weighted and undirected.
 *
 *  @param G Weighted graph in CSR format
 *  @param nClusters Number of partitions.
 *  @param clusters (Input, device memory, n entries) Partition
 *    assignments.
 *  @param edgeCut On exit, weight of edges cut by partition.
 *  @param cost On exit, partition cost function.
 *  @return error flag.
 */
template <typename vertex_t, typename weight_t>
void analyzePartition(raft::resources const& handle,
                      spectral::matrix::sparse_matrix_t<vertex_t, weight_t> const& csr_m,
                      vertex_t nClusters,
                      const vertex_t* __restrict__ clusters,
                      weight_t& edgeCut,
                      weight_t& cost)
{
  RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");

  vertex_t i;
  vertex_t n = csr_m.nrows_;

  auto stream   = resource::get_cuda_stream(handle);
  auto cublas_h = resource::get_cublas_handle(handle);

  weight_t partEdgesCut, clustersize;

  // Device memory
  spectral::matrix::vector_t<weight_t> part_i(handle, n);
  spectral::matrix::vector_t<weight_t> Lx(handle, n);

  // Initialize cuBLAS
  RAFT_CUBLAS_TRY(
    raft::linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

  // Initialize Laplacian
  /// sparse_matrix_t<vertex_t, weight_t> A{handle, graph};
  spectral::matrix::laplacian_matrix_t<vertex_t, weight_t> L{handle, csr_m};

  // Initialize output
  cost    = 0;
  edgeCut = 0;

  // Iterate through partitions
  for (i = 0; i < nClusters; ++i) {
    // Construct indicator vector for ith partition
    if (!construct_indicator(handle, i, n, clustersize, partEdgesCut, clusters, part_i, Lx, L)) {
      WARNING("empty partition");
      continue;
    }

    // Record results
    cost += partEdgesCut / clustersize;
    edgeCut += partEdgesCut / 2;
  }
}

}  // namespace detail
}  // namespace spectral
}  // namespace raft
