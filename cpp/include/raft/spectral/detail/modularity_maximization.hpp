/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/normalize.cuh>
#include <raft/spectral/detail/spectral_util.cuh>
#include <raft/spectral/detail/warn_dbg.hpp>
#include <raft/spectral/matrix_wrappers.hpp>

#include <cuda.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include <math.h>
#include <stdio.h>

#include <tuple>

namespace raft {
namespace spectral {
namespace detail {

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
template <typename vertex_t, typename weight_t, typename nnz_t>
void analyzeModularity(
  raft::resources const& handle,
  raft::spectral::matrix::sparse_matrix_t<vertex_t, weight_t, nnz_t> const& csr_m,
  vertex_t nClusters,
  vertex_t const* __restrict__ clusters,
  weight_t& modularity)
{
  RAFT_EXPECTS(clusters != nullptr, "Null clusters buffer.");

  vertex_t i;
  vertex_t n = csr_m.nrows_;
  weight_t partModularity, clustersize;

  auto cublas_h = resource::get_cublas_handle(handle);
  auto stream   = resource::get_cuda_stream(handle);

  // Device memory
  raft::spectral::matrix::vector_t<weight_t> part_i(handle, n);
  raft::spectral::matrix::vector_t<weight_t> Bx(handle, n);

  // Initialize cuBLAS
  RAFT_CUBLAS_TRY(linalg::detail::cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST, stream));

  // Initialize Modularity
  raft::spectral::matrix::modularity_matrix_t<vertex_t, weight_t, nnz_t> B{handle, csr_m};

  // Initialize output
  modularity = 0;

  // Iterate through partitions
  for (i = 0; i < nClusters; ++i) {
    if (!construct_indicator(handle, i, n, clustersize, partModularity, clusters, part_i, Bx, B)) {
      WARNING("empty partition");
      continue;
    }

    // Record results
    modularity += partModularity;
  }

  modularity = modularity / B.diagonal_.nrm1();
}

}  // namespace detail
}  // namespace spectral
}  // namespace raft
