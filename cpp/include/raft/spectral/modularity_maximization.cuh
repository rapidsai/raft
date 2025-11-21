/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __MODULARITY_MAXIMIZATION_H
#define __MODULARITY_MAXIMIZATION_H

#pragma once

#include <raft/spectral/detail/modularity_maximization.hpp>

#include <tuple>

namespace raft {
namespace spectral {

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
template <typename vertex_t, typename weight_t, typename nnz_t>
void analyzeModularity(raft::resources const& handle,
                       matrix::sparse_matrix_t<vertex_t, weight_t, nnz_t> const& csr_m,
                       vertex_t nClusters,
                       vertex_t const* __restrict__ clusters,
                       weight_t& modularity)
{
  raft::spectral::detail::analyzeModularity<vertex_t, weight_t, nnz_t>(
    handle, csr_m, nClusters, clusters, modularity);
}

}  // namespace spectral
}  // namespace raft

#endif
