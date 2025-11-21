/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PARTITION_H
#define __PARTITION_H

#pragma once

#include <raft/spectral/detail/partition.hpp>

#include <tuple>

namespace raft {
namespace spectral {

// =========================================================
// Analysis of graph partition
// =========================================================

/// Compute cost function for partition
/** This function determines the edges cut by a partition and a cost
 *  function:
 *    Cost = \f$sum_i\f$ (Edges cut by ith partition)/(Vertices in ith partition)
 *  Graph is assumed to be weighted and undirected.
 *
 *  @param handle raft handle for managing expensive resources
 *  @param csr_m Weighted graph in CSR format
 *  @param nClusters Number of partitions.
 *  @param clusters (Input, device memory, n entries) Partition
 *    assignments.
 *  @param edgeCut On exit, weight of edges cut by partition.
 *  @param cost On exit, partition cost function.
 */
template <typename vertex_t, typename weight_t, typename nnz_t>
void analyzePartition(raft::resources const& handle,
                      matrix::sparse_matrix_t<vertex_t, weight_t, nnz_t> const& csr_m,
                      vertex_t nClusters,
                      const vertex_t* __restrict__ clusters,
                      weight_t& edgeCut,
                      weight_t& cost)
{
  raft::spectral::detail::analyzePartition<vertex_t, weight_t, nnz_t>(
    handle, csr_m, nClusters, clusters, edgeCut, cost);
}

}  // namespace spectral
}  // namespace raft

#endif
