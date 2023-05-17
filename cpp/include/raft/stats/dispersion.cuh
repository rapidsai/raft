/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#ifndef __DISPERSION_H
#define __DISPERSION_H

#pragma once

#include <optional>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/dispersion.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute cluster dispersion metric. This is very useful for
 * automatically finding the 'k' (in kmeans) that improves this metric.
 * @tparam DataT data type
 * @tparam IdxT index type
 * @tparam TPB threads block for kernels launched
 * @param centroids the cluster centroids. This is assumed to be row-major
 *   and of dimension (nClusters x dim)
 * @param clusterSizes number of points in the dataset which belong to each
 *   cluster. This is of length nClusters
 * @param globalCentroid compute the global weighted centroid of all cluster
 *   centroids. This is of length dim. Pass a nullptr if this is not needed
 * @param nClusters number of clusters
 * @param nPoints number of points in the dataset
 * @param dim dataset dimensionality
 * @param stream cuda stream
 * @return the cluster dispersion value
 */
template <typename DataT, typename IdxT = int, int TPB = 256>
DataT dispersion(const DataT* centroids,
                 const IdxT* clusterSizes,
                 DataT* globalCentroid,
                 IdxT nClusters,
                 IdxT nPoints,
                 IdxT dim,
                 cudaStream_t stream)
{
  return detail::dispersion<DataT, IdxT, TPB>(
    centroids, clusterSizes, globalCentroid, nClusters, nPoints, dim, stream);
}

/**
 * @defgroup stats_cluster_dispersion Cluster Dispersion Metric
 * @{
 */

/**
 * @brief Compute cluster dispersion metric. This is very useful for
 * automatically finding the 'k' (in kmeans) that improves this metric.
 * The cluster dispersion metric is defined as the square root of the sum of the
 * squared distances between the cluster centroids and the global centroid
 * @tparam value_t data type
 * @tparam idx_t index type
 * @param[in]  handle the raft handle
 * @param[in]  centroids the cluster centroids. This is assumed to be row-major
 *   and of dimension (n_clusters x dim)
 * @param[in]  cluster_sizes number of points in the dataset which belong to each
 *   cluster. This is of length n_clusters
 * @param[out] global_centroid compute the global weighted centroid of all cluster
 *   centroids. This is of length dim. Use std::nullopt to not return it.
 * @param[in]  n_points number of points in the dataset
 * @return the cluster dispersion value
 */
template <typename value_t, typename idx_t>
value_t cluster_dispersion(
  raft::resources const& handle,
  raft::device_matrix_view<const value_t, idx_t, raft::row_major> centroids,
  raft::device_vector_view<const idx_t, idx_t> cluster_sizes,
  std::optional<raft::device_vector_view<value_t, idx_t>> global_centroid,
  const idx_t n_points)
{
  RAFT_EXPECTS(cluster_sizes.extent(0) == centroids.extent(0), "Size mismatch");
  RAFT_EXPECTS(cluster_sizes.is_exhaustive(), "cluster_sizes must be contiguous");

  value_t* global_centroid_ptr = nullptr;
  if (global_centroid.has_value()) {
    RAFT_EXPECTS(global_centroid.value().extent(0) == centroids.extent(1),
                 "Size mismatch between global_centroid and centroids");
    RAFT_EXPECTS(global_centroid.value().is_exhaustive(), "global_centroid must be contiguous");
    global_centroid_ptr = global_centroid.value().data_handle();
  }
  return detail::dispersion<value_t, idx_t>(centroids.data_handle(),
                                            cluster_sizes.data_handle(),
                                            global_centroid_ptr,
                                            centroids.extent(0),
                                            n_points,
                                            centroids.extent(1),
                                            resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_cluster_dispersion

/**
 * @brief Overload of `cluster_dispersion` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for the optional arguments.
 *
 * Please see above for documentation of `cluster_dispersion`.
 */
template <typename value_t, typename idx_t>
value_t cluster_dispersion(
  raft::resources const& handle,
  raft::device_matrix_view<const value_t, idx_t, raft::row_major> centroids,
  raft::device_vector_view<const idx_t, idx_t> cluster_sizes,
  std::nullopt_t global_centroid,
  const idx_t n_points)
{
  std::optional<raft::device_vector_view<value_t, idx_t>> opt_centroid = global_centroid;
  return cluster_dispersion(handle, centroids, cluster_sizes, opt_centroid, n_points);
}
}  // end namespace stats
}  // end namespace raft

#endif