/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <raft/core/mdarray.hpp>
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
 * @brief Compute cluster dispersion metric. This is very useful for
 * automatically finding the 'k' (in kmeans) that improves this metric.
 * @tparam DataT data type
 * @tparam IdxType index type
 * @tparam LayoutPolicy Layout type of the input data.
 * @tparam AccessorPolicy Accessor for the input and output, must be valid accessor on
 *                        device.
 * @tparam TPB threads block for kernels launched
 * @param handle the raft handle
 * @param centroids the cluster centroids. This is assumed to be row-major
 *   and of dimension (nClusters x dim)
 * @param clusterSizes number of points in the dataset which belong to each
 *   cluster. This is of length nClusters
 * @param globalCentroid compute the global weighted centroid of all cluster
 *   centroids. This is of length dim. Use std::nullopt to not return it.
 * @param nPoints number of points in the dataset
 * @return the cluster dispersion value
 */
template <typename DataT,
          typename IdxType        = int,
          typename LayoutPolicy   = raft::layout_c_contiguous,
          typename AccessorPolicy,
          int TPB                 = 256>
DataT dispersion(const raft::handle_t& handle,
                 raft::mdspan<const DataT, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> centroids,
                 raft::mdspan<const IdxType, raft::vector_extent<IdxType>> clusterSizes,
                 std::optional<raft::mdspan<DataT, raft::vector_extent<IdxType>>> globalCentroid,
                 const IdxType nPoints)
{
  DataT* globalCentroid_ptr = nullptr;
  if (globalCentroid.has_value())
  {
    globalCentroid_ptr = globalCentroid.value().data_handle();
  }
  return detail::dispersion<DataT, IdxType, TPB>(centroids.data_handle(),
                                                 clusterSizes.data_handle(),
                                                 globalCentroid_ptr,
                                                 centroids.extent(0),
                                                 nPoints,
                                                 centroids.extent(1),
                                                 handle.get_stream());
}

}  // end namespace stats
}  // end namespace raft

#endif