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
#ifndef __RAND_INDEX_H
#define __RAND_INDEX_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/stats/detail/rand_index.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate RandIndex
 * <a href="https://en.wikipedia.org/wiki/Rand_index">more info on rand index</a>
 * @param firstClusterArray: the array of classes of type T
 * @param secondClusterArray: the array of classes of type T
 * @param size: the size of the data points of type uint64_t
 * @param stream: the cudaStream object
 */
template <typename T>
double rand_index(T* firstClusterArray, T* secondClusterArray, uint64_t size, cudaStream_t stream)
{
  return detail::compute_rand_index(firstClusterArray, secondClusterArray, size, stream);
}

/**
 * @defgroup stats_rand_index Rand Index
 * @{
 */

/**
 * @brief Function to calculate RandIndex
 * <a href="https://en.wikipedia.org/wiki/Rand_index">more info on rand index</a>
 * @tparam value_t the data type
 * @tparam idx_t index type
 * @param[in] handle the raft handle
 * @param[in] first_cluster_array: the array of classes of type value_t
 * @param[in] second_cluster_array: the array of classes of type value_t
 * @return: The RandIndex value.
 */
template <typename value_t, typename idx_t>
double rand_index(raft::resources const& handle,
                  raft::device_vector_view<const value_t, idx_t> first_cluster_array,
                  raft::device_vector_view<const value_t, idx_t> second_cluster_array)
{
  RAFT_EXPECTS(first_cluster_array.extent(0) == second_cluster_array.extent(0),
               "Size mismatch between first_cluster_array and second_cluster_array");
  RAFT_EXPECTS(first_cluster_array.is_exhaustive(), "first_cluster_array must be contiguous");
  RAFT_EXPECTS(second_cluster_array.is_exhaustive(), "second_cluster_array must be contiguous");
  return detail::compute_rand_index(first_cluster_array.data_handle(),
                                    second_cluster_array.data_handle(),
                                    second_cluster_array.extent(0),
                                    resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_rand_index

};  // end namespace stats
};  // end namespace raft

#endif