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
/**
 * @file adjusted_rand_index.cuh
 * @brief The adjusted Rand index is the corrected-for-chance version of the Rand index.
 * Such a correction for chance establishes a baseline by using the expected similarity
 * of all pair-wise comparisons between clusterings specified by a random model.
 */
#ifndef __ADJUSTED_RAND_INDEX_H
#define __ADJUSTED_RAND_INDEX_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/adjusted_rand_index.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate Adjusted RandIndex
 * @see https://en.wikipedia.org/wiki/Rand_index
 * @tparam T data-type for input label arrays
 * @tparam MathT integral data-type used for computing n-choose-r
 * @param firstClusterArray: the array of classes
 * @param secondClusterArray: the array of classes
 * @param size: the size of the data points of type int
 * @param stream: the cudaStream object
 */
template <typename T, typename MathT = int>
double adjusted_rand_index(const T* firstClusterArray,
                           const T* secondClusterArray,
                           int size,
                           cudaStream_t stream)
{
  return detail::compute_adjusted_rand_index(firstClusterArray, secondClusterArray, size, stream);
}

/**
 * @defgroup stats_adj_rand_index Adjusted Rand Index
 * @{
 */

/**
 * @brief Function to calculate Adjusted RandIndex
 * @see https://en.wikipedia.org/wiki/Rand_index
 * @tparam value_t data-type for input label arrays
 * @tparam math_t integral data-type used for computing n-choose-r
 * @tparam idx_t Index type of matrix extent.
 * @param[in] handle: the raft handle.
 * @param[in] first_cluster_array: the array of classes
 * @param[in] second_cluster_array: the array of classes
 * @return the Adjusted RandIndex
 */
template <typename value_t, typename math_t, typename idx_t>
double adjusted_rand_index(raft::resources const& handle,
                           raft::device_vector_view<const value_t, idx_t> first_cluster_array,
                           raft::device_vector_view<const value_t, idx_t> second_cluster_array)
{
  RAFT_EXPECTS(first_cluster_array.size() == second_cluster_array.size(), "Size mismatch");
  RAFT_EXPECTS(first_cluster_array.is_exhaustive(), "first_cluster_array must be contiguous");
  RAFT_EXPECTS(second_cluster_array.is_exhaustive(), "second_cluster_array must be contiguous");

  return detail::compute_adjusted_rand_index<value_t, math_t>(first_cluster_array.data_handle(),
                                                              second_cluster_array.data_handle(),
                                                              first_cluster_array.extent(0),
                                                              resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_adj_rand_index

};  // end namespace stats
};  // end namespace raft

#endif