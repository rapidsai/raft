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

#ifndef __MUTUAL_INFO_SCORE_H
#define __MUTUAL_INFO_SCORE_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/mutual_info_score.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate the mutual information between two clusters
 * <a href="https://en.wikipedia.org/wiki/Mutual_information">more info on mutual information</a>
 * @param firstClusterArray: the array of classes of type T
 * @param secondClusterArray: the array of classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 */
template <typename T>
double mutual_info_score(const T* firstClusterArray,
                         const T* secondClusterArray,
                         int size,
                         T lowerLabelRange,
                         T upperLabelRange,
                         cudaStream_t stream)
{
  return detail::mutual_info_score(
    firstClusterArray, secondClusterArray, size, lowerLabelRange, upperLabelRange, stream);
}

/**
 * @defgroup stats_mutual_info Mutual Information
 * @{
 */

/**
 * @brief Function to calculate the mutual information between two clusters
 * <a href="https://en.wikipedia.org/wiki/Mutual_information">more info on mutual information</a>
 * @tparam value_t the data type
 * @tparam idx_t index type
 * @param[in] handle the raft handle
 * @param[in] first_cluster_array: the array of classes of type value_t
 * @param[in] second_cluster_array: the array of classes of type value_t
 * @param[in] lower_label_range: the lower bound of the range of labels
 * @param[in] upper_label_range: the upper bound of the range of labels
 * @return the mutual information score
 */
template <typename value_t, typename idx_t>
double mutual_info_score(raft::resources const& handle,
                         raft::device_vector_view<const value_t, idx_t> first_cluster_array,
                         raft::device_vector_view<const value_t, idx_t> second_cluster_array,
                         value_t lower_label_range,
                         value_t upper_label_range)
{
  RAFT_EXPECTS(first_cluster_array.extent(0) == second_cluster_array.extent(0),
               "Size mismatch between first_cluster_array and second_cluster_array");
  RAFT_EXPECTS(first_cluster_array.is_exhaustive(), "first_cluster_array must be contiguous");
  RAFT_EXPECTS(second_cluster_array.is_exhaustive(), "second_cluster_array must be contiguous");
  return detail::mutual_info_score(first_cluster_array.data_handle(),
                                   second_cluster_array.data_handle(),
                                   first_cluster_array.extent(0),
                                   lower_label_range,
                                   upper_label_range,
                                   resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_mutual_info

};  // end namespace stats
};  // end namespace raft

#endif