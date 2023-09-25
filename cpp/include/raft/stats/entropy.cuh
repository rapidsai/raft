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

#ifndef __ENTROPY_H
#define __ENTROPY_H

#pragma once
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/entropy.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate entropy
 * <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">more info on entropy</a>
 *
 * @tparam T data type
 * @param clusterArray: the array of classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 * @return the entropy score
 */
template <typename T>
double entropy(const T* clusterArray,
               const int size,
               const T lowerLabelRange,
               const T upperLabelRange,
               cudaStream_t stream)
{
  return detail::entropy(clusterArray, size, lowerLabelRange, upperLabelRange, stream);
}

/**
 * @defgroup stats_entropy Entropy
 * @{
 */

/**
 * @brief Function to calculate entropy
 * <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">more info on entropy</a>
 *
 * @tparam value_t data type
 * @tparam idx_t index type
 * @param[in] handle the raft handle
 * @param[in] cluster_array: the array of classes of type value_t
 * @param[in] lower_label_range: the lower bound of the range of labels
 * @param[in] upper_label_range: the upper bound of the range of labels
 * @return the entropy score
 */
template <typename value_t, typename idx_t>
double entropy(raft::resources const& handle,
               raft::device_vector_view<const value_t, idx_t> cluster_array,
               const value_t lower_label_range,
               const value_t upper_label_range)
{
  RAFT_EXPECTS(cluster_array.is_exhaustive(), "cluster_array must be contiguous");
  return detail::entropy(cluster_array.data_handle(),
                         cluster_array.extent(0),
                         lower_label_range,
                         upper_label_range,
                         resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_entropy

};  // end namespace stats
};  // end namespace raft

#endif