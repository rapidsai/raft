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

#ifndef __COMPLETENESS_SCORE_H
#define __COMPLETENESS_SCORE_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/stats/detail/homogeneity_score.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate the completeness score between two clusters
 *
 * @param truthClusterArray: the array of truth classes of type T
 * @param predClusterArray: the array of predicted classes of type T
 * @param size: the size of the data points of type int
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @param stream: the cudaStream object
 */
template <typename T>
double completeness_score(const T* truthClusterArray,
                          const T* predClusterArray,
                          int size,
                          T lowerLabelRange,
                          T upperLabelRange,
                          cudaStream_t stream)
{
  return detail::homogeneity_score(
    predClusterArray, truthClusterArray, size, lowerLabelRange, upperLabelRange, stream);
}

/**
 * @brief Function to calculate the completeness score between two clusters
 *
 * @tparam DataT the data type
 * @tparam IdxType Index type of matrix extent.
 * @param handle: the raft handle.
 * @param truthClusterArray: the array of truth classes of type DataT
 * @param predClusterArray: the array of predicted classes of type DataT
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 */
template <typename DataT, typename IdxType>
double completeness_score(const raft::handle_t& handle,
                          raft::device_vector_view<const DataT, IdxType> truthClusterArray,
                          raft::device_vector_view<const DataT, IdxType> predClusterArray,
                          DataT lowerLabelRange,
                          DataT upperLabelRange)
{
  RAFT_EXPECTS(truthClusterArray.size() == predClusterArray.size(), "Size mismatch");
  RAFT_EXPECTS(truthClusterArray.is_exhaustive(), "truthClusterArray must be contiguous");
  RAFT_EXPECTS(predClusterArray.is_exhaustive(), "predClusterArray must be contiguous");
  return detail::homogeneity_score(predClusterArray.data_handle(),
                                   truthClusterArray.data_handle(),
                                   truthClusterArray.extent(0),
                                   lowerLabelRange,
                                   upperLabelRange,
                                   handle.get_stream());
}

};  // end namespace stats
};  // end namespace raft

#endif