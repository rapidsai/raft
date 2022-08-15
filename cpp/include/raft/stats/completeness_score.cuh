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

#include <raft/core/mdarray.hpp>
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
 * @tparam T the data type
 * @tparam IdxType Index type of matrix extent.
 * @tparam LayoutPolicy Layout type of the input matrix. When layout is strided, it can
 *                      be a submatrix of a larger matrix. Arbitrary stride is not supported.
 * @tparam AccessorPolicy Accessor for the input and output, must be valid accessor on
 *                        device.
 * @param handle: the raft handle.
 * @param truthClusterArray: the array of truth classes of type T
 * @param predClusterArray: the array of predicted classes of type T
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 */
template <typename T, typename IdxType, typename LayoutPolicy, typename AccessorPolicy>
double completeness_score(
  const raft::handle_t& handle,
  raft::mdspan<const T, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy>
    truthClusterArray,
  raft::mdspan<const T, raft::vector_extent<IdxType>, LayoutPolicy, AccessorPolicy>
    predClusterArray,
  T lowerLabelRange,
  T upperLabelRange)
{
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