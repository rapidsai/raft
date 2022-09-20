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

#ifndef __ENTROPY_H
#define __ENTROPY_H

#pragma once
#include <raft/core/mdarray.hpp>
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
 * @brief Function to calculate entropy
 * <a href="https://en.wikipedia.org/wiki/Entropy_(information_theory)">more info on entropy</a>
 *
 * @tparam DataT data type
 * @tparam IdxT index type
 * @param handle the raft handle
 * @param clusterArray: the array of classes of type DataT
 * @param lowerLabelRange: the lower bound of the range of labels
 * @param upperLabelRange: the upper bound of the range of labels
 * @return the entropy score
 */
template <typename DataT, typename IdxType>
double entropy(
  const raft::handle_t& handle,
  raft::device_vector_view<const DataT, IdxType> clusterArray,
  const DataT lowerLabelRange,
  const DataT upperLabelRange)
{
  RAFT_EXPECTS(clusterArray.is_exhaustive(), "clusterArray must be contiguous");
  return detail::entropy(clusterArray.data_handle(),
                         clusterArray.size(),
                         lowerLabelRange,
                         upperLabelRange,
                         handle.get_stream());
}
};  // end namespace stats
};  // end namespace raft

#endif