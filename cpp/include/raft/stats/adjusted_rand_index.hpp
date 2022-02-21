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
/**
 * @file adjusted_rand_index.hpp
 * @brief The adjusted Rand index is the corrected-for-chance version of the Rand index.
 * Such a correction for chance establishes a baseline by using the expected similarity
 * of all pair-wise comparisons between clusterings specified by a random model.
 */

#pragma once

#include <raft/stats/detail/adjusted_rand_index.cuh>

namespace raft {
namespace stats {

/**
 * @brief Function to calculate Adjusted RandIndex as described
 *        <a href="https://en.wikipedia.org/wiki/Rand_index">here</a>
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

};  // end namespace stats
};  // end namespace raft
