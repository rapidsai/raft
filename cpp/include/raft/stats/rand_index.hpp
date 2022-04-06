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
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __RAND_INDEX_H
#define __RAND_INDEX_H

#pragma once

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

};  // end namespace stats
};  // end namespace raft

#endif