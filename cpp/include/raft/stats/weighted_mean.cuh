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

#pragma once

#include <raft/stats/detail/weighted_mean.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute the row-wise weighted mean of the input matrix
 *
 * @tparam Type the data type
 * @param mu the output mean vector
 * @param data the input matrix (assumed to be row-major)
 * @param weights per-column means
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream to launch work on
 */
template <typename Type>
void rowWeightedMean(
  Type* mu, const Type* data, const Type* weights, int D, int N, cudaStream_t stream)
{
  detail::rowWeightedMean(mu, data, weights, D, N, stream);
}

/**
 * @brief Compute the column-wise weighted mean of the input matrix
 *
 * @tparam Type the data type
 * @param mu the output mean vector
 * @param data the input matrix (assumed to be column-major)
 * @param weights per-column means
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream to launch work on
 */
template <typename Type>
void colWeightedMean(
  Type* mu, const Type* data, const Type* weights, int D, int N, cudaStream_t stream)
{
  detail::colWeightedMean(mu, data, weights, D, N, stream);
}
};  // end namespace stats
};  // end namespace raft
