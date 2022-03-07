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

#ifndef __WEIGHTED_MEAN_H
#define __WEIGHTED_MEAN_H

#pragma once

#include <raft/stats/detail/weighted_mean.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute the weighted mean of the input matrix with a
 * vector of weights, along rows or along columns
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param mu the output mean vector
 * @param data the input matrix
 * @param weights weight of size D if along_row is true, else of size N
 * @param D number of columns of data
 * @param N number of rows of data
 * @param row_major data input matrix is row-major or not
 * @param along_rows whether to reduce along rows or columns
 * @param stream cuda stream to launch work on
 */
template <typename Type, typename IdxType = int>
void weightedMean(Type* mu,
                  const Type* data,
                  const Type* weights,
                  IdxType D,
                  IdxType N,
                  bool row_major,
                  bool along_rows,
                  cudaStream_t stream)
{
  detail::weightedMean(mu, data, weights, D, N, row_major, along_rows, stream);
}

/**
 * @brief Compute the row-wise weighted mean of the input matrix with a
 * vector of column weights
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param mu the output mean vector
 * @param data the input matrix (assumed to be row-major)
 * @param weights per-column means
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream to launch work on
 */
template <typename Type, typename IdxType = int>
void rowWeightedMean(
  Type* mu, const Type* data, const Type* weights, IdxType D, IdxType N, cudaStream_t stream)
{
  weightedMean(mu, data, weights, D, N, true, true, stream);
}

/**
 * @brief Compute the column-wise weighted mean of the input matrix with a
 * vector of row weights
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param mu the output mean vector
 * @param data the input matrix (assumed to be row-major)
 * @param weights per-row means
 * @param D number of columns of data
 * @param N number of rows of data
 * @param stream cuda stream to launch work on
 */
template <typename Type, typename IdxType = int>
void colWeightedMean(
  Type* mu, const Type* data, const Type* weights, IdxType D, IdxType N, cudaStream_t stream)
{
  weightedMean(mu, data, weights, D, N, true, false, stream);
}
};  // end namespace stats
};  // end namespace raft

#endif