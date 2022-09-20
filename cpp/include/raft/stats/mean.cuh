/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#ifndef __MEAN_H
#define __MEAN_H

#pragma once

#include <raft/core/mdarray.hpp>
#include <raft/handle.hpp>
#include <raft/stats/detail/mean.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute mean of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type: the data type
 * @tparam IdxType Integer type used to for addressing
 * @param mu: the output mean vector
 * @param data: the input matrix
 * @param D: number of columns of data
 * @param N: number of rows of data
 * @param sample: whether to evaluate sample mean or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor: whether the input data is row or col major
 * @param stream: cuda stream
 */
template <typename Type, typename IdxType = int>
void mean(
  Type* mu, const Type* data, IdxType D, IdxType N, bool sample, bool rowMajor, cudaStream_t stream)
{
  detail::mean(mu, data, D, N, sample, rowMajor, stream);
}

/**
 * @brief Compute mean of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam DataT the data type
 * @tparam IdxType index type
 * @tparam LayoutPolicy Layout type of the input matrix.
 * @param handle the raft handle
 * @param mu: the output mean vector
 * @param data: the input matrix
 * @param sample: whether to evaluate sample mean or not. In other words, whether
 *   to normalize the output using N-1 or N, for true or false, respectively
 */
template <typename DataT,
          typename IdxType = int,
          typename LayoutPolicy>
void mean(const raft::handle_t& handle,
          raft::device_vector_view<DataT, IdxType> mu,
          raft::device_matrix_view<const DataT, IdxType, LayoutPolicy> data,
          bool sample)
{
  RAFT_EXPECTS(data.extent(0) == mu.size(), "Size mismatch betwen data and mu");
  RAFT_EXPECTS(mu.is_exhaustive(), "mu must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  detail::mean(mu.data_handle(),
               data.data_handle(),
               data.extent(1),
               data.extent(0),
               sample,
               std::is_same_v<LayoutPolicy, raft::row_major>,
               handle.get_stream());
}

};  // namespace stats
};  // namespace raft

#endif