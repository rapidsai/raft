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

#ifndef __COV_H
#define __COV_H

#pragma once

#include <raft/core/mdarray.hpp>
#include <raft/stats/detail/cov.cuh>
namespace raft {
namespace stats {
/**
 * @brief Compute covariance of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @param covar the output covariance matrix
 * @param data the input matrix (this will get mean-centered at the end!)
 * @param mu mean vector of the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample covariance or not. In other words,
 * whether to normalize the output using N-1 or N, for true or false,
 * respectively
 * @param rowMajor whether the input data is row or col major
 * @param stable whether to run the slower-but-numerically-stable version or not
 * @param handle cublas handle
 * @param stream cuda stream
 * @note if stable=true, then the input data will be mean centered after this
 * function returns!
 */
template <typename Type>
void cov(const raft::handle_t& handle,
         Type* covar,
         Type* data,
         const Type* mu,
         std::size_t D,
         std::size_t N,
         bool sample,
         bool rowMajor,
         bool stable,
         cudaStream_t stream)
{
  detail::cov(handle, covar, data, mu, D, N, sample, rowMajor, stable, stream);
}

/**
 * @brief Compute covariance of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxT the index type
 * @tparam LayoutPolicy Layout type of the input data.
 * @tparam AccessorPolicy Accessor for the input and output, must be valid accessor on
 *                        device.
 * @param handle the raft handle
 * @param covar the output covariance matrix
 * @param data the input matrix (this will get mean-centered at the end!)
 * @param mu mean vector of the input matrix
 * @param sample whether to evaluate sample covariance or not. In other words,
 * whether to normalize the output using N-1 or N, for true or false,
 * respectively
 * @param stable whether to run the slower-but-numerically-stable version or not
 * @note if stable=true, then the input data will be mean centered after this
 * function returns!
 */
template <typename Type, typename IdxType, typename LayoutPolicy, typename AccessorPolicy>
void cov(const raft::handle_t& handle,
         raft::mdspan<Type, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> covar,
         raft::mdspan<Type, raft::matrix_extent<IdxType>, LayoutPolicy, AccessorPolicy> data,
         raft::device_mdspan<const Type, raft::vector_extent<IdxType>, LayoutPolicy> mu,
         bool sample,
         bool stable)
{
  detail::cov(handle,
              covar.data_handle(),
              data.data_handle(),
              mu.data_handle(),
              data.extent(1),
              data.extent(0),
              std::is_same_v<LayoutPolicy, raft::row_major>,
              sample,
              stable,
              handle.get_stream());
}
};  // end namespace stats
};  // end namespace raft

#endif