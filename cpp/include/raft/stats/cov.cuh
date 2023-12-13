/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
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
void cov(raft::resources const& handle,
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
 * @defgroup stats_cov Covariance Matrix Construction
 * @{
 */

/**
 * @brief Compute covariance of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam value_t the data type
 * @tparam idx_t the index type
 * @tparam layout_t Layout type of the input data.
 * @param[in]  handle the raft handle
 * @param[in]  data the input matrix (this will get mean-centered at the end!)
 * (length = nrows * ncols)
 * @param[in]  mu mean vector of the input matrix (length = ncols)
 * @param[out] covar the output covariance matrix (length = ncols * ncols)
 * @param[in]  sample whether to evaluate sample covariance or not. In other words,
 * whether to normalize the output using N-1 or N, for true or false,
 * respectively
 * @param[in]  stable whether to run the slower-but-numerically-stable version or not
 * @note if stable=true, then the input data will be mean centered after this
 * function returns!
 */
template <typename value_t, typename idx_t, typename layout_t>
void cov(raft::resources const& handle,
         raft::device_matrix_view<value_t, idx_t, layout_t> data,
         raft::device_vector_view<const value_t, idx_t> mu,
         raft::device_matrix_view<value_t, idx_t, layout_t> covar,
         bool sample,
         bool stable)
{
  static_assert(
    std::is_same_v<layout_t, raft::row_major> || std::is_same_v<layout_t, raft::col_major>,
    "Data layout not supported");
  RAFT_EXPECTS(data.extent(1) == covar.extent(0) && data.extent(1) == covar.extent(1),
               "Size mismatch");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  RAFT_EXPECTS(covar.is_exhaustive(), "covar must be contiguous");
  RAFT_EXPECTS(mu.is_exhaustive(), "mu must be contiguous");

  detail::cov(handle,
              covar.data_handle(),
              data.data_handle(),
              mu.data_handle(),
              data.extent(1),
              data.extent(0),
              std::is_same_v<layout_t, raft::row_major>,
              sample,
              stable,
              resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_cov

};  // end namespace stats
};  // end namespace raft

#endif