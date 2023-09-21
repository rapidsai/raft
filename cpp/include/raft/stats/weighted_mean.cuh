/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
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

/**
 * @defgroup stats_weighted_mean Weighted Mean
 * @{
 */

/**
 * @brief Compute the weighted mean of the input matrix with a
 * vector of weights, along rows or along columns
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data the input matrix of size nrows * ncols
 * @param[in]  weights weight of size ncols if along_row is true, else of size nrows
 * @param[out] mu the output mean vector of size nrows if along_row is true, else of size ncols
 * @param[in]  along_rows whether to reduce along rows or columns
 */
template <typename value_t, typename idx_t, typename layout_t>
void weighted_mean(raft::resources const& handle,
                   raft::device_matrix_view<const value_t, idx_t, layout_t> data,
                   raft::device_vector_view<const value_t, idx_t> weights,
                   raft::device_vector_view<value_t, idx_t> mu,
                   bool along_rows)
{
  constexpr bool is_row_major = std::is_same_v<layout_t, raft::row_major>;
  constexpr bool is_col_major = std::is_same_v<layout_t, raft::col_major>;
  static_assert(is_row_major || is_col_major,
                "weighted_mean: Layout must be either "
                "raft::row_major or raft::col_major (or one of their aliases)");
  auto mean_vec_size = along_rows ? data.extent(0) : data.extent(1);
  auto weight_size   = along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(weights.extent(0) == weight_size,
               "Size mismatch between weights and expected weight_size");
  RAFT_EXPECTS(mu.extent(0) == mean_vec_size,
               "Size mismatch between mu and expected mean_vec_size");

  detail::weightedMean(mu.data_handle(),
                       data.data_handle(),
                       weights.data_handle(),
                       data.extent(1),
                       data.extent(0),
                       is_row_major,
                       along_rows,
                       resource::get_cuda_stream(handle));
}

/**
 * @brief Compute the row-wise weighted mean of the input matrix with a
 * vector of column weights
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data the input matrix of size nrows * ncols
 * @param[in]  weights weight vector of size ncols
 * @param[out] mu the output mean vector of size nrows
 */
template <typename value_t, typename idx_t, typename layout_t>
void row_weighted_mean(raft::resources const& handle,
                       raft::device_matrix_view<const value_t, idx_t, layout_t> data,
                       raft::device_vector_view<const value_t, idx_t> weights,
                       raft::device_vector_view<value_t, idx_t> mu)
{
  weighted_mean(handle, data, weights, mu, true);
}

/**
 * @brief Compute the column-wise weighted mean of the input matrix with a
 * vector of row weights
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data the input matrix of size nrows * ncols
 * @param[in]  weights weight vector of size nrows
 * @param[out] mu the output mean vector of size ncols
 */
template <typename value_t, typename idx_t, typename layout_t>
void col_weighted_mean(raft::resources const& handle,
                       raft::device_matrix_view<const value_t, idx_t, layout_t> data,
                       raft::device_vector_view<const value_t, idx_t> weights,
                       raft::device_vector_view<value_t, idx_t> mu)
{
  weighted_mean(handle, data, weights, mu, false);
}

/** @} */  // end group stats_weighted_mean

};  // end namespace stats
};  // end namespace raft

#endif