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

#include <raft/core/device_mdspan.hpp>
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
 * @brief Compute the weighted mean of the input matrix with a
 * vector of weights, along rows or along columns
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param handle the raft handle
 * @param data the input matrix of size nrows * ncols
 * @param weights weight of size ncols if along_row is true, else of size nrows
 * @param mu the output mean vector of size ncols if along_row is true, else of size nrows
 * @param along_rows whether to reduce along rows or columns
 */
template <typename value_t, typename idx_t = int, typename layout_t>
void weighted_mean(const raft::handle_t& handle,
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
  auto mean_vec_size = along_rows ? data.extent(1) : data.extent(0);

  RAFT_EXPECTS(weights.extent(0) == mean_vec_size,
               "Size mismatch betwen weights and mean_vec_size");
  RAFT_EXPECTS(mu.extent(0) == mean_vec_size, "Size mismatch betwen mu and mean_vec_size");
  RAFT_EXPECTS(weights.is_exhaustive(), "weights must be contiguous");
  RAFT_EXPECTS(mu.is_exhaustive(), "mu must be contiguous");

  detail::weightedMean(mu.data_handle(),
                       data.data_handle(),
                       weights.data_handle(),
                       data.extent(1),
                       data.extent(0),
                       is_row_major,
                       along_rows,
                       handle.get_stream());
}

/**
 * @brief Compute the row-wise weighted mean of the input matrix with a
 * vector of column weights
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @param handle the raft handle
 * @param data the input matrix of size nrows * ncols
 * @param weights per-col weight
 * @param mu the output mean vector of size ncols
 */
template <typename value_t, typename idx_t = int>
void rowWeightedMean(const raft::handle_t& handle,
                     raft::device_matrix_view<const value_t, idx_t, raft::row_major> data,
                     raft::device_vector_view<const value_t, idx_t> weights,
                     raft::device_vector_view<value_t, idx_t> mu)
{
  weightedMean(handle, data, weights, mu, true);
}

/**
 * @brief Compute the column-wise weighted mean of the input matrix with a
 * vector of row weights
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @param handle the raft handle
 * @param data the input matrix of size nrows * ncols
 * @param weights per-row weight
 * @param mu the output mean vector of size nrows
 */
template <typename value_t, typename idx_t>
void colWeightedMean(const raft::handle_t& handle,
                     raft::device_matrix_view<const value_t, idx_t, raft::row_major> data,
                     raft::device_vector_view<const value_t, idx_t> weights,
                     raft::device_vector_view<value_t, idx_t> mu)
{
  weightedMean(handle, data, weights, mu, false);
}
};  // end namespace stats
};  // end namespace raft

#endif