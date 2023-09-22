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

#ifndef __MEAN_CENTER_H
#define __MEAN_CENTER_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/mean_center.cuh>

namespace raft {
namespace stats {

/**
 * @brief Center the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-centered matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void meanCenter(Type* out,
                const Type* data,
                const Type* mu,
                IdxType D,
                IdxType N,
                bool rowMajor,
                bool bcastAlongRows,
                cudaStream_t stream)
{
  detail::meanCenter<Type, IdxType, TPB>(out, data, mu, D, N, rowMajor, bcastAlongRows, stream);
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output mean-added matrix
 * @param data input matrix
 * @param mu the mean vector
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void meanAdd(Type* out,
             const Type* data,
             const Type* mu,
             IdxType D,
             IdxType N,
             bool rowMajor,
             bool bcastAlongRows,
             cudaStream_t stream)
{
  detail::meanAdd<Type, IdxType, TPB>(out, data, mu, D, N, rowMajor, bcastAlongRows, stream);
}

/**
 * @defgroup stats_mean_center Mean Center
 * @{
 */

/**
 * @brief Center the input matrix wrt its mean
 * @tparam value_t the data type
 * @tparam idx_t index type
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  data input matrix of size nrows * ncols
 * @param[in]  mu the mean vector of size ncols if bcast_along_rows else nrows
 * @param[out] out the output mean-centered matrix
 * @param[in]  bcast_along_rows whether to broadcast vector along rows or columns
 */
template <typename value_t, typename idx_t, typename layout_t>
void mean_center(raft::resources const& handle,
                 raft::device_matrix_view<const value_t, idx_t, layout_t> data,
                 raft::device_vector_view<const value_t, idx_t> mu,
                 raft::device_matrix_view<value_t, idx_t, layout_t> out,
                 bool bcast_along_rows)
{
  static_assert(
    std::is_same_v<layout_t, raft::row_major> || std::is_same_v<layout_t, raft::col_major>,
    "Data layout not supported");
  auto mean_vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);
  RAFT_EXPECTS(out.extents() == data.extents(), "Size mismatch");
  RAFT_EXPECTS(mean_vec_size == mu.extent(0), "Size mismatch between data and mu");
  RAFT_EXPECTS(out.is_exhaustive(), "out must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  detail::meanCenter<value_t, idx_t>(out.data_handle(),
                                     data.data_handle(),
                                     mu.data_handle(),
                                     data.extent(1),
                                     data.extent(0),
                                     std::is_same_v<layout_t, raft::row_major>,
                                     bcast_along_rows,
                                     resource::get_cuda_stream(handle));
}

/**
 * @brief Add the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam idx_t index type
 * @tparam layout_t Layout type of the input matrix.
 * @tparam TPB threads per block of the cuda kernel launched
 * @param[in]  handle the raft handle
 * @param[in]  data input matrix of size nrows * ncols
 * @param[in]  mu the mean vector of size ncols if bcast_along_rows else nrows
 * @param[out] out the output mean-centered matrix
 * @param[in]  bcast_along_rows whether to broadcast vector along rows or columns
 */
template <typename value_t, typename idx_t, typename layout_t>
void mean_add(raft::resources const& handle,
              raft::device_matrix_view<const value_t, idx_t, layout_t> data,
              raft::device_vector_view<const value_t, idx_t> mu,
              raft::device_matrix_view<value_t, idx_t, layout_t> out,
              bool bcast_along_rows)
{
  static_assert(
    std::is_same_v<layout_t, raft::row_major> || std::is_same_v<layout_t, raft::col_major>,
    "Data layout not supported");
  auto mean_vec_size = bcast_along_rows ? data.extent(1) : data.extent(0);
  RAFT_EXPECTS(out.extents() == data.extents(), "Size mismatch");
  RAFT_EXPECTS(mean_vec_size == mu.extent(0), "Size mismatch between data and mu");
  RAFT_EXPECTS(out.is_exhaustive(), "out must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "data must be contiguous");
  detail::meanAdd<value_t, idx_t>(out.data_handle(),
                                  data.data_handle(),
                                  mu.data_handle(),
                                  data.extent(1),
                                  data.extent(0),
                                  std::is_same_v<layout_t, raft::row_major>,
                                  bcast_along_rows,
                                  resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_mean_center

};  // end namespace stats
};  // end namespace raft

#endif