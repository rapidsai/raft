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

#ifndef __MEAN_CENTER_H
#define __MEAN_CENTER_H

#pragma once

#include <raft/core/mdarray.hpp>
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
 * @brief Center the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam LayoutPolicy Layout type of the input matrix.
 * @tparam TPB threads per block of the cuda kernel launched
 * @param handle the raft handle
 * @param data input matrix
 * @param mu the mean vector
 * @param out the output mean-centered matrix
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 */
template <typename Type, typename IdxType, typename LayoutPolicy, int TPB = 256>
void meanCenter(const raft::handle_t& handle,
                const raft::device_matrix_view<const Type, IdxType, LayoutPolicy>& data,
                const raft::device_vector_view<const Type, IdxType, LayoutPolicy>& mu,
                const raft::device_matrix_view<Type, IdxType, LayoutPolicy>& out,
                bool bcastAlongRows)
{
  detail::meanCenter<Type, IdxType, TPB>(out.data(), data.data(), mu.data(), data.extent(1), data.extent(0),
    std::is_same_v<LayoutPolicy, raft::row_major>, bcastAlongRows, handle.get_stream());
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
 * @brief Add the input matrix wrt its mean
 * @tparam Type the data type
 * @tparam IdxType Integer type used for addressing
 * @tparam LayoutPolicy Layout type of the input matrix.
 * @tparam TPB threads per block of the cuda kernel launched
 * @param handle the raft handle
 * @param data input matrix
 * @param mu the mean vector
 * @param out the output mean-added matrix
 * @param bcastAlongRows whether to broadcast vector along rows or columns
 */
template <typename Type, typename IdxType, typename LayoutPolicy = raft::row_major, int TPB = 256>
void meanAdd(const raft::handle_t& handle,
  const raft::device_matrix_view<const Type, IdxType, LayoutPolicy>& data,
  const raft::device_vector_view<const Type, IdxType, LayoutPolicy>& mu,
  const raft::device_matrix_view<Type, IdxType, LayoutPolicy>& out,
  bool bcastAlongRows)
{
  detail::meanAdd<Type, IdxType, TPB>(out.data(), data.data(), mu.data(), data.extent(1), data.extent(0),
    std::is_same_v<LayoutPolicy, raft::row_major>, bcastAlongRows, handle.get_stream());
}
};  // end namespace stats
};  // end namespace raft

#endif