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

#ifndef __SUM_H
#define __SUM_H

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/stats/detail/sum.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace stats {

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param output the output mean vector
 * @param input the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void sum(Type* output, const Type* input, IdxType D, IdxType N, bool rowMajor, cudaStream_t stream)
{
  detail::sum(output, input, D, N, rowMajor, stream);
}

/**
 * @defgroup stats_sum Sum
 * @{
 */

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam value_t the data type
 * @tparam idx_t Integer type used to for addressing
 * @tparam layout_t Layout type of the input matrix.
 * @param[in]  handle the raft handle
 * @param[in]  input the input matrix
 * @param[out] output the output mean vector
 */
template <typename value_t, typename idx_t, typename layout_t>
void sum(raft::resources const& handle,
         raft::device_matrix_view<const value_t, idx_t, layout_t> input,
         raft::device_vector_view<value_t, idx_t> output)
{
  constexpr bool is_row_major = std::is_same_v<layout_t, raft::row_major>;
  constexpr bool is_col_major = std::is_same_v<layout_t, raft::col_major>;
  static_assert(is_row_major || is_col_major,
                "sum: Layout must be either "
                "raft::row_major or raft::col_major (or one of their aliases)");
  RAFT_EXPECTS(input.extent(1) == output.extent(0), "Size mismatch between input and output");
  detail::sum(output.data_handle(),
              input.data_handle(),
              input.extent(1),
              input.extent(0),
              is_row_major,
              resource::get_cuda_stream(handle));
}

/** @} */  // end group stats_sum

};  // end namespace stats
};  // end namespace raft

#endif