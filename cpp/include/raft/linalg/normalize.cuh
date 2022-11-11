/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "detail/normalize.cuh"

namespace raft {
namespace linalg {

/**
 * @brief Divide rows by their norm defined by main_op, reduce_op and fin_op
 *
 * @tparam ElementType Input/Output data type
 * @tparam IndexType Integer type used to for addressing
 * @tparam MainLambda Type of main_op
 * @tparam ReduceLambda Type of reduce_op
 * @tparam FinalLambda Type of fin_op
 * @param[in] handle raft::handle_t
 * @param[in] in the input raft::device_matrix_view
 * @param[out] out the output raft::device_matrix_view
 * @param[in] main_op Operation to apply to the elements before reducing them (e.g square for L2)
 * @param[in] reduce_op Operation to reduce a pair of elements (e.g sum for L2)
 * @param[in] fin_op Operation to apply once to the reduction result to finalize the norm
 *                   computation (e.g sqrt for L2)
 */
template <typename ElementType,
          typename IndexType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
void row_normalize(const raft::handle_t& handle,
                   raft::device_matrix_view<const ElementType, IndexType, row_major> in,
                   raft::device_matrix_view<ElementType, IndexType, row_major> out,
                   MainLambda main_op,
                   ReduceLambda reduce_op,
                   FinalLambda fin_op)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(in), "Input must be contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(out), "Output must be contiguous");
  RAFT_EXPECTS(in.extent(0) == out.extent(0),
               "The number of rows of the input and output should be equal");
  RAFT_EXPECTS(in.extent(1) == out.extent(1),
               "The number of columns of the input and output should be equal");

  detail::coalesced_normalize(out.data_handle(),
                              in.data_handle(),
                              in.extent(1),
                              in.extent(0),
                              handle.get_stream(),
                              main_op,
                              reduce_op,
                              fin_op);
}

/**
 * @brief Divide rows by their norm.
 *
 * @tparam ElementType Input/Output data type
 * @tparam IndexType Integer type used to for addressing
 * @param[in] handle raft::handle_t
 * @param[in] in the input raft::device_matrix_view
 * @param[out] out the output raft::device_matrix_view
 * @param[in] type the type of norm to be applied
 */
template <typename ElementType, typename IndexType>
void row_normalize(const raft::handle_t& handle,
                   raft::device_matrix_view<const ElementType, IndexType, row_major> in,
                   raft::device_matrix_view<ElementType, IndexType, row_major> out,
                   NormType norm_type)
{
  switch (norm_type) {
    case L1Norm:
      row_normalize(handle,
                    in,
                    out,
                    raft::L1Op<ElementType>(),
                    raft::Sum<ElementType>(),
                    raft::Nop<ElementType>());
      break;
    case L2Norm:
      row_normalize(handle,
                    in,
                    out,
                    raft::L2Op<ElementType>(),
                    raft::Sum<ElementType>(),
                    raft::SqrtOp<ElementType>());
      break;
    case LinfNorm:
      row_normalize(handle,
                    in,
                    out,
                    raft::L1Op<ElementType>(),
                    raft::Max<ElementType>(),
                    raft::Nop<ElementType>());
      break;
    default: THROW("Unsupported norm type: %d", norm_type);
  }
}

}  // namespace linalg
}  // namespace raft
