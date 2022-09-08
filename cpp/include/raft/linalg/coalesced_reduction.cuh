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
#ifndef __COALESCED_REDUCTION_H
#define __COALESCED_REDUCTION_H

#pragma once

#include "detail/coalesced_reduction.cuh"

#include <raft/core/handle.hpp>
#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Compute reduction of the input matrix along the leading dimension
 *
 * @tparam InType the data type of the input
 * @tparam OutType the data type of the output (as well as the data type for
 *  which reduction is performed)
 * @tparam IdxType data type of the indices of the array
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param dots the output reduction vector
 * @param data the input matrix
 * @param D leading dimension of data
 * @param N second dimension data
 * @param init initial value to use for the reduction
 * @param main_op elementwise operation to apply before reduction
 * @param reduce_op binary reduction operation
 * @param final_op elementwise operation to apply before storing results
 * @param inplace reduction result added inplace or overwrites old values?
 * @param stream cuda stream where to launch work
 */
template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::Nop<InType, IdxType>,
          typename ReduceLambda = raft::Sum<OutType>,
          typename FinalLambda  = raft::Nop<OutType>>
void coalescedReduction(OutType* dots,
                        const InType* data,
                        IdxType D,
                        IdxType N,
                        OutType init,
                        cudaStream_t stream,
                        bool inplace           = false,
                        MainLambda main_op     = raft::Nop<InType, IdxType>(),
                        ReduceLambda reduce_op = raft::Sum<OutType>(),
                        FinalLambda final_op   = raft::Nop<OutType>())
{
  detail::coalescedReduction(dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
}

/**
 * @defgroup coalesced_reduction Coalesced Memory Access Reductions
 * For reducing along rows for col-major and along columns for row-major
 * @{
 */

/**
 * @brief Compute reduction of the input matrix along the leading dimension
 *
 * @tparam InElementType the input data-type of underlying raft::matrix_view
 * @tparam LayoutPolicy The layout of Input/Output (row or col major)
 * @tparam OutElementType the output data-type of underlying raft::matrix_view and reduction
 * @tparam IndexType Integer type used to for addressing
 * @tparam MainLambda Unary lambda applied while acculumation (eg: L1 or L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*MainLambda)(InType, IdxType);</pre>
 * @tparam ReduceLambda Binary lambda applied for reduction (eg: addition(+) for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*ReduceLambda)(OutType);</pre>
 * @tparam FinalLambda the final lambda applied before STG (eg: Sqrt for L2 norm)
 * It must be a 'callable' supporting the following input and output:
 * <pre>OutType (*FinalLambda)(OutType);</pre>
 * @param handle raft::handle_t
 * @param dots Output of type raft::device_matrix_view
 * @param data Input of type raft::device_matrix_view
 * @param init initial value to use for the reduction
 * @param main_op elementwise operation to apply before reduction
 * @param reduce_op binary reduction operation
 * @param final_op elementwise operation to apply before storing results
 * @param inplace reduction result added inplace or overwrites old values?
 */
template <typename InElementType,
          typename LayoutPolicy,
          typename OutElementType = InElementType,
          typename IndexType      = std::uint32_t,
          typename MainLambda     = raft::Nop<InElementType>,
          typename ReduceLambda   = raft::Sum<OutElementType>,
          typename FinalLambda    = raft::Nop<OutElementType>>
void coalesced_reduction(
  const raft::handle_t& handle,
  raft::device_matrix_view<OutElementType, IndexType, LayoutPolicy> dots,
  const raft::device_matrix_view<InElementType, IndexType, LayoutPolicy> data,
  OutElementType init,
  bool inplace           = false,
  MainLambda main_op     = raft::Nop<InElementType>(),
  ReduceLambda reduce_op = raft::Sum<OutElementType>(),
  FinalLambda final_op   = raft::Nop<OutElementType>())
{
  RAFT_EXPECTS(dots.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(data.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(dots.size() == data.size(), "Size mismatch between Output and Input");

  if constexpr (std::is_same_v<LayoutPolicy, raft::row_major>) {
    coalescedReduction(dots.data_handle(),
                       data.data_handle(),
                       data.extent(1),
                       data.extent(0),
                       init,
                       handle.get_stream(),
                       inplace,
                       main_op,
                       reduce_op,
                       final_op);
  } else if constexpr (std::is_same_v<LayoutPolicy, raft::col_major>) {
    coalescedReduction(dots.data_handle(),
                       data.data_handle(),
                       data.extent(0),
                       data.extent(1),
                       init,
                       handle.get_stream(),
                       inplace,
                       main_op,
                       reduce_op,
                       final_op);
  }
}

/** @} */  // end of group coalesced_reduction

};  // end namespace linalg
};  // end namespace raft

#endif