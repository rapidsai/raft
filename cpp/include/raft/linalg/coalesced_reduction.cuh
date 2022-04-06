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
                        int D,
                        int N,
                        OutType init,
                        cudaStream_t stream,
                        bool inplace           = false,
                        MainLambda main_op     = raft::Nop<InType, IdxType>(),
                        ReduceLambda reduce_op = raft::Sum<OutType>(),
                        FinalLambda final_op   = raft::Nop<OutType>())
{
  detail::coalescedReduction(dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
}

};  // end namespace linalg
};  // end namespace raft

#endif