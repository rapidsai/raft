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

#include <raft/core/operators.hpp>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/strided_reduction.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void reduce(OutType* dots,
            const InType* data,
            IdxType D,
            IdxType N,
            OutType init,
            bool rowMajor,
            bool alongRows,
            cudaStream_t stream,
            bool inplace           = false,
            MainLambda main_op     = raft::identity_op(),
            ReduceLambda reduce_op = raft::add_op(),
            FinalLambda final_op   = raft::identity_op())
{
  if (rowMajor && alongRows) {
    raft::linalg::coalescedReduction<InType, OutType, IdxType>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (rowMajor && !alongRows) {
    raft::linalg::stridedReduction<InType, OutType, IdxType>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if (!rowMajor && alongRows) {
    raft::linalg::stridedReduction<InType, OutType, IdxType>(
      dots, data, N, D, init, stream, inplace, main_op, reduce_op, final_op);
  } else {
    raft::linalg::coalescedReduction<InType, OutType, IdxType>(
      dots, data, N, D, init, stream, inplace, main_op, reduce_op, final_op);
  }
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
