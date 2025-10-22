/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/linalg/strided_reduction.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <bool rowMajor,
          bool alongRows,
          typename InType,
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
            cudaStream_t stream,
            bool inplace           = false,
            MainLambda main_op     = raft::identity_op(),
            ReduceLambda reduce_op = raft::add_op(),
            FinalLambda final_op   = raft::identity_op())
{
  if constexpr (rowMajor && alongRows) {
    raft::linalg::coalescedReduction<InType, OutType, IdxType>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if constexpr (rowMajor && !alongRows) {
    raft::linalg::stridedReduction<InType, OutType, IdxType>(
      dots, data, D, N, init, stream, inplace, main_op, reduce_op, final_op);
  } else if constexpr (!rowMajor && alongRows) {
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
