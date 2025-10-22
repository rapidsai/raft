/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/host_mdspan.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InT, typename OutT = InT, typename IdxType = int>
void divideScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::div_const_op<InT>(scalar), stream);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
