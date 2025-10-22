/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename math_t, typename IdxType = int>
void multiplyScalar(
  math_t* out, const math_t* in, const math_t scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::mul_const_op<math_t>{scalar}, stream);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
