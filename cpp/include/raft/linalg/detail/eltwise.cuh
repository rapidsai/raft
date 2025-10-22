/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InType, typename IdxType, typename OutType = InType>
void scalarAdd(OutType* out, const InType* in, InType scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::add_const_op<InType>(scalar), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void scalarMultiply(OutType* out, const InType* in, InType scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::mul_const_op<InType>(scalar), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseAdd(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::add_op(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseSub(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::sub_op(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseMultiply(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::mul_op(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivide(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::div_op(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivideCheckZero(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::div_checkzero_op(), stream);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
