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

#include "functional.cuh"

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InType, typename IdxType, typename OutType = InType>
void scalarAdd(OutType* out, const InType* in, InType scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::ScalarAdd<InType, OutType>(scalar), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void scalarMultiply(OutType* out, const InType* in, InType scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::ScalarMul<InType, OutType>(scalar), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseAdd(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::Sum<InType>(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseSub(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::Subtract<InType>(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseMultiply(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::Multiply<InType>(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivide(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::Divide<InType>(), stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivideCheckZero(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::ScalarDivCheckZero<InType, OutType>(), stream);
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
