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
#ifndef __ELTWISE_H
#define __ELTWISE_H

#pragma once

#include "detail/eltwise.cuh"

namespace raft {
namespace linalg {

/**
 * @defgroup ScalarOps Scalar operations on the input buffer
 * @tparam InType data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in the input buffer
 * @param scalar the scalar used in the operations
 * @param len number of elements in the input buffer
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InType, typename IdxType, typename OutType = InType>
void scalarAdd(OutType* out, const InType* in, InType scalar, IdxType len, cudaStream_t stream)
{
  detail::scalarAdd(out, in, scalar, len, stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void scalarMultiply(OutType* out, const InType* in, InType scalar, IdxType len, cudaStream_t stream)
{
  detail::scalarMultiply(out, in, scalar, len, stream);
}
/** @} */

/**
 * @defgroup BinaryOps Element-wise binary operations on the input buffers
 * @tparam InType data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out the output buffer
 * @param in1 the first input buffer
 * @param in2 the second input buffer
 * @param len number of elements in the input buffers
 * @param stream cuda stream where to launch work
 * @{
 */
template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseAdd(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  detail::eltwiseAdd(out, in1, in2, len, stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseSub(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  detail::eltwiseSub(out, in1, in2, len, stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseMultiply(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  detail::eltwiseMultiply(out, in1, in2, len, stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivide(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  detail::eltwiseDivide(out, in1, in2, len, stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivideCheckZero(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  detail::eltwiseDivideCheckZero(out, in1, in2, len, stream);
}
/** @} */

};  // end namespace linalg
};  // end namespace raft

#endif