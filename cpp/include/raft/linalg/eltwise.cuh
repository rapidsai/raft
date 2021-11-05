/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include "detail/functional.cuh"

#include "binary_op.cuh"
#include "unary_op.cuh"

namespace raft {
namespace linalg {

using detail::adds_scalar;

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
void scalarAdd(OutType *out, const InType *in, InType scalar, IdxType len,
               cudaStream_t stream) {
  raft::linalg::unaryOp(
    out, in, len, adds_scalar<in, out>(scalar),
    stream);
}

using detail::multiplies_scalar;

template <typename InType, typename IdxType, typename OutType = InType>
void scalarMultiply(OutType *out, const InType *in, InType scalar, IdxType len,
                    cudaStream_t stream) {
  raft::linalg::unaryOp(
    out, in, len, multiplies_scalar<in, out>(scalar),
    stream);
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
void eltwiseAdd(OutType *out, const InType *in1, const InType *in2, IdxType len,
                cudaStream_t stream) {
  binaryOp(
    out, in1, in2, len, thrust::plus<InType>(),
    stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseSub(OutType *out, const InType *in1, const InType *in2, IdxType len,
                cudaStream_t stream) {
  binaryOp(
    out, in1, in2, len, thrust::minus<InType>(),
    stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseMultiply(OutType *out, const InType *in1, const InType *in2,
                     IdxType len, cudaStream_t stream) {
  binaryOp(
    out, in1, in2, len, thrust::multiplies<InType>(),
    stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivide(OutType *out, const InType *in1, const InType *in2,
                   IdxType len, cudaStream_t stream) {
  binaryOp(
    out, in1, in2, len, thrust::divides<InType>(),
    stream);
}

using detail::divides_check_zero;

template <typename InType, typename IdxType, typename OutType = InType>
void eltwiseDivideCheckZero(OutType *out, const InType *in1, const InType *in2,
                            IdxType len, cudaStream_t stream) {
  binaryOp(
    out, in1, in2, len,
    divides_check_zero<InType, OutType>(),
    stream);
}
/** @} */

};  // end namespace linalg
};  // end namespace raft
