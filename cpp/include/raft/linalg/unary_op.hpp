/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include "detail/unary_op.cuh"

namespace raft {
namespace linalg {

/**
 * @brief perform element-wise unary operation in the input array
 * @tparam InType input data-type
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType output data-type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in the input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val);`
 */
template <typename InType,
          typename Lambda,
          typename IdxType = int,
          typename OutType = InType,
          int TPB          = 256>
void unaryOp(OutType* out, const InType* in, IdxType len, Lambda op, cudaStream_t stream)
{
  detail::unaryOpCaller(out, in, len, op, stream);
}

/**
 * @brief Perform an element-wise unary operation into the output array
 *
 * Compared to `unaryOp()`, this method does not do any reads from any inputs
 *
 * @tparam OutType output data-type
 * @tparam Lambda  the device-lambda performing the actual operation
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB     threads-per-block in the final kernel launched
 *
 * @param[out] out    the output array [on device] [len = len]
 * @param[in]  len    number of elements in the input array
 * @param[in]  op     the device-lambda which must be of the form:
 *                    `void func(OutType* outLocationOffset, IdxType idx);`
 *                    where outLocationOffset will be out + idx.
 * @param[in]  stream cuda stream where to launch work
 */
template <typename OutType, typename Lambda, typename IdxType = int, int TPB = 256>
void writeOnlyUnaryOp(OutType* out, IdxType len, Lambda op, cudaStream_t stream)
{
  detail::writeOnlyUnaryOpCaller(out, len, op, stream);
}

};  // end namespace linalg
};  // end namespace raft
