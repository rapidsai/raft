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

#include "detail/binary_op.cuh"

#include <raft/cuda_utils.cuh>

namespace raft {
namespace linalg {

/**
 * @brief perform element-wise binary operation on the input arrays
 * @tparam InType input data-type
 * @tparam Lambda the device-lambda performing the actual operation
 * @tparam OutType output data-type
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads-per-block in the final kernel launched
 * @param out the output array
 * @param in1 the first input array
 * @param in2 the second input array
 * @param len number of elements in the input array
 * @param op the device-lambda
 * @param stream cuda stream where to launch work
 * @note Lambda must be a functor with the following signature:
 *       `OutType func(const InType& val1, const InType& val2);`
 */
template <typename InType, typename Lambda, typename OutType = InType,
          typename IdxType = int, int TPB = 256>
void binaryOp(OutType *out, const InType *in1, const InType *in2, IdxType len,
              Lambda op, cudaStream_t stream) {
  constexpr auto maxSize =
    sizeof(InType) > sizeof(OutType) ? sizeof(InType) : sizeof(OutType);
  size_t bytes = len * maxSize;
  uint64_t in1Addr = uint64_t(in1);
  uint64_t in2Addr = uint64_t(in2);
  uint64_t outAddr = uint64_t(out);
  if (16 / maxSize && bytes % 16 == 0 &&
      detail::addressAligned(in1Addr, in2Addr, outAddr, 16)) {
        detail::binaryOpImpl<InType, 16 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (8 / maxSize && bytes % 8 == 0 &&
             addressAligned(in1Addr, in2Addr, outAddr, 8)) {
              detail::binaryOpImpl<InType, 8 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (4 / maxSize && bytes % 4 == 0 &&
             addressAligned(in1Addr, in2Addr, outAddr, 4)) {
              detail:: binaryOpImpl<InType, 4 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (2 / maxSize && bytes % 2 == 0 &&
             addressAligned(in1Addr, in2Addr, outAddr, 2)) {
              detail::binaryOpImpl<InType, 2 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (1 / maxSize) {
    detail::binaryOpImpl<InType, 1 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else {
    detail::binaryOpImpl<InType, 1, Lambda, IdxType, OutType, TPB>(out, in1, in2, len,
                                                           op, stream);
  }
}

};  // end namespace linalg
};  // end namespace raft
