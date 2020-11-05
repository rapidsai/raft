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

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/vectorized.cuh>

namespace raft {
namespace linalg {

template <typename InType, int VecLen, typename Lambda, typename OutType,
          typename IdxType>
__global__ void unary_op_kernel(OutType *out, const InType *in, IdxType len,
                                Lambda op) {
  using in_vec_t = TxN_t<InType, VecLen>;
  using out_vec_t = TxN_t<OutType, VecLen>;
  in_vec_t a;
  out_vec_t b;
  IdxType idx = threadIdx.x + (static_cast<IdxType>(blockIdx.x) * blockDim.x);
  idx *= in_vec_t::Ratio;
  if (idx >= len) {
    return;
  }
  a.load(in, idx);
#pragma unroll
  for (int i = 0; i < in_vec_t::Ratio; ++i) {
    b.val.data[i] = op(a.val.data[i]);
  }
  b.store(out, idx);
}

template <typename InType, int VecLen, typename Lambda, typename OutType,
          typename IdxType, int TPB>
void unary_op_impl(OutType *out, const InType *in, IdxType len, Lambda op,
                   cudaStream_t stream) {
  const IdxType nblks =
    raft::ceildiv(VecLen ? len / VecLen : len, static_cast<IdxType>(TPB));
  unary_op_kernel<InType, VecLen, Lambda, OutType, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, in, len, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

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
template <typename InType, typename Lambda, typename IdxType = int,
          typename OutType = InType, int TPB = 256>
void unaryOp(OutType *out, const InType *in, IdxType len, Lambda op,  // NOLINT
             cudaStream_t stream) {
  if (len <= 0) {
    return;  //silently skip in case of 0 length input
  }
  constexpr auto kMaxSize =
    sizeof(InType) >= sizeof(OutType) ? sizeof(InType) : sizeof(OutType);
  size_t bytes = len * kMaxSize;
  auto in_addr = reinterpret_cast<uint64_t>(in);
  auto out_addr = reinterpret_cast<uint64_t>(out);
  if (16 / kMaxSize && bytes % 16 == 0 && in_addr % 16 == 0 &&
      out_addr % 16 == 0) {
    unary_op_impl<InType, 16 / kMaxSize, Lambda, OutType, IdxType, TPB>(
      out, in, len, op, stream);
  } else if (8 / kMaxSize && bytes % 8 == 0 && in_addr % 8 == 0 &&
             out_addr % 8 == 0) {
    unary_op_impl<InType, 8 / kMaxSize, Lambda, OutType, IdxType, TPB>(
      out, in, len, op, stream);
  } else if (4 / kMaxSize && bytes % 4 == 0 && in_addr % 4 == 0 &&
             out_addr % 4 == 0) {
    unary_op_impl<InType, 4 / kMaxSize, Lambda, OutType, IdxType, TPB>(
      out, in, len, op, stream);
  } else if (2 / kMaxSize && bytes % 2 == 0 && in_addr % 2 == 0 &&
             out_addr % 2 == 0) {
    unary_op_impl<InType, 2 / kMaxSize, Lambda, OutType, IdxType, TPB>(
      out, in, len, op, stream);
  } else if (1 / kMaxSize) {
    unary_op_impl<InType, 1 / kMaxSize, Lambda, OutType, IdxType, TPB>(
      out, in, len, op, stream);
  } else {
    unary_op_impl<InType, 1, Lambda, OutType, IdxType, TPB>(out, in, len, op,
                                                            stream);
  }
}

template <typename OutType, typename Lambda, typename IdxType>
__global__ void write_only_unary_op_kernel(OutType *out, IdxType len,
                                           Lambda op) {
  IdxType idx = threadIdx.x + (static_cast<IdxType>(blockIdx.x) * blockDim.x);
  if (idx < len) {
    op(out + idx, idx);
  }
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
template <typename OutType, typename Lambda, typename IdxType = int,
          int TPB = 256>
void writeOnlyUnaryOp(OutType *out, IdxType len, Lambda op,  // NOLINT
                      cudaStream_t stream) {
  if (len <= 0) {
    return;  // silently skip in case of 0 length input
  }
  auto nblks = raft::ceildiv<IdxType>(len, TPB);
  write_only_unary_op_kernel<OutType, Lambda, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, len, op);
  CUDA_CHECK(cudaGetLastError());
}

};  // end namespace linalg
};  // end namespace raft
