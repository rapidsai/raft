/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/util/vectorized.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InType, int VecLen, typename Lambda, typename IdxType, typename OutType>
__global__ void binaryOpKernel(
  OutType* out, const InType* in1, const InType* in2, IdxType len, Lambda op)
{
  typedef TxN_t<InType, VecLen> InVecType;
  typedef TxN_t<OutType, VecLen> OutVecType;
  InVecType a, b;
  OutVecType c;
  IdxType idx = threadIdx.x + ((IdxType)blockIdx.x * blockDim.x);
  idx *= InVecType::Ratio;
  if (idx >= len) return;
  a.load(in1, idx);
  b.load(in2, idx);
#pragma unroll
  for (int i = 0; i < InVecType::Ratio; ++i) {
    c.val.data[i] = op(a.val.data[i], b.val.data[i]);
  }
  c.store(out, idx);
}

template <typename InType, int VecLen, typename Lambda, typename IdxType, typename OutType, int TPB>
void binaryOpImpl(
  OutType* out, const InType* in1, const InType* in2, IdxType len, Lambda op, cudaStream_t stream)
{
  const IdxType nblks = raft::ceildiv(VecLen ? len / VecLen : len, (IdxType)TPB);
  binaryOpKernel<InType, VecLen, Lambda, IdxType, OutType>
    <<<nblks, TPB, 0, stream>>>(out, in1, in2, len, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Checks if addresses are aligned on N bytes
 */
inline bool addressAligned(uint64_t addr1, uint64_t addr2, uint64_t addr3, uint64_t N)
{
  return addr1 % N == 0 && addr2 % N == 0 && addr3 % N == 0;
}

template <typename InType,
          typename Lambda,
          typename OutType = InType,
          typename IdxType = int,
          int TPB          = 256>
void binaryOp(
  OutType* out, const InType* in1, const InType* in2, IdxType len, Lambda op, cudaStream_t stream)
{
  constexpr auto maxSize = sizeof(InType) > sizeof(OutType) ? sizeof(InType) : sizeof(OutType);
  size_t bytes           = len * maxSize;
  uint64_t in1Addr       = uint64_t(in1);
  uint64_t in2Addr       = uint64_t(in2);
  uint64_t outAddr       = uint64_t(out);
  if (16 / maxSize && bytes % 16 == 0 && addressAligned(in1Addr, in2Addr, outAddr, 16)) {
    binaryOpImpl<InType, 16 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (8 / maxSize && bytes % 8 == 0 && addressAligned(in1Addr, in2Addr, outAddr, 8)) {
    binaryOpImpl<InType, 8 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (4 / maxSize && bytes % 4 == 0 && addressAligned(in1Addr, in2Addr, outAddr, 4)) {
    binaryOpImpl<InType, 4 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (2 / maxSize && bytes % 2 == 0 && addressAligned(in1Addr, in2Addr, outAddr, 2)) {
    binaryOpImpl<InType, 2 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else if (1 / maxSize) {
    binaryOpImpl<InType, 1 / maxSize, Lambda, IdxType, OutType, TPB>(
      out, in1, in2, len, op, stream);
  } else {
    binaryOpImpl<InType, 1, Lambda, IdxType, OutType, TPB>(out, in1, in2, len, op, stream);
  }
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft