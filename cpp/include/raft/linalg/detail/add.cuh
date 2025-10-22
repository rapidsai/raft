/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InT, typename OutT = InT, typename IdxType = int>
void addScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::add_const_op<InT>(scalar), stream);
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void add(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::add_op(), stream);
}

template <class InT, typename IdxType, typename OutT = InT>
RAFT_KERNEL add_dev_scalar_kernel(OutT* outDev,
                                  const InT* inDev,
                                  const InT* singleScalarDev,
                                  IdxType len)
{
  IdxType i = ((IdxType)blockIdx.x * (IdxType)blockDim.x) + threadIdx.x;
  if (i < len) { outDev[i] = inDev[i] + *singleScalarDev; }
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void addDevScalar(
  OutT* outDev, const InT* inDev, const InT* singleScalarDev, IdxType len, cudaStream_t stream)
{
  // TODO: block dimension has not been tuned
  dim3 block(256);
  dim3 grid(raft::ceildiv(len, (IdxType)block.x));
  add_dev_scalar_kernel<<<grid, block, 0, stream>>>(outDev, inDev, singleScalarDev, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
