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

#include <thrust/functional.h>

namespace raft {
namespace linalg {
namespace detail {

template <typename InT, typename OutT = InT, typename IdxType = int>
void addScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, adds_scalar<InT, OutT>(scalar), stream);
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void add(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, thrust::plus<InT>(), stream);
}

template <class math_t, typename IdxType>
__global__ void add_dev_scalar_kernel(math_t* outDev,
                                      const math_t* inDev,
                                      const math_t* singleScalarDev,
                                      IdxType len)
{
  IdxType i = ((IdxType)blockIdx.x * (IdxType)blockDim.x) + threadIdx.x;
  if (i < len) { outDev[i] = inDev[i] + *singleScalarDev; }
}

template <typename math_t, typename IdxType = int>
void addDevScalar(math_t* outDev,
                  const math_t* inDev,
                  const math_t* singleScalarDev,
                  IdxType len,
                  cudaStream_t stream)
{
  // TODO: block dimension has not been tuned
  dim3 block(256);
  dim3 grid(raft::ceildiv(len, (IdxType)block.x));
  add_dev_scalar_kernel<math_t><<<grid, block, 0, stream>>>(outDev, inDev, singleScalarDev, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft