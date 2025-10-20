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

#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename InT, typename OutT = InT, typename IdxType = int>
void subtractScalar(OutT* out, const InT* in, InT scalar, IdxType len, cudaStream_t stream)
{
  raft::linalg::unaryOp(out, in, len, raft::sub_const_op<InT>(scalar), stream);
}

template <typename InT, typename OutT = InT, typename IdxType = int>
void subtract(OutT* out, const InT* in1, const InT* in2, IdxType len, cudaStream_t stream)
{
  raft::linalg::binaryOp(out, in1, in2, len, raft::sub_op(), stream);
}

template <class math_t, typename IdxType>
RAFT_KERNEL subtract_dev_scalar_kernel(math_t* outDev,
                                       const math_t* inDev,
                                       const math_t* singleScalarDev,
                                       IdxType len)
{
  // TODO: kernel do not use shared memory in current implementation
  int i = ((IdxType)blockIdx.x * (IdxType)blockDim.x) + threadIdx.x;
  if (i < len) { outDev[i] = inDev[i] - *singleScalarDev; }
}

template <typename math_t, typename IdxType = int, int TPB = 256>
void subtractDevScalar(math_t* outDev,
                       const math_t* inDev,
                       const math_t* singleScalarDev,
                       IdxType len,
                       cudaStream_t stream)
{
  // Just for the note - there is no way to express such operation with cuBLAS in effective way
  // https://stackoverflow.com/questions/14051064/add-scalar-to-vector-in-blas-cublas-cuda
  const IdxType nblks = raft::ceildiv(len, (IdxType)TPB);
  subtract_dev_scalar_kernel<math_t>
    <<<nblks, TPB, 0, stream>>>(outDev, inDev, singleScalarDev, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
