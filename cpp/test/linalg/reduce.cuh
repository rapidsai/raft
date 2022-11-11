/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cublas_v2.h>
#include <raft/linalg/detail/cublas_wrappers.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/memory.h>

namespace raft {
namespace linalg {

template <typename InType, typename OutType, typename IdxType>
__global__ void naiveCoalescedReductionKernel(OutType* dots,
                                              const InType* data,
                                              IdxType D,
                                              IdxType N)
{
  OutType acc      = (OutType)0;
  IdxType rowStart = threadIdx.x + static_cast<IdxType>(blockIdx.x) * blockDim.x;
  if (rowStart < N) {
    for (IdxType i = 0; i < D; ++i) {
      acc += static_cast<OutType>(data[rowStart * D + i] * data[rowStart * D + i]);
    }
    dots[rowStart] = 2 * acc;
  }
}

template <typename InType, typename OutType, typename IdxType>
void naiveCoalescedReduction(
  OutType* dots, const InType* data, IdxType D, IdxType N, cudaStream_t stream)
{
  static const IdxType TPB = 64;
  IdxType nblks            = raft::ceildiv(N, TPB);
  naiveCoalescedReductionKernel<InType, OutType><<<nblks, TPB, 0, stream>>>(dots, data, D, N);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename InType, typename OutType, typename IdxType>
void unaryAndGemv(OutType* dots, const InType* data, IdxType D, IdxType N, cudaStream_t stream)
{
  // computes a MLCommon unary op on data (squares it), then computes Ax
  //(A input matrix and x column vector) to sum columns
  rmm::device_uvector<OutType> sq(D * N, stream);
  raft::linalg::unaryOp(
    thrust::raw_pointer_cast(sq.data()),
    data,
    D * N,
    [] __device__(InType v) { return static_cast<OutType>(v * v); },
    stream);
  cublasHandle_t handle;
  RAFT_CUBLAS_TRY(cublasCreate(&handle));
  rmm::device_uvector<OutType> ones(N, stream);  // column vector [1...1]
  raft::linalg::unaryOp<OutType>(
    ones.data(), ones.data(), ones.size(), [=] __device__(OutType input) { return 1; }, stream);
  OutType alpha = 1, beta = 0;
  // #TODO: Call from public API when ready
  RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemv(
    handle, CUBLAS_OP_N, D, N, &alpha, sq.data(), D, ones.data(), 1, &beta, dots, 1, stream));
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  RAFT_CUBLAS_TRY(cublasDestroy(handle));
}

template <typename InType, typename OutType, typename IdxType>
void naiveReduction(OutType* dots,
                    const InType* data,
                    IdxType D,
                    IdxType N,
                    bool rowMajor,
                    bool alongRows,
                    cudaStream_t stream)
{
  if (rowMajor && alongRows) {
    naiveCoalescedReduction(dots, data, D, N, stream);
  } else if (rowMajor && !alongRows) {
    unaryAndGemv(dots, data, D, N, stream);
  } else if (!rowMajor && alongRows) {
    unaryAndGemv(dots, data, N, D, stream);
  } else {
    naiveCoalescedReduction(dots, data, N, D, stream);
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

}  // end namespace linalg
}  // end namespace raft
