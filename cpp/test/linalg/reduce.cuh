/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <raft/linalg/cublas_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/unary_op.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace linalg {

template <typename InType, typename OutType>
__global__ void naiveCoalescedReductionKernel(OutType *dots, const InType *data,
                                              int D, int N) {
  OutType acc = (OutType)0;
  int rowStart = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowStart < N) {
    for (int i = 0; i < D; ++i) {
      acc +=
        static_cast<OutType>(data[rowStart * D + i] * data[rowStart * D + i]);
    }
    dots[rowStart] = 2 * acc;
  }
}

template <typename InType, typename OutType>
void naiveCoalescedReduction(OutType *dots, const InType *data, int D, int N,
                             cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = raft::ceildiv(N, TPB);
  naiveCoalescedReductionKernel<InType, OutType>
    <<<nblks, TPB, 0, stream>>>(dots, data, D, N);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename InType, typename OutType>
void unaryAndGemv(OutType *dots, const InType *data, int D, int N,
                  cudaStream_t stream) {
  //computes a MLCommon unary op on data (squares it), then computes Ax
  //(A input matrix and x column vector) to sum columns
  rmm::device_uvector<OutType> sq(D * N, stream);
  raft::linalg::unaryOp(
    thrust::raw_pointer_cast(sq.data()), data, D * N,
    [] __device__(InType v) { return static_cast<OutType>(v * v); }, stream);
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  rmm::device_uvector<OutType> ones(N, stream);  //column vector [1...1]
  raft::linalg::unaryOp<OutType>(
    ones.data(), ones.data(), ones.size(),
    [=] __device__(OutType input) { return 1; }, stream);
  OutType alpha = 1, beta = 0;
  CUBLAS_CHECK(raft::linalg::cublasgemv(handle, CUBLAS_OP_N, D, N, &alpha,
                                        sq.data(), D, ones.data(), 1, &beta,
                                        dots, 1, stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUBLAS_CHECK(cublasDestroy(handle));
}

template <typename InType, typename OutType>
void naiveReduction(OutType *dots, const InType *data, int D, int N,
                    bool rowMajor, bool alongRows, cudaStream_t stream) {
  if (rowMajor && alongRows) {
    naiveCoalescedReduction(dots, data, D, N, stream);
  } else if (rowMajor && !alongRows) {
    unaryAndGemv(dots, data, D, N, stream);
  } else if (!rowMajor && alongRows) {
    unaryAndGemv(dots, data, N, D, stream);
  } else {
    naiveCoalescedReduction(dots, data, N, D, stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // end namespace linalg
}  // end namespace raft
