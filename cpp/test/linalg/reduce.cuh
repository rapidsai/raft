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

#include <cublas_v2.h>
#include <raft/linalg/cublas_wrappers.h>
#include <thrust/device_vector.h>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/unary_op.cuh>

namespace raft {
namespace linalg {

template <typename Type>
__global__ void naive_coalesced_reduction_kernel(Type *dots, const Type *data,
                                                 int D, int N) {
  auto acc = static_cast<Type>(0);
  int row_start = threadIdx.x + blockIdx.x * blockDim.x;
  if (row_start < N) {
    for (int i = 0; i < D; ++i) {
      acc += data[row_start * D + i] * data[row_start * D + i];
    }
    dots[row_start] = 2 * acc;
  }
}

template <typename Type>
void naive_coalesced_reduction(Type *dots, const Type *data, int D, int N,
                               cudaStream_t stream) {
  static const int kTpb = 64;
  auto nblks = raft::ceildiv(N, kTpb);
  naive_coalesced_reduction_kernel<Type>
    <<<nblks, kTpb, 0, stream>>>(dots, data, D, N);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
void unary_and_gemv(Type *dots, const Type *data, int D, int N,
                    cudaStream_t stream) {
  //computes a MLCommon unary op on data (squares it), then computes Ax
  //(A input matrix and x column vector) to sum columns
  thrust::device_vector<Type> sq(D * N);
  raft::linalg::unaryOp(
    thrust::raw_pointer_cast(sq.data()), data, D * N,
    [] __device__(Type v) { return v * v; }, stream);
  cublasHandle_t handle;
  CUBLAS_CHECK(cublasCreate(&handle));
  thrust::device_vector<Type> ones(N, 1);  //column vector [1...1]
  Type alpha = 1, beta = 0;
  CUBLAS_CHECK(raft::linalg::cublasgemv(
    handle, CUBLAS_OP_N, D, N, &alpha, thrust::raw_pointer_cast(sq.data()), D,
    thrust::raw_pointer_cast(ones.data()), 1, &beta, dots, 1, stream));
  CUDA_CHECK(cudaDeviceSynchronize());
  CUBLAS_CHECK(cublasDestroy(handle));
}

template <typename Type>
void naive_reduction(Type *dots, const Type *data, int D, int N, bool rowMajor,
                     bool alongRows, cudaStream_t stream) {
  if (rowMajor && alongRows) {
    naive_coalesced_reduction(dots, data, D, N, stream);
  } else if (rowMajor && !alongRows) {
    unary_and_gemv(dots, data, D, N, stream);
  } else if (!rowMajor && alongRows) {
    unary_and_gemv(dots, data, N, D, stream);
  } else {
    naive_coalesced_reduction(dots, data, N, D, stream);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
}

}  // end namespace linalg
}  // end namespace raft
