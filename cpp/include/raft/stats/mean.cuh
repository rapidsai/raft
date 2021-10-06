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

#include "detail/mean.cuh"

#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/eltwise.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute mean of the input matrix
 *
 * Mean operation is assumed to be performed on a given column.
 *
 * @tparam Type: the data type
 * @tparam IdxType Integer type used to for addressing
 * @param mu: the output mean vector
 * @param data: the input matrix
 * @param D: number of columns of data
 * @param N: number of rows of data
 * @param sample: whether to evaluate sample mean or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 * @param rowMajor: whether the input data is row or col major
 * @param stream: cuda stream
 */
template <typename Type, typename IdxType = int>
void mean(Type *mu, const Type *data, IdxType D, IdxType N, bool sample,
          bool rowMajor, cudaStream_t stream) {
  static const int TPB = 256;
  if (rowMajor) {
    static const int RowsPerThread = 4;
    static const int ColsPerBlk = 32;
    static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
    dim3 grid(raft::ceildiv(N, (IdxType)RowsPerBlk),
              raft::ceildiv(D, (IdxType)ColsPerBlk));
    CUDA_CHECK(cudaMemsetAsync(mu, 0, sizeof(Type) * D, stream));
    detail::meanKernelRowMajor<Type, IdxType, TPB, ColsPerBlk>
      <<<grid, TPB, 0, stream>>>(mu, data, D, N);
    CUDA_CHECK(cudaPeekAtLastError());
    Type ratio = Type(1) / (sample ? Type(N - 1) : Type(N));
    raft::linalg::scalarMultiply(mu, mu, ratio, D, stream);
  } else {
    detail::meanKernelColMajor<Type, IdxType, TPB>
      <<<D, TPB, 0, stream>>>(mu, data, D, N);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // namespace stats
};  // namespace raft
