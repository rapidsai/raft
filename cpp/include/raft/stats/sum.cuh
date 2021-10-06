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

#include "detail/sum.cuh"

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/eltwise.cuh>

namespace raft {
namespace stats {

/**
 * @brief Compute sum of the input matrix
 *
 * Sum operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param output the output mean vector
 * @param input the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param rowMajor whether the input data is row or col major
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename IdxType = int>
void sum(Type *output, const Type *input, IdxType D, IdxType N, bool rowMajor,
         cudaStream_t stream) {
  static const int TPB = 256;
  if (rowMajor) {
    static const int RowsPerThread = 4;
    static const int ColsPerBlk = 32;
    static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
    dim3 grid(raft::ceildiv(N, (IdxType)RowsPerBlk),
              raft::ceildiv(D, (IdxType)ColsPerBlk));
    CUDA_CHECK(cudaMemset(output, 0, sizeof(Type) * D));
    detail::sumKernelRowMajor<Type, IdxType, TPB, ColsPerBlk>
      <<<grid, TPB, 0, stream>>>(output, input, D, N);
  } else {
    detail::sumKernelColMajor<Type, IdxType, TPB>
      <<<D, TPB, 0, stream>>>(output, input, D, N);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

};  // end namespace stats
};  // end namespace raft
