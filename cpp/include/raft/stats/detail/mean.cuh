/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/linalg/eltwise.cuh>
#include <raft/util/cuda_utils.cuh>

#include <cub/cub.cuh>

namespace raft {
namespace stats {
namespace detail {

///@todo: ColsPerBlk has been tested only for 32!
template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
RAFT_KERNEL meanKernelRowMajor(Type* mu, const Type* data, IdxType D, IdxType N)
{
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId           = threadIdx.x % ColsPerBlk;
  IdxType thisRowId           = threadIdx.x / ColsPerBlk;
  IdxType colId               = thisColId + ((IdxType)blockIdx.y * ColsPerBlk);
  IdxType rowId               = thisRowId + ((IdxType)blockIdx.x * RowsPerBlkPerIter);
  Type thread_data            = Type(0);
  const IdxType stride        = RowsPerBlkPerIter * gridDim.x;
  for (IdxType i = rowId; i < N; i += stride)
    thread_data += (colId < D) ? data[i * D + colId] : Type(0);
  __shared__ Type smu[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk) smu[threadIdx.x] = Type(0);
  __syncthreads();
  raft::myAtomicAdd(smu + thisColId, thread_data);
  __syncthreads();
  if (threadIdx.x < ColsPerBlk) raft::myAtomicAdd(mu + colId, smu[thisColId]);
}

template <typename Type, typename IdxType, int TPB>
RAFT_KERNEL meanKernelColMajor(Type* mu, const Type* data, IdxType D, IdxType N)
{
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type thread_data = Type(0);
  IdxType colStart = N * blockIdx.x;
  for (IdxType i = threadIdx.x; i < N; i += TPB) {
    IdxType idx = colStart + i;
    thread_data += data[idx];
  }
  Type acc = BlockReduce(temp_storage).Sum(thread_data);
  if (threadIdx.x == 0) { mu[blockIdx.x] = acc / N; }
}

template <typename Type, typename IdxType = int>
void mean(
  Type* mu, const Type* data, IdxType D, IdxType N, bool sample, bool rowMajor, cudaStream_t stream)
{
  static const int TPB = 256;
  if (rowMajor) {
    static const int RowsPerThread = 4;
    static const int ColsPerBlk    = 32;
    static const int RowsPerBlk    = (TPB / ColsPerBlk) * RowsPerThread;
    dim3 grid(raft::ceildiv(N, (IdxType)RowsPerBlk), raft::ceildiv(D, (IdxType)ColsPerBlk));
    RAFT_CUDA_TRY(cudaMemsetAsync(mu, 0, sizeof(Type) * D, stream));
    meanKernelRowMajor<Type, IdxType, TPB, ColsPerBlk><<<grid, TPB, 0, stream>>>(mu, data, D, N);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
    Type ratio = Type(1) / (sample ? Type(N - 1) : Type(N));
    raft::linalg::scalarMultiply(mu, mu, ratio, D, stream);
  } else {
    meanKernelColMajor<Type, IdxType, TPB><<<D, TPB, 0, stream>>>(mu, data, D, N);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace stats
}  // namespace raft