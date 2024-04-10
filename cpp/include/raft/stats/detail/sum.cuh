/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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
RAFT_KERNEL sumKernelRowMajor(Type* mu, const Type* data, IdxType D, IdxType N)
{
  const int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId           = threadIdx.x % ColsPerBlk;
  IdxType thisRowId           = threadIdx.x / ColsPerBlk;
  IdxType colId               = thisColId + ((IdxType)blockIdx.y * ColsPerBlk);
  IdxType rowId               = thisRowId + ((IdxType)blockIdx.x * RowsPerBlkPerIter);
  Type thread_sum             = Type(0);
  const IdxType stride        = RowsPerBlkPerIter * gridDim.x;
  for (IdxType i = rowId; i < N; i += stride) {
    thread_sum += (colId < D) ? data[i * D + colId] : Type(0);
  }
  __shared__ Type smu[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk) smu[threadIdx.x] = Type(0);
  __syncthreads();
  raft::myAtomicAdd(smu + thisColId, thread_sum);
  __syncthreads();
  if (threadIdx.x < ColsPerBlk && colId < D) raft::myAtomicAdd(mu + colId, smu[thisColId]);
}

template <typename Type, typename IdxType, int TPB, int ColsPerBlk = 32>
RAFT_KERNEL sumKahanKernelRowMajor(Type* mu, const Type* data, IdxType D, IdxType N)
{
  constexpr int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxType thisColId               = threadIdx.x % ColsPerBlk;
  IdxType thisRowId               = threadIdx.x / ColsPerBlk;
  IdxType colId                   = thisColId + ((IdxType)blockIdx.y * ColsPerBlk);
  IdxType rowId                   = thisRowId + ((IdxType)blockIdx.x * RowsPerBlkPerIter);
  Type thread_sum                 = Type(0);
  Type thread_c                   = Type(0);
  const IdxType stride            = RowsPerBlkPerIter * gridDim.x;
  for (IdxType i = rowId; i < N; i += stride) {
    // KahanBabushkaNeumaierSum
    const Type cur_value = (colId < D) ? data[i * D + colId] : Type(0);
    const Type t         = thread_sum + cur_value;
    if (abs(thread_sum) >= abs(cur_value)) {
      thread_c += (thread_sum - t) + cur_value;
    } else {
      thread_c += (cur_value - t) + thread_sum;
    }
    thread_sum = t;
  }
  thread_sum += thread_c;
  __shared__ Type smu[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk) smu[threadIdx.x] = Type(0);
  __syncthreads();
  raft::myAtomicAdd(smu + thisColId, thread_sum);
  __syncthreads();
  if (threadIdx.x < ColsPerBlk && colId < D) raft::myAtomicAdd(mu + colId, smu[thisColId]);
}

template <typename Type, typename IdxType, int TPB>
RAFT_KERNEL sumKahanKernelColMajor(Type* mu, const Type* data, IdxType D, IdxType N)
{
  typedef cub::BlockReduce<Type, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  Type thread_sum  = Type(0);
  Type thread_c    = Type(0);
  IdxType colStart = N * blockIdx.x;
  for (IdxType i = threadIdx.x; i < N; i += TPB) {
    // KahanBabushkaNeumaierSum
    IdxType idx          = colStart + i;
    const Type cur_value = data[idx];
    const Type t         = thread_sum + cur_value;
    if (abs(thread_sum) >= abs(cur_value)) {
      thread_c += (thread_sum - t) + cur_value;
    } else {
      thread_c += (cur_value - t) + thread_sum;
    }
    thread_sum = t;
  }
  thread_sum += thread_c;
  Type acc = BlockReduce(temp_storage).Sum(thread_sum);
  if (threadIdx.x == 0) { mu[blockIdx.x] = acc; }
}

template <typename Type, typename IdxType = int>
void sum(Type* output, const Type* input, IdxType D, IdxType N, bool rowMajor, cudaStream_t stream)
{
  static const int TPB = 256;
  if (rowMajor) {
    static const int ColsPerBlk       = 8;
    static const int MinRowsPerThread = 16;
    static const int MinRowsPerBlk    = (TPB / ColsPerBlk) * MinRowsPerThread;
    static const int MaxBlocksDimX    = 8192;

    const IdxType grid_y = raft::ceildiv(D, (IdxType)ColsPerBlk);
    const IdxType grid_x =
      raft::min((IdxType)MaxBlocksDimX, raft::ceildiv(N, (IdxType)MinRowsPerBlk));

    dim3 grid(grid_x, grid_y);
    RAFT_CUDA_TRY(cudaMemset(output, 0, sizeof(Type) * D));
    sumKahanKernelRowMajor<Type, IdxType, TPB, ColsPerBlk>
      <<<grid, TPB, 0, stream>>>(output, input, D, N);
  } else {
    sumKahanKernelColMajor<Type, IdxType, TPB><<<D, TPB, 0, stream>>>(output, input, D, N);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace stats
}  // namespace raft