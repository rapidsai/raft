/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>
#include <raft/util/vectorized.cuh>

namespace raft::detail {

template <typename DataT, int VecLen, typename Lambda, typename IdxT>
RAFT_KERNEL scatterKernel(DataT* out, const DataT* in, const IdxT* idx, IdxT len, Lambda op)
{
  typedef TxN_t<DataT, VecLen> DataVec;
  typedef TxN_t<IdxT, VecLen> IdxVec;
  IdxT tid = threadIdx.x + ((IdxT)blockIdx.x * blockDim.x);
  tid *= VecLen;
  if (tid >= len) return;
  IdxVec idxIn;
  idxIn.load(idx, tid);
  DataVec dataIn;
#pragma unroll
  for (int i = 0; i < VecLen; ++i) {
    auto inPos         = idxIn.val.data[i];
    dataIn.val.data[i] = op(in[inPos], tid + i);
  }
  dataIn.store(out, tid);
}

template <typename DataT, int VecLen, typename Lambda, typename IdxT, int TPB>
void scatterImpl(
  DataT* out, const DataT* in, const IdxT* idx, IdxT len, Lambda op, cudaStream_t stream)
{
  const IdxT nblks = raft::ceildiv(VecLen ? len / VecLen : len, (IdxT)TPB);
  scatterKernel<DataT, VecLen, Lambda, IdxT><<<nblks, TPB, 0, stream>>>(out, in, idx, len, op);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace raft::detail
