/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
