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

#include <raft/cuda_utils.cuh>
#include <raft/vectorized.cuh>

namespace raft {

template <typename DataT, int VecLen, typename Lambda, typename IdxT>
__global__ void scatter_kernel(DataT *out, const DataT *in, const IdxT *idx,
                               IdxT len, Lambda op) {
  using DataVec = TxN_t<DataT, VecLen>;
  using IdxVec = TxN_t<IdxT, VecLen>;
  IdxT tid = threadIdx.x + (static_cast<IdxT>(blockIdx.x) * blockDim.x);
  tid *= VecLen;
  if (tid >= len) return;
  IdxVec idx_in;
  idx_in.load(idx, tid);
  DataVec data_in;
#pragma unroll
  for (int i = 0; i < VecLen; ++i) {
    auto in_pos = idx_in.val.data[i];
    data_in.val.data[i] = op(in[in_pos], tid + i);
  }
  data_in.store(out, tid);
}

template <typename DataT, int VecLen, typename Lambda, typename IdxT, int TPB>
void scatter_impl(DataT *out, const DataT *in, const IdxT *idx, IdxT len,
                  Lambda op, cudaStream_t stream) {
  const IdxT nblks = raft::ceildiv(VecLen ? len / VecLen : len, static_cast<IdxT>(TPB));
  scatter_kernel<DataT, VecLen, Lambda, IdxT>
    <<<nblks, TPB, 0, stream>>>(out, in, idx, len, op);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Performs scatter operation based on the input indexing array
 * @tparam DataT data type whose array gets scattered
 * @tparam IdxT indexing type
 * @tparam TPB threads-per-block in the final kernel launched
 * @tparam Lambda the device-lambda performing a unary operation on the loaded
 * data before it gets scattered
 * @param out the output array
 * @param in the input array
 * @param idx the indexing array
 * @param len number of elements in the input array
 * @param stream cuda stream where to launch work
 * @param op the device-lambda with signature `DataT func(DataT, IdxT);`. This
 * will be applied to every element before scattering it to the right location.
 * The second param in this method will be the destination index.
 */
template <typename DataT, typename IdxT,
          typename Lambda = raft::Nop<DataT, IdxT>, int TPB = 256>
void scatter(DataT *out, const DataT *in, const IdxT *idx, IdxT len,
             cudaStream_t stream, Lambda op = raft::Nop<DataT, IdxT>()) {
  if (len <= 0) return;
  constexpr auto kDataSize = sizeof(DataT);
  constexpr auto kIdxSize = sizeof(IdxT);
  constexpr auto kMaxPerElem = kDataSize > kIdxSize ? kDataSize : kIdxSize;
  size_t bytes = len * kMaxPerElem;
  if (16 / kMaxPerElem && bytes % 16 == 0) {
    scatter_impl<DataT, 16 / kMaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len,
                                                           op, stream);
  } else if (8 / kMaxPerElem && bytes % 8 == 0) {
    scatter_impl<DataT, 8 / kMaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op,
                                                          stream);
  } else if (4 / kMaxPerElem && bytes % 4 == 0) {
    scatter_impl<DataT, 4 / kMaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op,
                                                          stream);
  } else if (2 / kMaxPerElem && bytes % 2 == 0) {
    scatter_impl<DataT, 2 / kMaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op,
                                                          stream);
  } else if (1 / kMaxPerElem) {
    scatter_impl<DataT, 1 / kMaxPerElem, Lambda, IdxT, TPB>(out, in, idx, len, op,
                                                          stream);
  } else {
    scatter_impl<DataT, 1, Lambda, IdxT, TPB>(out, in, idx, len, op, stream);
  }
}

}  // namespace raft
