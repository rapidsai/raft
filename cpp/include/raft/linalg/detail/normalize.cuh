/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

namespace raft {
namespace linalg {
namespace detail {

template <int warpSize, int rpb>
struct NormalizeThinPolicy {
  static constexpr int LogicalWarpSize = warpSize;
  static constexpr int RowsPerBlock    = rpb;
  static constexpr int ThreadsPerBlock = LogicalWarpSize * RowsPerBlock;
};

template <typename Policy,
          typename Type,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
__global__ void __launch_bounds__(Policy::ThreadsPerBlock)
  coalesced_normalize_thin_kernel(Type* out,
                                  const Type* in,
                                  IdxType D,
                                  IdxType N,
                                  MainLambda main_op,
                                  ReduceLambda reduce_op,
                                  FinalLambda fin_op)
{
  IdxType i = threadIdx.y + (Policy::RowsPerBlock * static_cast<IdxType>(blockIdx.x));
  if (i >= N) return;

  Type acc = 0.0;
  for (IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize) {
    Type val = in[j + D * i];
    acc      = reduce_op(acc, main_op(val));
  }
  acc = raft::logicalWarpReduce<Policy::LogicalWarpSize>(acc, reduce_op);
  if (acc <= 1e-8) return;
  for (IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize) {
    out[j + D * i] = in[j + D * i] / fin_op(acc);
  }
}

template <typename Policy,
          typename Type,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
inline void coalesced_normalize_thin(Type* out,
                                     const Type* in,
                                     IdxType D,
                                     IdxType N,
                                     cudaStream_t stream,
                                     MainLambda main_op,
                                     ReduceLambda reduce_op,
                                     FinalLambda fin_op)
{
  dim3 grid(ceildiv(N, (IdxType)Policy::RowsPerBlock), 1, 1);
  dim3 block(Policy::LogicalWarpSize, Policy::RowsPerBlock, 1);
  coalesced_normalize_thin_kernel<Policy>
    <<<grid, block, 0, stream>>>(out, in, D, N, main_op, reduce_op, fin_op);
}

template <typename Type,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
void coalesced_normalize(Type* out,
                         const Type* in,
                         IdxType D,
                         IdxType N,
                         cudaStream_t stream,
                         MainLambda main_op,
                         ReduceLambda reduce_op,
                         FinalLambda fin_op)
{
  if (D <= 2) {
    coalesced_normalize_thin<NormalizeThinPolicy<2, 64>>(
      out, in, D, N, stream, main_op, reduce_op, fin_op);
  } else if (D <= 4) {
    coalesced_normalize_thin<NormalizeThinPolicy<4, 32>>(
      out, in, D, N, stream, main_op, reduce_op, fin_op);
  } else if (D <= 8) {
    coalesced_normalize_thin<NormalizeThinPolicy<8, 16>>(
      out, in, D, N, stream, main_op, reduce_op, fin_op);
  } else if (D <= 16) {
    coalesced_normalize_thin<NormalizeThinPolicy<16, 8>>(
      out, in, D, N, stream, main_op, reduce_op, fin_op);
  } else {
    coalesced_normalize_thin<NormalizeThinPolicy<32, 4>>(
      out, in, D, N, stream, main_op, reduce_op, fin_op);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
