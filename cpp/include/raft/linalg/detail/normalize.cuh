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
struct NormalizeWarpPolicy {
  static constexpr int LogicalWarpSize = warpSize;
  static constexpr int RowsPerBlock    = rpb;
  static constexpr int ThreadsPerBlock = LogicalWarpSize * RowsPerBlock;
};

template <typename Policy, typename Type, typename IdxType>
__global__ void __launch_bounds__(Policy::ThreadsPerBlock)
  coalescedNormalizeWarpKernel(Type* out, const Type* in, IdxType D, IdxType N)
{
  IdxType i = threadIdx.y + (blockDim.y * static_cast<IdxType>(blockIdx.x));
  if (i >= N) return;

  Type sqsum = 0.0;
  for (IdxType j = threadIdx.x; j < D; j += blockDim.x) {
    Type val = in[j + D * i];
    sqsum += val * val;
  }
  sqsum = raft::logicalWarpReduce<Policy::LogicalWarpSize>(sqsum);
  if (sqsum <= 1e-8) return;
  sqsum = rsqrt(sqsum);
  for (IdxType j = threadIdx.x; j < D; j += blockDim.x) {
    out[j + D * i] = in[j + D * i] * sqsum;
  }
}

template <typename Policy, typename Type, typename IdxType>
inline void coalescedNormalizeLauncher(
  Type* out, const Type* in, IdxType D, IdxType N, cudaStream_t stream)
{
  dim3 grid(ceildiv(N, (IdxType)Policy::RowsPerBlock), 1, 1);
  dim3 block(Policy::LogicalWarpSize, Policy::RowsPerBlock, 1);
  coalescedNormalizeWarpKernel<Policy><<<grid, block, 0, stream>>>(out, in, D, N);
}

template <typename Type, typename IdxType>
void coalescedNormalize(Type* out, const Type* in, IdxType D, IdxType N, cudaStream_t stream)
{
  if (D <= 2) {
    coalescedNormalizeLauncher<NormalizeWarpPolicy<2, 64>>(out, in, D, N, stream);
  } else if (D <= 4) {
    coalescedNormalizeLauncher<NormalizeWarpPolicy<4, 32>>(out, in, D, N, stream);
  } else if (D <= 8) {
    coalescedNormalizeLauncher<NormalizeWarpPolicy<8, 16>>(out, in, D, N, stream);
  } else if (D <= 16) {
    coalescedNormalizeLauncher<NormalizeWarpPolicy<16, 8>>(out, in, D, N, stream);
  } else {
    coalescedNormalizeLauncher<NormalizeWarpPolicy<32, 4>>(out, in, D, N, stream);
  }
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
