/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <limits>
#include <raft/core/kvp.hpp>
#include <raft/distance/detail/pairwise_distance_cutlass_base.cuh>
#include <raft/util/cuda_utils.cuh>
#include <stdint.h>
#include <raft/distance/detail/fused_l2_nn.cuh>

namespace raft {
namespace distance {

namespace detail {


// TODO: specialize this function for MinAndDistanceReduceOp<int, float>
// with atomicCAS of 64 bit which will eliminate mutex and shfls
template <typename P, typename OutT, typename IdxT, typename KVPair, typename ReduceOpT>
DI void updateReducedVal(
  int* mutex, OutT* min, KVPair* val, ReduceOpT red_op, IdxT m, IdxT gridStrideY)
{
  const auto lid      = threadIdx.x % raft::WarpSize;
  const auto accrowid = threadIdx.x / P::AccThCols;

  // Update each output row in order within a warp. This will resolve hang
  // issues with pre-Volta architectures
#pragma unroll
  for (int j = 0; j < (raft::WarpSize / P::AccThCols); j++) {
    if (lid == j * P::AccThCols) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        auto rid = gridStrideY + accrowid + i * P::AccThRows;
        if (rid < m) {
          auto value = val[i];
          while (atomicCAS(mutex + rid, 0, 1) == 1)
            ;
          __threadfence();
          red_op(rid, min + rid, value);
          __threadfence();
          atomicCAS(mutex + rid, 1, 0);
        }
      }
    }
  }
}

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedL2NNImpl(OutT* min,
                   const DataT* x,
                   const DataT* y,
                   const DataT* xn,
                   const DataT* yn,
                   IdxT m,
                   IdxT n,
                   IdxT k,
                   int* workspace,
                   ReduceOpT redOp,
                   KVPReduceOpT pairRedOp,
                   bool sqrt,
                   bool initOutBuffer,
                   cudaStream_t stream)
{
  // The kernel policy is determined by fusedL2NN.
  typedef Policy P;

  dim3 blk(P::Nthreads);
  auto nblks            = raft::ceildiv<int>(m, P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef KeyValuePair<IdxT, DataT> KVPair;

  // Accumulation operation lambda
  auto core_lambda = [] __device__(DataT & acc, DataT & x, DataT & y) { acc += x * y; };

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  auto fin_op = [] __device__(DataT d_val, int g_d_idx) { return d_val; };

  constexpr size_t shmemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
  if (sqrt) {
    auto fusedL2NNSqrt = fusedL2NNkernel<DataT,
                                         OutT,
                                         IdxT,
                                         true,
                                         P,
                                         ReduceOpT,
                                         KVPReduceOpT,
                                         decltype(core_lambda),
                                         decltype(fin_op)>;
    dim3 grid          = launchConfigGenerator<P>(m, n, shmemSize, fusedL2NNSqrt);

    fusedL2NNSqrt<<<grid, blk, shmemSize, stream>>>(
      min, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp, core_lambda, fin_op);
  } else {
    auto fusedL2NN = fusedL2NNkernel<DataT,
                                     OutT,
                                     IdxT,
                                     false,
                                     P,
                                     ReduceOpT,
                                     KVPReduceOpT,
                                     decltype(core_lambda),
                                     decltype(fin_op)>;
    dim3 grid      = launchConfigGenerator<P>(m, n, shmemSize, fusedL2NN);
    fusedL2NN<<<grid, blk, shmemSize, stream>>>(
      min, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp, core_lambda, fin_op);
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
