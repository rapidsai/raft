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

#include <cub/cub.cuh>
#include <limits>
#include <stdint.h>

#include <raft/distance/detail/compress_to_bits.cuh>
#include <raft/distance/detail/fused_l2_nn.cuh>
#include <raft/distance/detail/sparse_distance_base.cuh>
#include <raft/linalg/contractions.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace distance {
namespace detail {

#if (ENABLE_MEMCPY_ASYNC == 1)
#include <cuda_pipeline.h>
using namespace nvcuda::experimental;
#endif

template <typename DataT,
          typename OutT,
          typename IdxT,
          bool Sqrt,
          typename P,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename CoreLambda,
          typename FinalLambda>
__global__ __launch_bounds__(P::Nthreads, 2) void sparseL2NNkernel(OutT* min,
                                                                   const DataT* x,
                                                                   const DataT* y,
                                                                   const DataT* xn,
                                                                   const DataT* yn,
                                                                   const uint64_t* adj,
                                                                   const IdxT* group_idxs,
                                                                   IdxT num_groups,
                                                                   IdxT m,
                                                                   IdxT n,
                                                                   IdxT k,
                                                                   DataT maxVal,
                                                                   int* mutex,
                                                                   ReduceOpT redOp,
                                                                   KVPReduceOpT pairRedOp,
                                                                   CoreLambda core_op,
                                                                   FinalLambda fin_op)
{
  extern __shared__ char smem[];

  typedef cub::KeyValuePair<IdxT, DataT> KVPair;
  KVPair val[P::AccRowsPerTh];
#pragma unroll
  for (int i = 0; i < P::AccRowsPerTh; ++i) {
    val[i] = {-1, maxVal};
  }

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [pairRedOp, &val, maxVal] __device__(
                         DataT acc[P::AccRowsPerTh][P::AccColsPerTh],
                         int acc_adj,
                         DataT* regxn,
                         DataT* regyn,
                         IdxT tile_idx_n,
                         IdxT tile_idx_m,
                         IdxT tile_end_n) {
    KVPReduceOpT pairRed_op(pairRedOp);

#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = regxn[i] + regyn[j] - (DataT)2.0 * acc[i][j];
      }
    }
    if (Sqrt) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
          acc[i][j] = raft::mySqrt(acc[i][j]);
        }
      }
    }

    // intra thread reduce
    const auto acccolid = threadIdx.x % P::AccThCols;
    const auto accrowid = threadIdx.x / P::AccThCols;

#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      const bool ignore = (acc_adj & (1 << i)) == 0;
      if (ignore) { continue; }
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto tmpkey = acccolid + j * P::AccThCols + tile_idx_n;
        if (tile_end_n <= tmpkey) {
          // Do not process beyond end of tile.
          continue;
        }
        KVPair tmp = {tmpkey, acc[i][j]};
        if (tmpkey < tile_end_n) {
          val[i] = pairRed_op(accrowid + i * P::AccThRows + tile_idx_m, tmp, val[i]);
        }
      }
    }
  };

  auto rowEpilog_lambda =
    [m, mutex, min, pairRedOp, redOp, &val, maxVal] __device__(IdxT tile_idx_m) {
      KVPReduceOpT pairRed_op(pairRedOp);
      ReduceOpT red_op(redOp);

      const auto accrowid = threadIdx.x / P::AccThCols;
      const auto lid      = raft::laneId();
    // reduce
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = P::AccThCols / 2; j > 0; j >>= 1) {
          auto tmpkey   = raft::shfl(val[i].key, lid + j);
          auto tmpvalue = raft::shfl(val[i].value, lid + j);
          KVPair tmp    = {tmpkey, tmpvalue};
          val[i]        = pairRed_op(accrowid + i * P::AccThRows + tile_idx_m, tmp, val[i]);
        }
      }

      updateReducedVal<P, OutT, IdxT, KVPair, ReduceOpT>(mutex, min, val, red_op, m, tile_idx_m);

    // reset the val array.
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        val[i] = {-1, maxVal};
      }
    };

  IdxT lda = k, ldb = k, ldd = n;
  SparseDistances<true,
                  DataT,
                  DataT,
                  DataT,
                  IdxT,
                  P,
                  CoreLambda,
                  decltype(epilog_lambda),
                  FinalLambda,
                  decltype(rowEpilog_lambda),
                  true>
    obj(x,
        y,
        m,
        n,
        k,
        lda,
        ldb,
        ldd,
        xn,
        yn,
        adj,
        group_idxs,
        num_groups,
        smem,
        core_op,
        epilog_lambda,
        fin_op,
        rowEpilog_lambda);
  obj.run();
}

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void sparseL2NNImpl(OutT* min,
                    const DataT* x,
                    const DataT* y,
                    const DataT* xn,
                    const DataT* yn,
                    const bool* adj,
                    const IdxT* group_idxs,
                    IdxT num_groups,
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
  typedef typename linalg::Policy4x4<DataT, 1>::Policy P;

  static_assert(P::Mblk == 64, "sparseL2NNImpl only supports a policy with 64 rows per block.");

  // First, compress boolean to bitfield.

  // TODO 1: Remove allocation; use workspace instead(?)
  // TODO 2: Use a faster compress_to_bits implementation that does not require a pre-zeroed output.
  rmm::device_uvector<uint64_t> adj64(raft::ceildiv(m, IdxT(64)) * num_groups, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(adj64.data(), 0, adj64.size() * sizeof(uint64_t), stream));
  dim3 compress_grid(raft::ceildiv(m, 32), raft::ceildiv(num_groups, 32));
  compress_to_bits_naive<<<compress_grid, dim3(32, 32), 0, stream>>>(
    adj, num_groups, m, adj64.data());

  dim3 blk(P::Nthreads);
  auto nblks            = raft::ceildiv<int>(m, P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef cub::KeyValuePair<IdxT, DataT> KVPair;

  // Accumulation operation lambda
  auto core_lambda = [] __device__(DataT & acc, DataT & x, DataT & y) { acc += x * y; };

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  // TODO 3: remove fin_op
  auto fin_op = [] __device__(DataT d_val, int g_d_idx) { return d_val; };

  constexpr size_t shmemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
  if (sqrt) {
    auto sparseL2NNSqrt = sparseL2NNkernel<DataT,
                                           OutT,
                                           IdxT,
                                           true,
                                           P,
                                           ReduceOpT,
                                           KVPReduceOpT,
                                           decltype(core_lambda),
                                           decltype(fin_op)>;
    dim3 grid           = launchConfigGenerator<P>(m, n, shmemSize, sparseL2NNSqrt);

    sparseL2NNSqrt<<<grid, blk, shmemSize, stream>>>(min,
                                                     x,
                                                     y,
                                                     xn,
                                                     yn,
                                                     adj64.data(),
                                                     group_idxs,
                                                     num_groups,
                                                     m,
                                                     n,
                                                     k,
                                                     maxVal,
                                                     workspace,
                                                     redOp,
                                                     pairRedOp,
                                                     core_lambda,
                                                     fin_op);
  } else {
    auto sparseL2NN = sparseL2NNkernel<DataT,
                                       OutT,
                                       IdxT,
                                       false,
                                       P,
                                       ReduceOpT,
                                       KVPReduceOpT,
                                       decltype(core_lambda),
                                       decltype(fin_op)>;
    dim3 grid       = launchConfigGenerator<P>(m, n, shmemSize, sparseL2NN);
    sparseL2NN<<<grid, blk, shmemSize, stream>>>(min,
                                                 x,
                                                 y,
                                                 xn,
                                                 yn,
                                                 adj64.data(),
                                                 group_idxs,
                                                 num_groups,
                                                 m,
                                                 n,
                                                 k,
                                                 maxVal,
                                                 workspace,
                                                 redOp,
                                                 pairRedOp,
                                                 core_lambda,
                                                 fin_op);
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
