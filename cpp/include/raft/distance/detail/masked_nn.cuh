/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/distance/detail/compress_to_bits.cuh>
#include <raft/distance/detail/fused_distance_nn/fused_l2_nn.cuh>
#include <raft/distance/detail/masked_distance_base.cuh>
#include <raft/linalg/contractions.cuh>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <stdint.h>

#include <limits>

namespace raft {
namespace distance {
namespace detail {

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename P,
          typename ReduceOpT,
          typename KVPReduceOpT,
          typename CoreLambda,
          typename FinalLambda>
__launch_bounds__(P::Nthreads, 2) RAFT_KERNEL masked_l2_nn_kernel(OutT* min,
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
                                                                  bool sqrt,
                                                                  DataT maxVal,
                                                                  int* mutex,
                                                                  ReduceOpT redOp,
                                                                  KVPReduceOpT pairRedOp,
                                                                  CoreLambda core_op,
                                                                  FinalLambda fin_op)
{
  extern __shared__ char smem[];

  typedef raft::KeyValuePair<IdxT, DataT> KVPair;
  KVPair val[P::AccRowsPerTh];
#pragma unroll
  for (int i = 0; i < P::AccRowsPerTh; ++i) {
    val[i] = {-1, maxVal};
  }

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [pairRedOp, &val, maxVal, sqrt] __device__(
                         DataT acc[P::AccRowsPerTh][P::AccColsPerTh],
                         int thread_adj,
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
    if (sqrt) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
          acc[i][j] = raft::sqrt(acc[i][j]);
        }
      }
    }

    // intra thread reduce
    const auto acccolid = threadIdx.x % P::AccThCols;
    const auto accrowid = threadIdx.x / P::AccThCols;

#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      // thread_adj is a bitfield that contains a 1 at location i iff we must
      // compute row i of acc (the accumulator register tile). It is described in
      // more detail in the maskedDistances.run() method.
      const bool ignore = (thread_adj & (1 << i)) == 0;
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
  MaskedDistances<true,
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

/**
 * @brief Wrapper for masked_l2_nn_kernel
 *
 * Responsibilities:
 * - Allocate (and initialize) workspace memory for:
 *   - mutexes used in nearest neighbor update step
 *   - adjacency matrix bitfield
 * - Compress adjacency matrix to bitfield
 * - Initialize output buffer (conditional on `initOutBuffer`)
 * - Specify core and final operations for the L2 norm
 * - Determine optimal launch configuration for kernel.
 * - Launch kernel and check for errors.
 *
 * @tparam DataT         Input data-type (for x and y matrices).
 * @tparam OutT          Output data-type (for key-value pairs).
 * @tparam IdxT          Index data-type.
 * @tparam ReduceOpT     A struct to perform the final needed reduction
 *                       operation and also to initialize the output array
 *                       elements with the appropriate initial value needed for
 *                       reduction.
 * @tparam KVPReduceOpT  Type of Reduction operation on key value pairs.
 *
 * @param      handle            RAFT handle for managing expensive resources
 * @param[out] out               Will contain reduced output (nn key-value pairs)
 * @param[in]  x                 First matrix. Row major. Dim = `m x k`. (on device)
 * @param[in]  y                 Second matrix. Row major. Dim = `n x k`. (on device)
 * @param[in]  xn                L2 squared norm of `x`. Length = `m`.
 * @param[in]  yn                L2 squared norm of `y`. Length = `n`.
 * @param[in]  adj           A boolean adjacency matrix indicating for each
 *                           row of `x` and each group in `y` whether to compute the
 *                           distance. Dim = `m x num_groups`.
 * @param[in]  group_idxs    An array containing the *end* indices of each group
 *                           in `y`. The value of group_idxs[j] indicates the
 *                           start of group j + 1, i.e., it is the inclusive
 *                           scan of the group lengths. The first group is
 *                           always assumed to start at index 0 and the last
 *                           group typically ends at index `n`. Length =
 *                           `num_groups`.
 * @param[in]  num_groups    Length of `group_idxs`.
 * @param      m             Rows of `x`.
 * @param      n             Rows of `y`.
 * @param      k             Cols of `x` and `y`.
 * @param      redOp         Reduction operator in the epilogue
 * @param      pairRedOp     Reduction operation on key value pairs
 * @param      sqrt          Whether to compute the squared or actual (i.e. sqrt) L2 norm.
 * @param      initOutBuffer Whether to initialize the output buffer
 *
 *
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT, typename KVPReduceOpT>
void masked_l2_nn_impl(raft::resources const& handle,
                       OutT* out,
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
                       ReduceOpT redOp,
                       KVPReduceOpT pairRedOp,
                       bool sqrt,
                       bool initOutBuffer)
{
  typedef typename linalg::Policy4x4<DataT, 1>::Policy P;

  static_assert(P::Mblk == 64, "masked_l2_nn_impl only supports a policy with 64 rows per block.");

  // Get stream and workspace memory resource
  auto stream = resource::get_cuda_stream(handle);
  auto ws_mr  = resource::get_workspace_resource(handle);

  // Acquire temporary buffers and initialize to zero:
  // 1) Adjacency matrix bitfield
  // 2) Workspace for fused nearest neighbor operation
  size_t m_div_64 = raft::ceildiv(m, IdxT(64));
  rmm::device_uvector<uint64_t> ws_adj64{m_div_64 * num_groups, stream, ws_mr};
  rmm::device_uvector<int> ws_fused_nn{size_t(m), stream, ws_mr};
  RAFT_CUDA_TRY(cudaMemsetAsync(ws_adj64.data(), 0, ws_adj64.size() * sizeof(uint64_t), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(ws_fused_nn.data(), 0, ws_fused_nn.size() * sizeof(int), stream));

  // Compress boolean adjacency matrix to bitfield.
  auto adj_view = raft::make_device_matrix_view<const bool, int>(adj, m, num_groups);
  auto adj64_view =
    raft::make_device_matrix_view<uint64_t, int>(ws_adj64.data(), m_div_64, num_groups);
  compress_to_bits(handle, adj_view, adj64_view);

  // Initialize output buffer with keyvalue pairs as determined by the reduction
  // operator (it will be called with maxVal).
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  if (initOutBuffer) {
    dim3 grid(raft::ceildiv<int>(m, P::Nthreads));
    dim3 block(P::Nthreads);

    initKernel<DataT, OutT, IdxT, ReduceOpT><<<grid, block, 0, stream>>>(out, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  // Accumulation operation lambda
  auto core_lambda = [] __device__(DataT & acc, DataT & x, DataT & y) { acc += x * y; };
  auto fin_op      = raft::identity_op{};

  auto kernel               = masked_l2_nn_kernel<DataT,
                                    OutT,
                                    IdxT,
                                    P,
                                    ReduceOpT,
                                    KVPReduceOpT,
                                    decltype(core_lambda),
                                    decltype(fin_op)>;
  constexpr size_t smemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
  dim3 block(P::Nthreads);
  dim3 grid = launchConfigGenerator<P>(m, n, smemSize, kernel);

  kernel<<<grid, block, smemSize, stream>>>(out,
                                            x,
                                            y,
                                            xn,
                                            yn,
                                            ws_adj64.data(),
                                            group_idxs,
                                            num_groups,
                                            m,
                                            n,
                                            k,
                                            sqrt,
                                            maxVal,
                                            ws_fused_nn.data(),
                                            redOp,
                                            pairRedOp,
                                            core_lambda,
                                            fin_op);

  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
