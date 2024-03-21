/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/kvp.hpp>                             // raft::KeyValuePair
#include <raft/core/operators.hpp>                       // raft::identity_op
#include <raft/distance/detail/distance_ops/l2_exp.cuh>  // ops::l2_exp_distance_op
#include <raft/distance/detail/fused_distance_nn/cutlass_base.cuh>
#include <raft/distance/detail/fused_distance_nn/fused_cosine_nn.cuh>
#include <raft/distance/detail/fused_distance_nn/fused_l2_nn.cuh>
#include <raft/distance/detail/fused_distance_nn/helper_structs.cuh>
#include <raft/distance/detail/fused_distance_nn/simt_kernel.cuh>
#include <raft/distance/detail/pairwise_distance_base.cuh>  // PairwiseDistances
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/contractions.cuh>  // Policy
#include <raft/util/arch.cuh>            // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>      // raft::ceildiv, raft::shfl

#include <cstddef>  // size_t
#include <limits>   // std::numeric_limits

namespace raft {
namespace distance {

namespace detail {

template <typename DataT,
          typename OutT,
          typename IdxT,
          typename Policy,
          typename ReduceOpT,
          typename KVPReduceOpT>
void fusedDistanceNNImpl(OutT* min,
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
                         bool isRowMajor,
                         raft::distance::DistanceType metric,
                         float metric_arg,
                         cudaStream_t stream)
{
  // The kernel policy is determined by fusedDistanceNN.
  typedef Policy P;

  dim3 blk(P::Nthreads);
  auto nblks            = raft::ceildiv<int>(m, P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef KeyValuePair<IdxT, DataT> KVPair;

  RAFT_CUDA_TRY(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, P::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    RAFT_CUDA_TRY(cudaGetLastError());
  }

  switch (metric) {
    case raft::distance::DistanceType::CosineExpanded:
      fusedCosineNN<DataT, OutT, IdxT, P, ReduceOpT, KVPReduceOpT>(
        min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, sqrt, stream);
      break;
    case raft::distance::DistanceType::L2SqrtExpanded:
    case raft::distance::DistanceType::L2Expanded:
      // initOutBuffer is take care by fusedDistanceNNImpl() so we set it false to fusedL2NNImpl.
      fusedL2NNImpl<DataT, OutT, IdxT, P, ReduceOpT, KVPReduceOpT>(
        min, x, y, xn, yn, m, n, k, workspace, redOp, pairRedOp, sqrt, false, stream);
      break;
    default: assert("only cosine/l2 metric is supported with fusedDistanceNN\n"); break;
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
