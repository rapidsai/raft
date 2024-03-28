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
#include <raft/distance/detail/distance_ops/cosine.cuh>  // ops::l2_exp_distance_op
#include <raft/distance/detail/fused_distance_nn/cutlass_base.cuh>
#include <raft/distance/detail/fused_distance_nn/helper_structs.cuh>
#include <raft/distance/detail/fused_distance_nn/simt_kernel.cuh>
#include <raft/distance/detail/pairwise_distance_base.cuh>  // PairwiseDistances
#include <raft/linalg/contractions.cuh>                     // Policy
#include <raft/util/arch.cuh>                               // raft::util::arch::SM_*
#include <raft/util/cuda_utils.cuh>                         // raft::ceildiv, raft::shfl

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
void fusedCosineNN(OutT* min,
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
                   cudaStream_t stream)
{
  // The kernel policy is determined by fusedL2NN.
  typedef Policy P;

  dim3 blk(P::Nthreads);
  constexpr auto maxVal = std::numeric_limits<DataT>::max();
  typedef KeyValuePair<IdxT, DataT> KVPair;

  namespace arch = raft::util::arch;
  using AccT     = DataT;
  ops::cosine_distance_op<DataT, AccT, IdxT> distance_op{};

  raft::identity_op fin_op{};

  auto kernel = fusedDistanceNNkernel<DataT,
                                      OutT,
                                      IdxT,
                                      P,
                                      ReduceOpT,
                                      KVPReduceOpT,
                                      decltype(distance_op),
                                      decltype(fin_op)>;

  // Get pointer to fp32 SIMT kernel to determine the runtime architecture of the
  // current system. Other methods to determine the architecture (that do not
  // require a pointer) can be error prone. See:
  // https://github.com/NVIDIA/cub/issues/545
  void* kernel_ptr   = reinterpret_cast<void*>(kernel);
  auto runtime_arch  = arch::kernel_virtual_arch(kernel_ptr);
  auto cutlass_range = arch::SM_range(arch::SM_80(), arch::SM_future());

  if (cutlass_range.contains(runtime_arch)) {
    // If device is SM_80 or later, use CUTLASS-based kernel.
    using cosineOp              = raft::distance::detail::ops::cosine_cutlass_op<DataT, DataT>;
    using kvp_cg_min_reduce_op_ = kvp_cg_min_reduce_op<DataT, IdxT, OutT>;
    kvp_cg_min_reduce_op_ cg_reduce_op;
    cosineOp cosine_dist_op;

    IdxT lda, ldb, ldd;
    lda = k, ldb = k, ldd = n;

    cutlassFusedDistanceNN<DataT,
                           DataT,
                           OutT,
                           IdxT,
                           P::Veclen,
                           decltype(cg_reduce_op),
                           decltype(cosine_dist_op),
                           ReduceOpT,
                           KVPReduceOpT>(x,
                                         y,
                                         xn,
                                         yn,
                                         m,
                                         n,
                                         k,
                                         lda,
                                         ldb,
                                         ldd,
                                         min,
                                         workspace,
                                         cg_reduce_op,
                                         cosine_dist_op,
                                         redOp,
                                         pairRedOp,
                                         stream);
  } else {
    // If device less than SM_80, use fp32 SIMT kernel.
    constexpr size_t shmemSize = P::SmemSize + ((P::Mblk + P::Nblk) * sizeof(DataT));
    dim3 grid                  = launchConfigGenerator<P>(m, n, shmemSize, kernel);

    kernel<<<grid, blk, shmemSize, stream>>>(
      min, x, y, xn, yn, m, n, k, maxVal, workspace, redOp, pairRedOp, distance_op, fin_op);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
