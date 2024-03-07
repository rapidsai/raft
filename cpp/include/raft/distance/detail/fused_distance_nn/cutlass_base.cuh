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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Wtautological-compare"

// We define CUTLASS_NAMESPACE in case
// RAFT cmake is not used
#ifndef CUTLASS_NAMESPACE
#define cutlass raft_cutlass
#endif

#include <raft/distance/detail/fused_distance_nn/epilogue_elementwise.cuh>  // FusedDistanceNNEpilogueElementwise
#include <raft/distance/detail/fused_distance_nn/gemm.h>                    // FusedDistanceNNGemm
#include <raft/util/cudart_utils.hpp>   // getMultiProcessorCount
#include <raft/util/cutlass_utils.cuh>  // RAFT_CUTLASS_TRY

#include <rmm/device_uvector.hpp>

#include <cuda/semaphore>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_view.h>

namespace raft {
namespace distance {
namespace detail {

template <typename IdxT>
RAFT_KERNEL initBinMutexKernel(cuda::binary_semaphore<cuda::thread_scope_device>* mut, IdxT m)
{
  auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid < m) { mut[tid].release(); }
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename CGReduceOpT,
          typename DistanceFn,
          typename ReduceOpT,
          typename KVPReduceOpT>
void cutlassFusedDistanceNN(const DataT* x,
                            const DataT* y,
                            const DataT* xn,
                            const DataT* yn,
                            IdxT m,
                            IdxT n,
                            IdxT k,
                            IdxT lda,
                            IdxT ldb,
                            IdxT ldd,
                            OutT* dOutput,
                            int* mutexes,
                            CGReduceOpT cg_reduce_op,
                            DistanceFn dist_op,
                            ReduceOpT redOp,
                            KVPReduceOpT pairRedOp,
                            cudaStream_t stream)
{
  using EpilogueOutputOp = cutlass::epilogue::thread::FusedDistanceNNEpilogueElementwise<
    DataT,  // ElementC_
    AccT,   // ElementAccumulator_
    DataT,  // ElementCompute_
    AccT,   // ElementZ_
    OutT,   // ElementT_
    // 128 / cutlass::sizeof_bits<DataT>::value,
    1,  // Elements per access 1
    DistanceFn,
    CGReduceOpT,
    ReduceOpT,
    KVPReduceOpT>;
  constexpr int batch_count = 1;

  rmm::device_uvector<cuda::binary_semaphore<cuda::thread_scope_device>> bin_mutex(m, stream);

  int blks_ = (m / 256) + 1;

  initBinMutexKernel<<<blks_, 256, 0, stream>>>(bin_mutex.data(), m);

  typename EpilogueOutputOp::Params epilog_op_param(
    dist_op, cg_reduce_op, redOp, pairRedOp, mutexes, bin_mutex.data());

  // Number of pipelines you want to use
  constexpr int NumStages = 3;
  // Alignment
  constexpr int Alignment = VecLen;

  // default initialize problem size with row major inputs
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k);

  constexpr bool isRowMajor = true;

  using fusedDistanceNNKernel =
    typename cutlass::gemm::kernel::FusedDistanceNNGemm<DataT,
                                                        Alignment,
                                                        DataT,
                                                        Alignment,
                                                        AccT,
                                                        AccT,
                                                        EpilogueOutputOp,
                                                        NumStages,  // Number of pipeline stages
                                                        isRowMajor>::GemmKernel;

  using fusedDistanceNN = cutlass::gemm::device::GemmGrouped<fusedDistanceNNKernel>;

  int num_blocks_per_sm   = fusedDistanceNN::maximum_active_blocks();
  int num_sms             = raft::getMultiProcessorCount();
  int full_wave           = num_blocks_per_sm * num_sms;
  constexpr int mmaShapeM = fusedDistanceNNKernel::Mma::Shape::kM;
  constexpr int mmaShapeN = fusedDistanceNNKernel::Mma::Shape::kN;
  int columnTiles         = (problem_size.n() - 1 + mmaShapeN) / mmaShapeN;
  int rowTiles            = (problem_size.m() - 1 + mmaShapeM) / mmaShapeM;
  int totalTiles          = columnTiles * rowTiles;
  int thread_blocks =
    rowTiles < full_wave ? (totalTiles < full_wave ? totalTiles : full_wave) : rowTiles;

  typename fusedDistanceNN::Arguments arguments{
    problem_size,
    batch_count,  // num of problems.
    thread_blocks,
    epilog_op_param,
    x,
    y,
    xn,            // C matrix eq vector param, which here is A norm
    (DataT*)yn,    // this is broadcast vec, which is required to be non-const param
    dOutput,       // Output distance matrix
    (int64_t)lda,  // stride A
    (int64_t)ldb,  // stride B
    (int64_t)1,    // stride A norm
    (int64_t)ldd   // stride Output matrix
  };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = fusedDistanceNN::get_workspace_size(arguments);
  // Allocate workspace memory
  rmm::device_uvector<uint8_t> workspace(workspace_size, stream);
  // Instantiate CUTLASS kernel depending on templates
  fusedDistanceNN fusedDistanceNN_op;
  // Check the problem size is supported or not
  RAFT_CUTLASS_TRY(fusedDistanceNN_op.can_implement(arguments));
  // Initialize CUTLASS kernel with arguments and workspace pointer
  RAFT_CUTLASS_TRY(fusedDistanceNN_op.initialize(arguments, workspace.data(), stream));
  // Launch initialized CUTLASS kernel
  RAFT_CUTLASS_TRY(fusedDistanceNN_op.run(stream));
}

};  // namespace detail
};  // namespace distance
};  // namespace raft

#pragma GCC diagnostic pop
