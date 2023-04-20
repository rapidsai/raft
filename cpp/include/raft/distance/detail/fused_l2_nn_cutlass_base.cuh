/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

// We define CUTLASS_NAMESPACE in case
// RAFT cmake is not used
#ifndef CUTLASS_NAMESPACE
#define cutlass raft_cutlass
#endif

#include <rmm/device_uvector.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/device/gemm_grouped.h>

#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_view.h>

#include <raft/distance/detail/fused_l2_nn_epilogue_elementwise.cuh>
#include <raft/distance/detail/fused_l2_nn_gemm.h>
#include <raft/util/cutlass_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace distance {
namespace detail {

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename FinalLambda,
          typename DistanceFn,
          typename ReduceOpT,
          typename KVPReduceOpT>
void cutlassFusedL2NNKernel(const DataT* x,
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
                            FinalLambda fin_op,
                            DistanceFn dist_op,
                            ReduceOpT redOp,
                            KVPReduceOpT pairRedOp,
                            cudaStream_t stream)
{
  using EpilogueOutputOp =
    cutlass::epilogue::thread::FusedL2NNEpilogueElementwise<DataT,  // ElementC_
                                                            AccT,   // ElementAccumulator_
                                                            DataT,  // ElementCompute_
                                                            AccT,   // ElementZ_
                                                            OutT,   // ElementT_
                                                            1, //128 / cutlass::sizeof_bits<DataT>::value,      // Elements per access 1
                                                            DistanceFn,
                                                            FinalLambda,
                                                            ReduceOpT,
                                                            KVPReduceOpT>;
  constexpr int batch_count = 1;

  typename EpilogueOutputOp::Params epilog_op_param(dist_op, fin_op, redOp, pairRedOp, mutexes);

  const DataT *a, *b;

  IdxT gemm_lda, gemm_ldb;

  // Number of pipelines you want to use
  constexpr int NumStages = 3;
  // Alignment
  constexpr int Alignment = VecLen;

  // default initialize problem size with row major inputs
  //auto problem_size = cutlass::gemm::GemmCoord(n, m, k);
  auto problem_size = cutlass::gemm::GemmCoord(m, n, k);

  constexpr bool isRowMajor = true;

  using cutlassDistKernel =
    typename cutlass::gemm::kernel::FusedL2NNGemm<DataT,
                                                  Alignment,
                                                  DataT,
                                                  Alignment,
                                                  AccT,
                                                  AccT,
                                                  EpilogueOutputOp,
                                                  NumStages,  // Number of pipeline stages
                                                  isRowMajor>::GemmKernel;

#if 0
  using cutlassDist = cutlass::gemm::device::GemmUniversalAdapter<cutlassDistKernel>;

  a        = y;
  b        = x;
  gemm_lda = ldb;
  gemm_ldb = lda;
  constexpr auto mode = cutlass::gemm::GemmUniversalMode::kGemm;

  typename cutlassDist::Arguments arguments{
    mode,
    problem_size,
    batch_count,
    epilog_op_param,
    a,
    b,
    xn,          // C matrix eq vector param, which here is A norm
    nullptr,     // tensor_Z,
    (DataT*)yn,  // this is broadcast vec, which is required to be non-const param
    dOutput,     // Output distance matrix
    (int64_t)0,  // batch stride A
    (int64_t)0,  // batch stride B
    (int64_t)0,  // batch stride Norm A
    (int64_t)0,
    (int64_t)0,         // batch stride Norm B
    (int64_t)0,         // batch stride Output
    (int64_t)gemm_lda,  // stride A
    (int64_t)gemm_ldb,  // stride B
    1,                  // stride A norm
    0,                  // this is no-op for Z
    0,                  // This must be zero
    (int64_t)ldd        // stride Output matrix
  };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = cutlassDist::get_workspace_size(arguments);
  // Allocate workspace memory
  rmm::device_uvector<uint8_t> workspace(workspace_size, stream);
  // Instantiate CUTLASS kernel depending on templates
  cutlassDist cutlassDist_op;
  // Check the problem size is supported or not
  RAFT_CUTLASS_TRY(cutlassDist_op.can_implement(arguments));
  // Initialize CUTLASS kernel with arguments and workspace pointer
  RAFT_CUTLASS_TRY(cutlassDist_op.initialize(arguments, workspace.data(), stream));
  // Launch initialized CUTLASS kernel
  RAFT_CUTLASS_TRY(cutlassDist_op());
#else


  using cutlassDist = cutlass::gemm::device::GemmGrouped<cutlassDistKernel>;

  a        = x;
  b        = y;
  gemm_lda = lda;
  gemm_ldb = ldb;
  int num_blocks = cutlassDist::maximum_active_blocks();
  int num_sms = raft::getMultiProcessorCount();
  num_blocks = num_blocks * num_sms;
  auto thread_blocks = std::max(num_blocks, int((problem_size.m() - 1 + cutlassDistKernel::Mma::Shape::kM)/ cutlassDistKernel::Mma::Shape::kM));
  //printf("num blocks = %d sms = %d thread_blocks_sel = %d shapekM = %d\n", num_blocks, num_sms, (int)thread_blocks,  (int)cutlassDistKernel::Mma::Shape::kM);
  //rmm::device_uvector<decltype(problem_size)> problem_sizes(sizeof(decltype(problem_size)), stream);
  //raft::copy(problem_sizes.data(), &problem_size, 1, stream);
  typename cutlassDist::Arguments arguments{
    //problem_sizes.data(),
    problem_size,
    batch_count,
    thread_blocks,
    epilog_op_param,
    a,
    b,
    xn,          // C matrix eq vector param, which here is A norm
    (DataT*)yn,  // this is broadcast vec, which is required to be non-const param
    dOutput,     // Output distance matrix
    (int64_t)gemm_lda,  // stride A
    (int64_t)gemm_ldb,  // stride B
    (int64_t)1,                  // stride A norm
    (int64_t)ldd        // stride Output matrix
  };

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = cutlassDist::get_workspace_size(arguments);
  // Allocate workspace memory
  rmm::device_uvector<uint8_t> workspace(workspace_size, stream);
  // Instantiate CUTLASS kernel depending on templates
  cutlassDist cutlassDist_op;
  // Check the problem size is supported or not
  RAFT_CUTLASS_TRY(cutlassDist_op.can_implement(arguments));
  // Initialize CUTLASS kernel with arguments and workspace pointer
  RAFT_CUTLASS_TRY(cutlassDist_op.initialize(arguments, workspace.data(), stream));
  // Launch initialized CUTLASS kernel
  RAFT_CUTLASS_TRY(cutlassDist_op.run(stream));
#endif

}

};  // namespace detail
};  // namespace distance
};  // namespace raft

#pragma GCC diagnostic pop
