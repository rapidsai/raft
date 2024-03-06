/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "pairwise_distance_epilogue_elementwise.h"
#include "pairwise_distance_gemm.h"

#include <raft/distance/detail/distance_ops/cutlass.cuh>
#include <raft/util/cutlass_utils.cuh>

#include <rmm/device_uvector.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/tensor_view.h>

#include <type_traits>

namespace raft {
namespace distance {
namespace detail {

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename FinalLambda,
          typename OpT,
          bool isRowMajor>
std::enable_if_t<ops::has_cutlass_op<OpT>::value> cutlassDistanceKernel(const DataT* x,
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
                                                                        FinalLambda fin_op,
                                                                        OpT distance_op,
                                                                        cudaStream_t stream)
{
  static_assert(!(std::is_same<OutT, bool>::value),
                "OutType bool is not supported use uint8_t instead");

  auto dist_op     = distance_op.get_cutlass_op();
  using DistanceFn = decltype(dist_op);
  using EpilogueOutputOp =
    cutlass::epilogue::thread::PairwiseDistanceEpilogueElementwise<DataT,  // ElementC_
                                                                   AccT,   // ElementAccumulator_
                                                                   DataT,  // ElementCompute_
                                                                   AccT,   // ElementZ_
                                                                   OutT,   // ElementT_
                                                                   1,      // Elements per access 1
                                                                   DistanceFn,
                                                                   FinalLambda>;
  constexpr int batch_count = 1;

  constexpr auto mode = cutlass::gemm::GemmUniversalMode::kGemm;

  typename EpilogueOutputOp::Params epilog_op_param(dist_op, fin_op);

  // Number of pipelines you want to use
  constexpr int NumStages = 3;
  // Alignment
  constexpr int Alignment = VecLen;

  using cutlassDistKernel =
    typename cutlass::gemm::kernel::PairwiseDistanceGemm<DataT,
                                                         Alignment,
                                                         DataT,
                                                         Alignment,
                                                         AccT,
                                                         AccT,
                                                         EpilogueOutputOp,
                                                         NumStages,  // Number of pipeline stages
                                                         isRowMajor>::GemmKernel;

  using cutlassDist = cutlass::gemm::device::GemmUniversalAdapter<cutlassDistKernel>;

  constexpr uint32_t gridYZMax      = ((1 << (sizeof(uint16_t) * 8)) - 1);
  constexpr uint32_t max_batch_size = gridYZMax * cutlassDistKernel::ThreadblockShape::kN;
  IdxT numNbatches                  = (n - 1 + max_batch_size) / max_batch_size;

  for (IdxT i = 0; i < numNbatches; i++) {
    const DataT *a, *b;
    IdxT gemm_lda, gemm_ldb;
    size_t offsetN = i * max_batch_size;

    if constexpr (isRowMajor) {
      gemm_lda = ldb;
      gemm_ldb = lda;
      a        = y + offsetN * gemm_lda;
      b        = x;
    } else {
      gemm_lda = lda;
      gemm_ldb = ldb;
      a        = x;
      b        = y + offsetN;
    }
    IdxT chunkN   = (i + 1) * max_batch_size;
    IdxT currentN = (chunkN < n) ? max_batch_size : (n - offsetN);

    // default initialize problem size with row major inputs
    auto problem_size = isRowMajor ? cutlass::gemm::GemmCoord(currentN, m, k)
                                   : cutlass::gemm::GemmCoord(m, currentN, k);

    typename cutlassDist::Arguments arguments{
      mode,
      problem_size,
      batch_count,
      epilog_op_param,
      a,
      b,
      xn,                    // C matrix eq vector param, which here is A norm
      nullptr,               // tensor_Z,
      (DataT*)yn + offsetN,  // this is broadcast vec, which is required to be non-const param
      dOutput + offsetN,     // Output distance matrix
      (int64_t)0,            // batch stride A
      (int64_t)0,            // batch stride B
      (int64_t)0,            // batch stride Norm A
      (int64_t)0,
      (int64_t)0,  // batch stride Norm B
      (int64_t)0,  // batch stride Output
      gemm_lda,    // stride A
      gemm_ldb,    // stride B
      1,           // stride A norm
      0,           // this is no-op for Z
      0,           // This must be zero
      ldd          // stride Output matrix
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
    RAFT_CUTLASS_TRY(cutlassDist_op(stream));
  }
}

};  // namespace detail
};  // namespace distance
};  // namespace raft

#pragma GCC diagnostic pop
