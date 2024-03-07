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

#include "./pairwise_distance_epilogue.h"

#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/gemm/kernel/gemm_with_fused_epilogue.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  /// Element type for A matrix operand
  typename ElementA_,
  /// Layout type for A matrix operand
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB_,
  /// Layout type for B matrix operand
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Element type for final output
  // typename ElementOutT,
  /// Epilogue output operator      - must satisfy concept of 'EpilogueWithBroadcastOp'
  typename EpilogueOutputOp,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// data layout row/column major of inputs
  bool isRowMajor>
struct PairwiseDistanceGemm {
  // This struct is specialized for fp32/3xTF32

  /// Threadblock-level tile size (concept: GemmShape)
  using ThreadblockShape =
    cutlass::gemm::GemmShape<128, 128, 16>;  // <- threadblock tile M = 128, N = 128, K = 16
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 16>;  // <- warp tile M = 64, N = 64, K = 16
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes the size of MMA op
  using InstructionShape =
    cutlass::gemm::GemmShape<16, 8, 4>;  // <- MMA Op tile M = 16, N = 8, K = 4

  /// Operation performed by GEMM
  using Operator = cutlass::arch::OpMultiplyAddFastF32;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using ArchTag = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  /// Threadblock-level swizzling operator
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  /// data layout for final output matrix.
  // we keep this same layout even for column major inputs
  using LayoutOutput = cutlass::layout::RowMajor;

  typedef typename std::conditional<isRowMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor>::type NormXLayout;

  typedef typename std::
    conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;

  typedef typename std::
    conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;

  using GemmBase = typename DefaultGemmUniversal<ElementA_,
                                                 LayoutA_,
                                                 cutlass::ComplexTransform::kNone,
                                                 kAlignmentA,
                                                 ElementB_,
                                                 LayoutB_,
                                                 cutlass::ComplexTransform::kNone,
                                                 kAlignmentB,
                                                 ElementC_,
                                                 LayoutOutput,
                                                 ElementAccumulator,
                                                 OperatorClass,
                                                 ArchTag,
                                                 ThreadblockShape,
                                                 WarpShape,
                                                 InstructionShape,
                                                 EpilogueOutputOp,
                                                 ThreadblockSwizzle,
                                                 Stages,
                                                 Operator>::GemmKernel;

  // Replace epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::PairwiseDistanceEpilogue<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    GemmBase::Epilogue::kPartitionsK,
    ElementAccumulator,
    typename EpilogueOutputOp::ElementT,
    ElementAccumulator,
    EpilogueOutputOp,
    NormXLayout,
    GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel = GemmWithFusedEpilogue<typename GemmBase::Mma, Epilogue, ThreadblockSwizzle>;
};

template <
  /// Layout type for A matrix operand
  int kAlignmentA,
  /// Layout type for B matrix operand
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC_,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Epilogue output operator      - must satisfy concept of 'EpilogueWithBroadcastOp'
  typename EpilogueOutputOp,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// data layout row/column major of inputs
  bool isRowMajor>
struct PairwiseDistanceGemm<double,
                            kAlignmentA,
                            double,
                            kAlignmentB,
                            ElementC_,
                            ElementAccumulator,
                            EpilogueOutputOp,
                            Stages,
                            isRowMajor> {
  // using Transform = cutlass::ComplexTransform::kNone;
  // Threadblock-level tile size (concept: GemmShape)
  using ThreadblockShape =
    cutlass::gemm::GemmShape<64, 64, 16>;  // <- threadblock tile M = 64, N = 64, K = 16
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;  // <- warp tile M = 32, N = 32, K = 16
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes the size of MMA op
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

  // Operation performed by GEMM
  using Operator = cutlass::arch::OpMultiplyAdd;
  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU
  // SM
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using ArchTag = cutlass::arch::Sm80;

  // This code section describes how threadblocks are scheduled on GPU
  /// Threadblock-level swizzling operator
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  /// data layout for final output matrix.
  // we keep this same layout even for column major inputs
  using LayoutOutput = cutlass::layout::RowMajor;

  typedef typename std::conditional<isRowMajor,
                                    cutlass::layout::RowMajor,
                                    cutlass::layout::ColumnMajor>::type NormXLayout;

  typedef typename std::
    conditional<isRowMajor, cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>::type LayoutA_;

  typedef typename std::
    conditional<isRowMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>::type LayoutB_;

  using GemmBase = typename DefaultGemmUniversal<double,
                                                 LayoutA_,
                                                 cutlass::ComplexTransform::kNone,
                                                 1,
                                                 double,
                                                 LayoutB_,
                                                 cutlass::ComplexTransform::kNone,
                                                 1,
                                                 ElementC_,
                                                 LayoutOutput,
                                                 ElementAccumulator,
                                                 OperatorClass,
                                                 ArchTag,
                                                 ThreadblockShape,
                                                 WarpShape,
                                                 InstructionShape,
                                                 EpilogueOutputOp,
                                                 ThreadblockSwizzle,
                                                 Stages,
                                                 Operator>::GemmKernel;

  // Replace epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::PairwiseDistanceEpilogue<
    typename GemmBase::Epilogue::Shape,
    typename GemmBase::Epilogue::WarpMmaOperator,
    GemmBase::Epilogue::kPartitionsK,
    ElementC_,
    typename EpilogueOutputOp::ElementT,
    ElementC_,
    EpilogueOutputOp,
    NormXLayout,
    GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  // Compose the GEMM kernel
  using GemmKernel = GemmWithFusedEpilogue<typename GemmBase::Mma, Epilogue, ThreadblockSwizzle>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass