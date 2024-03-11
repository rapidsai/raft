/***************************************************************************************************
 * Copyright (c) 2017 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
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

#include <raft/distance/detail/fused_distance_nn/epilogue.cuh>
#include <raft/distance/detail/fused_distance_nn/persistent_gemm.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/kernel/default_gemm_universal.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/tensor.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////
/*
 * This configuration is used for float inputs with veclen(kAlignmentA/B) = 2 or 4,
 * ideal threadblock tile shape is 32x256x16 for such cases as there is no
 * registers spills for it.
 *
 */
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
  /// Epilogue output operator      - must satisfy concept of 'EpilogueWithBroadcastOp'
  typename EpilogueOutputOp,
  /// Number of stages used in the pipelined mainloop
  int Stages,
  /// data layout row/column major of inputs
  bool isRowMajor>
struct FusedDistanceNNGemm {
  // This struct is specialized for fp32/3xTF32

  /// Threadblock-level tile size (concept: GemmShape)
  // <- threadblock tile M = 32, N = 256, K = 16
  // this is more performant but note that for veclen = 1
  // this shape has register spills
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 256, 16>;

  // <- threadblock tile M = 32, N = 128, K = 16
  // this shape has high occupancy but less perf
  // this is less performant but this shape has *no* register spills
  // for any veclens(1, 2, 4)
  // using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;

  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  // <- warp tile M = 64, N = 64, K = 16
  // this is more performant for veclen 2,4.
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;

  //  this shape has high occupancy but less perf used for 32x128x16
  // using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;

  /// Warp-level tile size (concept: GemmShape)
  // This code section describes the size of MMA op
  // <- MMA Op tile M = 16, N = 8, K = 4
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  /// Operation performed by GEMM
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
  // using Operator = cutlass::arch::OpMultiplyAdd; // this runs only 1xTF32 for float inputs

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
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<
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
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,
                                               Epilogue,
                                               ThreadblockSwizzle,
                                               GroupScheduleMode::kDeviceOnly>;
};

/*
 * This configuration is used for float inputs with veclen(kAlignmentA/B) = 1,
 * ideal threadblock tile shape is 32x128x16 for such cases as there is no
 * registers spills for it.
 *
 */
template <
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
struct FusedDistanceNNGemm<float,  /// Element type for A matrix operand
                           1,      /// Layout type (veclen) for A matrix operand
                           float,  /// Element type for B matrix operand
                           1,      /// Layout type (veclen) for B matrix operand
                           ElementC_,
                           ElementAccumulator,
                           EpilogueOutputOp,
                           Stages,
                           isRowMajor> {
  // This struct is specialized for fp32/3xTF32
  using ElementA_ = float;
  using ElementB_ = float;

  /// Threadblock-level tile size (concept: GemmShape)
  // <- threadblock tile M = 32, N = 128, K = 16
  // this shape has high occupancy and no register spills for veclen = 1.
  using ThreadblockShape = cutlass::gemm::GemmShape<32, 128, 16>;

  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  // <- warp tile M = 32, N = 32, K = 16
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;

  /// Warp-level tile size (concept: GemmShape)
  // This code section describes the size of MMA op
  // <- MMA Op tile M = 16, N = 8, K = 4
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;

  /// Operation performed by GEMM
  using Operator = cutlass::arch::OpMultiplyAddFastF32;
  // using Operator = cutlass::arch::OpMultiplyAdd; // this runs only 1xTF32 for float inputs

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
                                                 1,
                                                 ElementB_,
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
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<
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
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,
                                               Epilogue,
                                               ThreadblockSwizzle,
                                               GroupScheduleMode::kDeviceOnly>;
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
struct FusedDistanceNNGemm<double,
                           kAlignmentA,
                           double,
                           kAlignmentB,
                           ElementC_,
                           ElementAccumulator,
                           EpilogueOutputOp,
                           Stages,
                           isRowMajor> {
  // Threadblock-level tile size (concept: GemmShape)
  // <- threadblock tile M = 64, N = 64, K = 16
  using ThreadblockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  // using ThreadblockShape = cutlass::gemm::GemmShape<16, 128, 16>;
  /// Warp-level tile size (concept: GemmShape)
  // This code section describes tile size a warp will compute
  // <- warp tile M = 32, N = 32, K = 16
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  // using WarpShape = cutlass::gemm::GemmShape<16, 32, 16>;
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
  using Epilogue = typename cutlass::epilogue::threadblock::FusedDistanceNNEpilogue<
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
  using GemmKernel = FusedDistanceNNPersistent<typename GemmBase::Mma,
                                               Epilogue,
                                               ThreadblockSwizzle,
                                               GroupScheduleMode::kDeviceOnly>;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass