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

/*! \file
    \brief Problem visitor for grouped GEMMs
This file contains heavily customized version of GemmGrouped from CUTLASS 2.10.0
(https://github.com/NVIDIA/cutlass/blob/v2.10.0/include/cutlass/gemm/kernel/gemm_grouped.h)

Changes:
- adds support for only single problem size to be launched persistently
  where each threablock processes more than one tile of the same problem.
*/

#pragma once

#include <cutlass/complex.h>
#include <cutlass/cutlass.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_grouped_problem_visitor.h>
#include <cutlass/gemm/kernel/gemm_transpose_operands.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/matrix_coord.h>
#include <cutlass/semaphore.h>
#include <cutlass/trace.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Mma_,                         ///! Threadblock-scoped matrix multiply-accumulate
          typename Epilogue_,                    ///! Epilogue
          typename ThreadblockSwizzle_,          ///! Threadblock swizzling function
          GroupScheduleMode GroupScheduleMode_,  ///! Type of scheduling to perform
          bool Transposed = false>
struct FusedDistanceNNPersistent {
 public:
  using Mma                                         = Mma_;
  using Epilogue                                    = Epilogue_;
  using EpilogueOutputOp                            = typename Epilogue::OutputOp;
  using ThreadblockSwizzle                          = ThreadblockSwizzle_;
  static GroupScheduleMode const kGroupScheduleMode = GroupScheduleMode_;
  static bool const kTransposed                     = Transposed;

  // Optional transpose
  using MapArguments = kernel::detail::MapArguments<typename Mma::IteratorA::Element,
                                                    typename Mma::IteratorA::Layout,
                                                    Mma::kTransformA,
                                                    Mma::IteratorA::AccessType::kElements,
                                                    typename Mma::IteratorB::Element,
                                                    typename Mma::IteratorB::Layout,
                                                    Mma::kTransformB,
                                                    Mma::IteratorB::AccessType::kElements,
                                                    typename Mma::LayoutC,
                                                    kTransposed>;

  // Public-facing type definitions related to operand element type, layout, and complex conjugate
  // operation. Must interact with the 'kTransposed' notion.
  using ElementA = typename MapArguments::ElementA;
  using LayoutA  = typename MapArguments::LayoutA;
  using ElementB = typename MapArguments::ElementB;
  using LayoutB  = typename MapArguments::LayoutB;
  using ElementC = typename Epilogue::OutputTileIterator::Element;
  using LayoutC  = typename MapArguments::LayoutC;

  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;

  // Type definitions about the mainloop.
  using Operator         = typename Mma::Operator;
  using OperatorClass    = typename Mma::Operator::OperatorClass;
  using ThreadblockShape = typename Mma::Shape;
  using WarpShape        = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  using ArchTag          = typename Mma::ArchTag;

  static int const kStages     = Mma::kStages;
  static int const kAlignmentA = MapArguments::kAlignmentA;
  static int const kAlignmentB = MapArguments::kAlignmentB;
  static int const kAlignmentC = Epilogue::OutputTileIterator::kElementsPerAccess;

  /// Warp count (concept: GemmShape)
  using WarpCount               = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  using ProblemVisitor = GemmGroupedProblemVisitor<ThreadblockShape,
                                                   kGroupScheduleMode,
                                                   kThreadCount,
                                                   kThreadCount,
                                                   kTransposed>;

  //
  // Structures
  //

  struct temp_problem_visitor {
    int problem_count;

    CUTLASS_HOST_DEVICE temp_problem_visitor() : problem_count(0){};
    CUTLASS_HOST_DEVICE temp_problem_visitor(int problem_count_) : problem_count(problem_count_){};
  };

  /// Argument structure
  struct Arguments {
    //
    // Data members
    //
    GemmCoord problem_sizes;
    temp_problem_visitor problem_visitor;
    int problem_count;
    int threadblock_count;

    typename EpilogueOutputOp::Params output_op;

    void const* ptr_A;
    void const* ptr_B;
    void const* ptr_C;
    void* ptr_Vector;
    void* ptr_Tensor;

    typename LayoutA::Stride::Index lda;
    typename LayoutB::Stride::Index ldb;
    typename LayoutC::Stride::Index ldc;
    typename LayoutC::Stride::Index ldt;

    // Only used by device-level operator
    GemmCoord* host_problem_sizes;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments()
      : threadblock_count(0),
        ptr_A(nullptr),
        ptr_B(nullptr),
        ptr_C(nullptr),
        ptr_Vector(nullptr),
        ptr_Tensor(nullptr),
        lda(0),
        ldb(0),
        ldc(0),
        ldt(0),
        host_problem_sizes(nullptr)
    {
    }

    /// Ctor
    CUTLASS_HOST_DEVICE
    Arguments(GemmCoord problem_sizes,
              int problem_count,
              int threadblock_count,
              typename EpilogueOutputOp::Params output_op,
              void const* ptr_A,
              void const* ptr_B,
              void const* ptr_C,
              void* ptr_Vector,
              void* ptr_Tensor,
              typename LayoutA::Stride::Index lda,
              typename LayoutB::Stride::Index ldb,
              typename LayoutC::Stride::Index ldc,
              typename LayoutC::Stride::Index ldt,
              GemmCoord* host_problem_sizes = nullptr)
      : problem_sizes(problem_sizes),
        threadblock_count(threadblock_count),
        output_op(output_op),
        ptr_A(ptr_A),
        ptr_B(ptr_B),
        ptr_C(ptr_C),
        ptr_Vector(ptr_Vector),
        ptr_Tensor(ptr_Tensor),
        lda(lda),
        ldb(ldb),
        ldc(ldc),
        ldt(ldt),
        host_problem_sizes(host_problem_sizes)
    {
      problem_visitor.problem_count = problem_count;
    }
  };

  //
  // Structure for precomputing values in host memory and passing to kernels
  //

  /// Parameters structure
  struct Params {
    temp_problem_visitor problem_visitor;
    int threadblock_count;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::TensorTileIterator::Params params_Tensor;

    typename EpilogueOutputOp::Params output_op;

    void* ptr_A;
    void* ptr_B;
    void* ptr_C;
    void* ptr_Vector;
    void* ptr_Tensor;

    GemmCoord problem_size;
    typename LayoutA::Stride::Index lda;
    typename LayoutB::Stride::Index ldb;
    typename LayoutC::Stride::Index ldc;
    typename LayoutC::Stride::Index ldt;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params()
      : params_A(0),
        params_B(0),
        params_C(0),
        ptr_A(nullptr),
        ptr_B(nullptr),
        ptr_C(nullptr),
        ptr_Vector(nullptr),
        ptr_Tensor(nullptr),
        lda(0),
        ldb(0),
        ldc(0),
        ldt(0)
    {
    }

    CUTLASS_HOST_DEVICE
    Params(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
      : problem_size(args.problem_sizes),
        threadblock_count(args.threadblock_count),
        output_op(args.output_op),
        params_A(args.lda),
        params_B(args.ldb),
        params_C(args.ldc),
        // Here we pass additional user args via args.output_op
        // to the reduction output tile iterator
        params_Tensor(args.ldt, args.output_op),
        ptr_A(const_cast<void*>(args.ptr_A)),
        ptr_B(const_cast<void*>(args.ptr_B)),
        ptr_C(const_cast<void*>(args.ptr_C)),
        ptr_Vector(args.ptr_Vector),
        ptr_Tensor(args.ptr_Tensor),
        lda(args.lda),
        ldb(args.ldb),
        ldc(args.ldc),
        ldt(args.ldt)
    {
      problem_visitor.problem_count = args.problem_visitor.problem_count;
    }

    CUTLASS_HOST_DEVICE
    void update(Arguments const& args, void* workspace = nullptr, int tile_count = 0)
    {
      threadblock_count = args.threadblock_count;
      output_op         = args.output_op;
      ptr_A             = const_cast<void*>(args.ptr_A);
      ptr_B             = const_cast<void*>(args.ptr_B);
      ptr_C             = const_cast<void*>(args.ptr_C);
      ptr_Vector        = args.ptr_Vector;
      ptr_Tensor        = args.ptr_Tensor;
      lda               = args.lda;
      ldb               = args.ldb;
      ldc               = args.ldc;
      ldt               = args.ldt;

      problem_size = args.problem_sizes;
    }
  };

  /// Shared memory storage structure
  struct SharedStorage {
    union {
      typename Mma::SharedStorage main_loop;
      typename Epilogue::SharedStorage epilogue;
    } kernel;

    typename Epilogue::TensorTileIterator::SharedStorage reduced_store;
    typename Epilogue::OutputTileIterator::SharedStorage rownorm_store;
  };

 public:
  //
  // Methods
  //

  CUTLASS_DEVICE
  FusedDistanceNNPersistent() {}

  /// Determines whether kernel satisfies alignment
  static Status can_implement(cutlass::gemm::GemmCoord const& problem_size)
  {
    return Status::kSuccess;
  }

  static Status can_implement(Arguments const& args) { return Status::kSuccess; }

  static size_t get_extra_workspace_size(Arguments const& args,
                                         cutlass::gemm::GemmCoord const& grid_tiled_shape)
  {
    return 0;
  }

  CUTLASS_DEVICE
  static uint32_t tile_count(const cutlass::MatrixCoord& grid)
  {
    return grid.row() * grid.column();
  }

  /// Get the grid shape
  CUTLASS_DEVICE
  static cutlass::MatrixCoord grid_shape(const cutlass::gemm::GemmCoord& problem)
  {
    return cutlass::MatrixCoord(((problem.m() - 1 + ThreadblockShape::kM) / ThreadblockShape::kM),
                                ((problem.n() - 1 + ThreadblockShape::kN) / ThreadblockShape::kN));
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage)
  {
#if __CUDA_ARCH__ >= 800
    //
    // These types shadow the type-level definitions and support the ability to implement
    // a 'transposed' GEMM that computes the transposed problems.
    //
    using ElementA = typename Mma::IteratorA::Element;
    using LayoutA  = typename Mma::IteratorA::Layout;
    using ElementB = typename Mma::IteratorB::Element;
    using LayoutB  = typename Mma::IteratorB::Layout;
    using ElementC = typename Epilogue::OutputTileIterator::Element;
    using LayoutC  = typename Epilogue::OutputTileIterator::Layout;

    const GemmCoord& problem_size    = params.problem_size;
    const auto grid_shape_           = grid_shape(problem_size);
    const uint32_t problem_chunk     = (tile_count(grid_shape_) - 1 + gridDim.x) / gridDim.x;
    const uint32_t problem_chunk_end = blockIdx.x * problem_chunk + problem_chunk;
    typename LayoutB::Index column =
      ((blockIdx.x * problem_chunk) % grid_shape_.column()) * Mma::Shape::kN;

    typename LayoutB::Index row =
      ((blockIdx.x * problem_chunk) / grid_shape_.column()) * Mma::Shape::kM;
    if (column) {
      shared_storage.reduced_store.initSmem(params.output_op);
      shared_storage.rownorm_store.initSmem(params.ptr_C, problem_size.m(), row, sizeof(ElementC));
    }

    // Outer 'persistent' loop to iterate over tiles
    for (uint32_t tile_idx = blockIdx.x * problem_chunk; tile_idx < problem_chunk_end; tile_idx++) {
      const auto grid_shape_ = grid_shape(problem_size);
      cutlass::MatrixCoord threadblock_offset(
        int(tile_idx / grid_shape_.column()) * Mma::Shape::kM,
        int(tile_idx % grid_shape_.column()) * Mma::Shape::kN);

      const bool isNextTile = ((tile_idx + 1) < problem_chunk_end);
      const bool doesRowChange =
        ((threadblock_offset.column() + Mma::Shape::kN) >= problem_size.n());
      const bool do_gmem_reduce = (doesRowChange || !isNextTile) ? true : false;

      ElementA* ptr_A = static_cast<ElementA*>(params.ptr_A);
      ElementB* ptr_B = static_cast<ElementB*>(params.ptr_B);

      // Compute initial location in logical coordinates
      cutlass::MatrixCoord tb_offset_A{threadblock_offset.row(), 0};
      cutlass::MatrixCoord tb_offset_B{0, threadblock_offset.column()};

      // Compute position within threadblock
      int thread_idx = threadIdx.x;

      // Construct iterators to A and B operands
      typename Mma::IteratorA iterator_A(
        params.params_A, ptr_A, {problem_size.m(), problem_size.k()}, thread_idx, tb_offset_A);

      typename Mma::IteratorB iterator_B(
        params.params_B, ptr_B, {problem_size.k(), problem_size.n()}, thread_idx, tb_offset_B);

      // Broadcast the warp_id computed by lane 0 to ensure dependent code
      // is compiled as warp-uniform.
      int warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);

      int lane_idx = threadIdx.x % 32;

      //
      // Matrix multiply phase
      //

      // Construct thread-scoped matrix multiply
      Mma mma(shared_storage.kernel.main_loop, thread_idx, warp_idx, lane_idx);

      typename Mma::FragmentC accumulators;

      accumulators.clear();
      // Compute threadblock-scoped matrix multiply-add
      int gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;

      // Wait for all threads to finish their epilogue phases from the previous tile.
      //__syncthreads();

      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators);

      //
      // Epilogue
      //

      EpilogueOutputOp output_op(params.output_op);

      ElementC* ptr_C = static_cast<ElementC*>(params.ptr_C);
      typename Epilogue::ElementTensor* ptr_Tensor =
        static_cast<typename Epilogue::ElementTensor*>(params.ptr_Tensor);

      // Define the reduction output pointer and move to the appropriate place
      typename Epilogue::ElementVector* ptr_Vector =
        static_cast<typename Epilogue::ElementVector*>(params.ptr_Vector);

      // Tile iterator loading from source tensor.
      typename Epilogue::OutputTileIterator iterator_rownorm(shared_storage.rownorm_store,
                                                             params.params_C,
                                                             ptr_C,
                                                             problem_size.mn(),
                                                             thread_idx,
                                                             threadblock_offset);

      // Additional tensor to load from
      typename Epilogue::TensorTileIterator tensor_iterator(shared_storage.reduced_store,
                                                            params.params_Tensor,
                                                            // Only the final block outputs Tensor
                                                            ptr_Tensor,
                                                            problem_size.mn(),
                                                            thread_idx,
                                                            do_gmem_reduce,
                                                            threadblock_offset);

      Epilogue epilogue(shared_storage.kernel.epilogue, thread_idx, warp_idx, lane_idx);

      // Execute the epilogue operator to update the destination tensor.
      // Move to appropriate location for this output tile
      if (ptr_Vector) { ptr_Vector += threadblock_offset.column(); }

      // Execute the epilogue operator to update the destination tensor.
      epilogue(output_op,
               ptr_Vector,
               // iterator_D,
               accumulators,
               iterator_rownorm,
               tensor_iterator,
               problem_size.mn(),
               threadblock_offset);
    }
#endif
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
