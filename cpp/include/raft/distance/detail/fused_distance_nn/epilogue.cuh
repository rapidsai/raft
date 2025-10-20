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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

This is adapted from DefaultEpilogueWithBroadcastTensorOp from CUTLASS 2.9.0
(https://github.com/NVIDIA/cutlass/blob/master/include/cutlass/epilogue/threadblock/default_epilogue_with_broadcast.h#L75)

This epilogue allows us to load norm buffers using PredicatedTileIteratorNormVec
and EpilogueWithBroadcast used for distances L2/cosine as well as applies user-define elementwise
operation.
-- A norm load is provided PredicatedTileIteratorNormVec
-- B norm load is provided by EpilogueWithBroadcast
-- elementwise operation is provided by OutputOp
*/

#pragma once

#include <raft/distance/detail/fused_distance_nn/custom_epilogue_with_broadcast.h>
#include <raft/distance/detail/fused_distance_nn/predicated_tile_iterator_normvec_smem.h>
#include <raft/distance/detail/fused_distance_nn/predicated_tile_iterator_reduced_vec.h>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <typename Shape,
          typename WarpMmaTensorOp,
          int PartitionsK,
          typename ElementOutput,
          typename ElementTensor,
          typename ElementVector,
          typename OutputOp,
          typename LayoutT,
          int ElementsPerAccess,
          bool ScatterD = false>
struct FusedDistanceNNEpilogue {
  /// Use defaults related to the existing epilogue
  using Base =
    DefaultEpilogueTensorOp<Shape, WarpMmaTensorOp, PartitionsK, OutputOp, ElementsPerAccess>;

  //
  // Stores the result z = (y = GEMM(A, B, C), broadcast)
  //
  using RowNormTileIterator = cutlass::epilogue::threadblock::
    PredicatedTileIteratorNormVecSmem<typename Base::OutputTileThreadMap, ElementOutput, LayoutT>;

  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using OutputTileIterator = cutlass::epilogue::threadblock::PredicatedTileIteratorReducedVec<
    typename Base::OutputTileThreadMap,
    ElementTensor,
    LayoutT,
    typename OutputOp::Params>;

  /// Define the epilogue
  using Epilogue = cutlass::epilogue::threadblock::EpilogueWithBroadcastCustom<
    Shape,
    WarpMmaTensorOp,
    PartitionsK,
    RowNormTileIterator,
    OutputTileIterator,
    ElementVector,
    typename Base::AccumulatorFragmentIterator,
    typename Base::WarpTileIterator,
    typename Base::SharedLoadIterator,
    OutputOp,
    typename Base::Padding,
    Base::kFragmentsPerIteration>;
};

}  // namespace threadblock
}  // namespace epilogue
}  // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
