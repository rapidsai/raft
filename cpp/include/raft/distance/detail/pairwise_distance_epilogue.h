/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

*/

#pragma once

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include <cutlass/gemm/gemm.h>

#include <cutlass/epilogue/threadblock/default_epilogue_tensor_op.h>
#include <cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h>
#include <cutlass/epilogue/threadblock/epilogue.h>
#include <cutlass/epilogue/threadblock/epilogue_with_broadcast.h>
#include "./predicated_tile_iterator_normvec.h"

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
struct PairwiseDistanceEpilogue {
  /// Use defaults related to the existing epilogue
  using Base =
    DefaultEpilogueTensorOp<Shape, WarpMmaTensorOp, PartitionsK, OutputOp, ElementsPerAccess>;

  //
  // Stores the result z = (y = GEMM(A, B, C), broadcast)
  //
  using OutputTileIterator = cutlass::epilogue::threadblock::
    PredicatedTileIteratorNormVec<typename Base::OutputTileThreadMap, ElementOutput, LayoutT>;

  //
  // Additional tensor tile iterator - stores t = Elementwise(z)
  //
  using TensorTileIterator =
    cutlass::epilogue::threadblock::PredicatedTileIterator<typename Base::OutputTileThreadMap,
                                                           ElementTensor>;

  /// Define the epilogue
  using Epilogue = EpilogueWithBroadcast<Shape,
                                         WarpMmaTensorOp,
                                         PartitionsK,
                                         OutputTileIterator,
                                         TensorTileIterator,
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
