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

//
/*! \file
  \brief Functor performing distance operations used by epilogues of pairwise distance
  * kernels.
* This is adapted from LinearCombinationBiasElementwise from CUTLASS 2.9.0
* customized for applying elementwise distance formula on accumulated GEMM value
* and applying user-defined final custom operation on the distance value.
*/

#pragma once

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
#include <cutlass/layout/pitch_linear.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace thread {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// This base class is meant to define the concept required of the
/// EpilogueWithBroadcast::OutputOp
template <typename ElementC_,
          typename ElementAccumulator_,
          typename ElementCompute_,
          typename ElementZ_,
          typename ElementT_,
          int ElementsPerAccess,
          typename DistanceOp_,
          typename FinalOp_>
class PairwiseDistanceEpilogueElementwise {
 public:
  using ElementOutput                 = ElementC_;
  using ElementC                      = ElementC_;
  using ElementAccumulator            = ElementAccumulator_;
  using ElementCompute                = ElementCompute_;
  using ElementZ                      = ElementZ_;
  using ElementT                      = ElementT_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount             = kElementsPerAccess;

  using DistanceOp = DistanceOp_;
  using FinalOp    = FinalOp_;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute     = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC           = Array<ElementOutput, kElementsPerAccess>;
  using FragmentZ           = Array<ElementZ, kElementsPerAccess>;
  using FragmentT           = Array<ElementT, kElementsPerAccess>;

  using FragmentOutput = FragmentZ;

  static bool const kIsHeavy = false;  // ElementwiseOp::kIsHeavy;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = false;  // We don't store anything in Z,

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = true;  // this is our final output storage.

  /// Host-constructable parameters structure
  struct Params {
    FinalOp_ final_op_;
    DistanceOp_ dist_op_;

    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params(DistanceOp_ dist_op, FinalOp final_op) : final_op_(final_op), dist_op_(dist_op) {}

    CUTLASS_HOST_DEVICE
    Params() {}
  };

 private:
  //
  // Data members
  //
  FinalOp_ final_op;
  DistanceOp_ elementwise_op;

 public:
  //
  // Methods
  //

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  PairwiseDistanceEpilogueElementwise(Params const& params)
    : final_op(params.final_op_), elementwise_op(params.dist_op_)
  {
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const
  {
    // we use for making sure C matrix path is used for A mat norm.
    return true;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& frag_Z,
                  FragmentT& frag_T,
                  FragmentAccumulator const& AB,
                  FragmentC const& frag_C,
                  FragmentCompute const& V) const
  {
    FragmentCompute tmp_Accum =
      NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C =
      NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;
    FragmentCompute result_T;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      result_Z[i] = elementwise_op(tmp_C[i], V[i], tmp_Accum[i]);
      result_T[i] = final_op(result_Z[i], 0);
    }

    NumericArrayConverter<ElementT, ElementCompute, kElementsPerAccess> convert_t;
    frag_T = convert_t(result_T);
  }

  /// Applies the operation when is_source_needed() is false
  CUTLASS_HOST_DEVICE
  void operator()(FragmentZ& frag_Z,
                  FragmentT& frag_T,
                  FragmentAccumulator const& AB,
                  FragmentCompute const& V) const
  {
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
