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

//
/*! \file
  \brief Functor performing distance operations used by epilogues of pairwise distance
  * kernels.
* This is adapted from LinearCombinationBiasElementwise from CUTLASS 2.9.0
* customized for applying elementwise distance formula on accumulated GEMM value
* and applying user-defined operation which can convert distance values to key-value pair.
* .
*/

#pragma once

#include <cuda/semaphore>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/functional.h>
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
          typename CGReduceOp_,
          typename ReduceOpT_,
          typename KVPReduceOpT_>
class FusedDistanceNNEpilogueElementwise {
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
  using CGReduceOp = CGReduceOp_;

  using FragmentAccumulator = Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute     = Array<ElementCompute, kElementsPerAccess>;
  using FragmentC           = Array<ElementOutput, kElementsPerAccess>;
  using FragmentZ           = Array<ElementZ, kElementsPerAccess>;
  using OutValT             = typename CGReduceOp::AccTypeT;
  using FragmentT           = Array<OutValT, kElementsPerAccess>;

  using FragmentOutput = FragmentZ;

  static bool const kIsHeavy = true;  // ElementwiseOp::kIsHeavy;

  /// If true, the 'Z' tensor is stored
  static bool const kStoreZ = false;  // We don't store anything in Z,

  /// If true, the 'T' tensor is stored
  static bool const kStoreT = true;  // this is our final output storage.

  /// Host-constructable parameters structure
  struct Params {
    CGReduceOp_ cg_reduce_op;
    DistanceOp_ dist_op_;
    KVPReduceOpT_ pair_redop_;
    ReduceOpT_ red_op_;
    int* mutexes_;
    cuda::binary_semaphore<cuda::thread_scope_device>* bin_mutex_;
    using CGReduceT = CGReduceOp_;
    //
    // Methods
    //
    CUTLASS_HOST_DEVICE
    Params(DistanceOp_ dist_op,
           CGReduceOp cg_reduce_op,
           ReduceOpT_ red_op,
           KVPReduceOpT_ pair_redop,
           int* mutexes,
           cuda::binary_semaphore<cuda::thread_scope_device>* bin_mutex)
      : cg_reduce_op(cg_reduce_op),
        dist_op_(dist_op),
        pair_redop_(pair_redop),
        red_op_(red_op),
        mutexes_(mutexes),
        bin_mutex_(bin_mutex)
    {
    }

    CUTLASS_HOST_DEVICE
    Params() {}
  };

 private:
  //
  // Data members
  //
  DistanceOp_ elementwise_op;
  KVPReduceOpT_ pair_redop;

 public:
  ReduceOpT_ red_op;

  //
  // Methods
  //

  /// Constructor from Params
  CUTLASS_HOST_DEVICE
  FusedDistanceNNEpilogueElementwise(Params const& params)
    : elementwise_op(params.dist_op_), pair_redop(params.pair_redop_), red_op(params.red_op_)
  {
  }

  /// Returns true if source is needed
  CUTLASS_HOST_DEVICE
  bool is_source_needed() const
  {
    // we use for making sure C matrix is used for A mat norm.
    return true;
  }

  /// Functionally required for serial reduction in the epilogue
  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {}

  /// Applies the operation when is_source_needed() is true
  CUTLASS_HOST_DEVICE
  void operator()(FragmentT& frag_T,
                  FragmentAccumulator const& AB,
                  FragmentC const& frag_C,
                  FragmentCompute const& V) const
  {
    FragmentCompute tmp_Accum =
      NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(AB);
    FragmentCompute tmp_C =
      NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(frag_C);
    FragmentCompute result_Z;

    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute res_Z = elementwise_op(tmp_C[i], V[i], tmp_Accum[i]);
      frag_T[i]            = res_Z;
    }
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
