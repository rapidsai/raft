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

#include <cstdio>
#include <utility>
#include <raft/linalg/contractions.cuh>
#include <raft/distance/detail/pairwise_distance_cutlass_base.cuh>
#include "kernel_sm60.cuh"

namespace raft::distance::detail {

template <typename DataT>
struct params_dispatch {
  int vectorized_load_num_elem = 1;
  bool row_major               = true;

  template <int vl, bool rm>
  struct params_constexpr {
    static constexpr int vec_len = vl;
    static constexpr bool is_row_major = rm;
  };

  // Turn run-time parameters into compile-time parameters.
  // Call the provided function f with these compile-time parameters.
  // Returns false if dispatch fails, i.e., if there is no implementation
  // for the given runtime parameters.
  template <typename F>
  bool dispatch_with_compile_time_params(F&& f) const
  {
    return convert_vectorized_load_num_elem(f);
  }

  // Step 1: convert alignment into a compile time constant
  template <typename F>
  bool convert_vectorized_load_num_elem(F&& f) const
  {
    bool fail = false;
    switch (vectorized_load_num_elem) {
      case 1: return layout<1>(f);
      case 2: return layout<2>(f);
      case 4: return layout<4>(f);
      default: return fail;
    };
  }

  // Step 2: convert layout into a compile time constant
  template <int vec_len, typename F>
  bool layout(F&& f) const
  {
    if (row_major) {
      return to_compile_time_params<vec_len, true>(f);
    } else {
      return to_compile_time_params<vec_len, false>(f);
    }
  }

  // Step 3: convert compile-time constant into compile-time parameter struct and invoke
  // function f with these compile time parameters.
  template <int vec_len, bool is_row_major, typename F>
  bool to_compile_time_params(F&& f) const
  {
    // Create compile-time parameter type and instantiate a struct;
    using ct_params_T = params_constexpr<vec_len, is_row_major>;
    ct_params_T compile_time_params{};

    // Dispatch to f
    f(compile_time_params);

    bool dispatch_success = true;
    return dispatch_success;
  }
};

// Determine the largest number of elements that can be loaded in one
// instruction without causing misalignment errors.
template <typename DataT, typename IdxT>
int vectorized_load_num_elem(const DataT* x, const DataT* y, IdxT ldx, IdxT ldy)
{
  auto base_x     = reinterpret_cast<uintptr_t>(x);
  auto base_y     = reinterpret_cast<uintptr_t>(y);
  size_t stride_X = sizeof(DataT) * ldx;  // stride in bytes
  size_t stride_Y = sizeof(DataT) * ldy;  // stride in bytes

  bool base_16B_aligned = base_x % 16 == 0 && base_y % 16 == 0;
  bool base_8B_aligned  = base_x % 8 == 0 && base_y % 8 == 0;

  bool stride_16B_aligned = stride_X % 16 == 0 && stride_Y % 16 == 0;
  bool stride_8B_aligned  = stride_X % 8 == 0 && stride_Y % 8 == 0;

  if (16 % sizeof(DataT) == 0 && base_16B_aligned && stride_16B_aligned) {
    return 16 / sizeof(DataT);
  } else if (8 % sizeof(DataT) == 0 && base_8B_aligned && stride_8B_aligned) {
    return 8 / sizeof(DataT);
  } else {
    return 1;
  }
}

template <typename opT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_matrix_dispatch(opT distance_op,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              const DataT* x,
                              const DataT* y,
                              const DataT* x_norm,
                              const DataT* y_norm,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major)
{
  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  IdxT ldx, ldy, ld_out;
  if (is_row_major) {
    ldx = k, ldy = k, ld_out = n;
  } else {
    // Flip x, y, and m, n.
    std::swap<const DataT*>(x, y);
    std::swap<const DataT*>(x_norm, y_norm);
    std::swap(m, n);
    ldx = m, ldy = n, ld_out = n;
  }

  // Create run-time parameter struct that does the dispatching.
  //
  // In addition to the template parameters of this function (IdxT, DataT,
  // etc..), we explicitly dispatch based on:
  params_dispatch<DataT> run_time_params{
    vectorized_load_num_elem(x, y, ldx, ldy),   // 1. num array elements per load instruction
    is_row_major                                // 2. the layout of x, y, and out
  };

  // Turn run-time parameters into compile-time parameters.
  bool dispatch_success = run_time_params.dispatch_with_compile_time_params(
    // We pass a lambda that receives the compile-time parameters and can use these
    // to call the correct kernel.
    [&](auto p) {
      // p has two constexpr members:
      // - vec_len
      // - is_row_major

      // There is no instruction to load 4 doubles, so we catch this situation
      // and load 2 doubles.
      constexpr bool load_4_doubles = sizeof(DataT) > 4 && p.vec_len == 4;
      constexpr int vec_len = (load_4_doubles) ? 2 : p.vec_len;

      // Determine kernel policy using vec_len and layout
      typedef typename raft::linalg::Policy4x4<DataT, vec_len>::Policy RowPolicy;
      typedef typename raft::linalg::Policy4x4<DataT, vec_len>::ColPolicy ColPolicy;
      typedef typename std::conditional<p.is_row_major, RowPolicy, ColPolicy>::type Policy;

      // Create compile-time template parameter
      using KP_T = kernel_params_T<DataT, AccT, OutT, IdxT, Policy, opT, FinOpT, p.is_row_major>;

      return pairwise_matrix<KP_T>(
        distance_op,
        fin_op,
        x,
        y,
        x_norm,
        y_norm,
        m,
        n,
        k,
        ldx,
        ldy,
        ld_out,
        out,
        stream);
    });

  if (!dispatch_success) {
    std::printf("Dispatch error(!)\n");
    // TODO
  }
}

template <typename opT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void distance_matrix_cutlass_dispatch(opT cutlass_op,
                                      IdxT m,
                                      IdxT n,
                                      IdxT k,
                                      const DataT* x,
                                      const DataT* y,
                                      const DataT* x_norm,
                                      const DataT* y_norm,
                                      OutT* out,
                                      FinOpT fin_op,
                                      cudaStream_t stream,
                                      bool is_row_major)
{
  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  IdxT ldx, ldy, ld_out;
  if (is_row_major) {
    ldx = k, ldy = k, ld_out = n;
  } else {
    std::swap<const DataT*>(x, y);
    std::swap<const DataT*>(x_norm, y_norm);
    std::swap(m, n);
    ldx = m, ldy = n, ld_out = n;
  }

  params_dispatch<DataT> run_time_params{
    vectorized_load_num_elem(x, y, ldx, ldy),
    is_row_major
  };

  bool dispatch_success = run_time_params.dispatch_with_compile_time_params(
    [&](auto p) {
      // Prevent loading 4 doubles in one instruction.
      constexpr bool load_4_doubles = sizeof(DataT) > 4 && p.vec_len == 4;
      constexpr int vec_len = (load_4_doubles) ? 2 : p.vec_len;

      cutlassDistanceKernel<DataT, AccT, OutT, IdxT, vec_len, FinOpT, opT, p.is_row_major>(
        x, y, x_norm, y_norm, m, n, k, ldx, ldy, ld_out, out, fin_op, cutlass_op, stream);
    });

  if (!dispatch_success) {
    std::printf("Dispatch error(!)\n");
    // TODO
  }
}

};  // namespace raft::distance::detail
