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

/* This file has two responsibilities:
 *
 * 1. Dispatch to the correct implementation of a kernel based on the
 *    architecture of the device on which the kernel will be launched. For
 *    instance, the cosine distance has a CUTLASS-based implementation that can
 *    be used on SM80+ and the normal implementation that is used on older
 *    architectures.
 *
 * 2. Provide concise function templates that can be instantiated in
 *    src/distance/distance/specializations/detail/. Previously,
 *    raft::distance::detail::distance was instantiated. The function
 *    necessarily required a large set of include files, which slowed down the
 *    build. The raft::distance::detail::pairwise_matrix_arch_dispatch functions
 *    do not require as large an include files set, which speeds up the build.
 */

#include <raft/distance/detail/distance_ops/cutlass.cuh>           // ops::has_cutlass_op
#include <raft/distance/detail/pairwise_matrix/dispatch_sm60.cuh>  // dispatch_sm60
#include <raft/distance/detail/pairwise_matrix/params.cuh>         // pairwise_matrix_params

// NOTE: to minimize compile times, we do not include dispatch_sm80.cuh.
// Including dispatch_sm80.cuh can slow down compile times (due to CUTLASS).
// Therefore, it is the including file's responsibility to include the correct
// dispatch_smXX.cuh headers, as is done in raft/distance/detail/distance.cuh
// and the specializations in src/distance/distance/specializations/detail/.

namespace raft::distance::detail {

// This forward-declaration ensures that we do not need to include
// dispatch_sm80.cuh if we are not calling it in practice. This makes compiling
// all the non-CUTLASS based distance specializations faster. For CUTLASS-based
// distances, dispatch_sm80.cuh has to be included by the file including this
// file.
template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_sm80_dispatch(OpT,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>,
                                   SM_compat_t,
                                   cudaStream_t);

template <typename OpT, typename IdxT, typename DataT, typename OutT, typename FinOpT>
void pairwise_matrix_instantiation_point(OpT distance_op,
                                         pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                         cudaStream_t stream)
{
  // On CUDA 12:
  // - always execute normal kernel
  //
  // On CUDA 11 and below:
  // - execute CUTLASS-based kernel on SM_80 and above
  // - execute normal kernel below SM_80

  constexpr bool is_ctk_12              = __CUDACC_VER_MAJOR__ == 12;
  constexpr bool cutlass_op_unavailable = !ops::has_cutlass_op<OpT>();

  if constexpr (is_ctk_12 || cutlass_op_unavailable) {
    // Always execute legacy kernels on CUDA 12
    auto any_range = raft::arch::SM_range(raft::arch::SM_min(), raft::arch::SM_future());
    pairwise_matrix_sm60_dispatch(distance_op, params, any_range, stream);
  } else {
    auto cutlass_range = raft::arch::SM_range(raft::arch::SM_80(), raft::arch::SM_future());
    auto legacy_range  = raft::arch::SM_range(raft::arch::SM_min(), raft::arch::SM_80());

    // Get pointer to SM60 kernel to determine the runtime architecture of the
    // current system. Other methods to determine the architecture (that do not
    // require a pointer) can be error prone. See:
    // https://github.com/NVIDIA/cub/issues/545
    auto sm60_wrapper = pairwise_matrix_sm60_get_wrapper(distance_op, params, legacy_range);
    void* kernel_ptr  = reinterpret_cast<void*>(sm60_wrapper.kernel_ptr);
    auto runtime_arch = raft::arch::kernel_runtime_arch(kernel_ptr);

    if (cutlass_range.contains(runtime_arch)) {
      // If device is SM_80 or later, use CUTLASS-based kernel.
      pairwise_matrix_sm80_dispatch(distance_op, params, cutlass_range, stream);
    } else {
      // Reuse kernel wrapper that we obtained above. This avoids performing the
      // dispatch twice.
      sm60_wrapper.launch(distance_op, params, stream);
    }
  }
}

template <typename OpT,
          typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void pairwise_matrix_dispatch(OpT distance_op,
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
  // Create kernel parameter struct. Flip x and y if column major.
  IdxT ldx = is_row_major ? k : m;
  IdxT ldy = is_row_major ? k : n;
  IdxT ld_out = is_row_major ? n : m;

  pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params{
    m, n, k, ldx, ldy, ld_out, x, y, x_norm, y_norm, out, fin_op, is_row_major};

  if (!params.is_row_major) {
    params = params.flip_x_and_y();
  }
  pairwise_matrix_instantiation_point(distance_op, params, stream);
}

};  // namespace raft::distance::detail
