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

#include <raft/distance/detail/pairwise_distance_cutlass_base.cuh>   // cutlassDistanceKernel
#include <raft/distance/detail/pairwise_matrix/dispatch_layout.cuh>  // dispatch_layout

#include <algorithm>  // std::min

namespace raft::distance::detail {

template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_sm80_dispatch(OpT distance_op,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                   SM_compat_t sm_compat_range,
                                   cudaStream_t stream)
{
  int vec_len = determine_vec_len(params);

  // f takes compile-time constants row_major and vec_len aligned and runs the
  // corresponding cutlass launch code.
  auto f = [&](auto row_major, auto vec_len_aligned) {
    // row_major and vec_len are std::integral_constants of type bool and int
    // respectively.

    // Prevent double, vec_len=4 combination (this is not supported)
    constexpr int vec_len = std::min(vec_len_aligned(), static_cast<int>(16 / sizeof(DataT)));

    using AccT = typename OpT::AccT;
    cutlassDistanceKernel<DataT, AccT, OutT, IdxT, vec_len, FinOpT, OpT, row_major()>(params.x,
                                                                                      params.y,
                                                                                      params.x_norm,
                                                                                      params.y_norm,
                                                                                      params.m,
                                                                                      params.n,
                                                                                      params.k,
                                                                                      params.ldx,
                                                                                      params.ldy,
                                                                                      params.ld_out,
                                                                                      params.out,
                                                                                      params.fin_op,
                                                                                      distance_op,
                                                                                      stream);
  };

  // Dispatch_layout calls f with appropriate compile time constants based on
  // the runtime values of params.is_row_major and vec_len.
  dispatch_layout(params.is_row_major, vec_len, f);
}

};  // namespace raft::distance::detail
