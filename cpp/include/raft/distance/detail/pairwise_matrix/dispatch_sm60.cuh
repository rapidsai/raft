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

#include <raft/distance/detail/pairwise_matrix/dispatch_layout.cuh>  // dispatch_layout
#include <raft/distance/detail/pairwise_matrix/kernel_sm60.cuh>      // pairwise_matrix_sm60_wrapper
#include <raft/linalg/contractions.cuh>                              // raft::linalg::Policy4x4

#include <algorithm>  // std::min

namespace raft::distance::detail {

template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
pairwise_matrix_sm60_wrapper<OpT, IdxT, DataT, OutT, FinOpT> pairwise_matrix_sm60_get_wrapper(
  OpT distance_op,
  pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
  SM_compat_t sm_compat_range)
{
  int vec_len = determine_vec_len(params);

  // f takes compile-time constants row_major and vec_len aligned and returns
  // the corresponding kernel wrapper. The wrapper contains the launch
  // parameters of the kernel: a pointer to the kernel function, grid size,
  // block size, and shared memory size.
  auto f = [&](auto row_major, auto vec_len_aligned) {
    // row_major and vec_len are std::integral_constants of type bool and int
    // respectively.

    // To keep compile times in check, we only specialize on veclen > 1 when
    // the inner loop is relatively cheap (< 5 flops).
    constexpr int vec_len_op = distance_op.expensive_inner_loop ? 1 : vec_len_aligned();

    // Prevent double, vec_len=4 combination (this is not supported)
    constexpr int vec_len = std::min(vec_len_op, static_cast<int>(16 / sizeof(DataT)));

    using RowPolicy = typename raft::linalg::Policy4x4<DataT, vec_len>::Policy;
    using ColPolicy = typename raft::linalg::Policy4x4<DataT, vec_len>::ColPolicy;
    using Policy    = typename std::conditional<row_major(), RowPolicy, ColPolicy>::type;

    auto wrapper =
      make_pairwise_matrix_sm60_wrapper<Policy, row_major()>(distance_op, params, sm_compat_range);

    return wrapper;
  };

  // Dispatch_layout calls f with appropriate compile time constants based on
  // the runtime values of params.is_row_major and vec_len.
  return dispatch_layout(params.is_row_major, vec_len, f);
}

template <typename OpT,
          typename IdxT,
          typename DataT,
          typename OutT,
          typename FinOpT,
          typename SM_compat_t>
void pairwise_matrix_sm60_dispatch(OpT distance_op,
                                   pairwise_matrix_params<IdxT, DataT, OutT, FinOpT> params,
                                   SM_compat_t sm_compat_range,
                                   cudaStream_t stream)
{
  auto wrapper = pairwise_matrix_sm60_get_wrapper(distance_op, params, sm_compat_range);

  wrapper.launch(distance_op, params, stream);
}

}  // namespace raft::distance::detail
