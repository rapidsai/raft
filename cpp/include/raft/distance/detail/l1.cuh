/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
#include <raft/distance/detail/distance_operators.cuh>
#include <raft/distance/detail/pairwise_distance_base.cuh>
#include <raft/distance/detail/pairwise_distance_op.cuh>

namespace raft {
namespace distance {
namespace detail {

template <typename PCT>
static void distance_matrix_launch(typename PCT::opT distance_op,
                                   typename PCT::FinOpT fin_op,
                                   const typename PCT::DataT* x,
                                   const typename PCT::DataT* y,
                                   const typename PCT::DataT* _xn,
                                   const typename PCT::DataT* _yn,
                                   typename PCT::IdxT m,
                                   typename PCT::IdxT n,
                                   typename PCT::IdxT k,
                                   typename PCT::IdxT lda,
                                   typename PCT::IdxT ldb,
                                   typename PCT::IdxT ldd,
                                   typename PCT::OutT* dOutput,
                                   cudaStream_t stream)
{
  using Policy = typename PCT::PolicyT;

  dim3 blk(Policy::Nthreads);
  size_t smem_size = distance_op.template shared_mem_size<Policy>();
  dim3 grid        = launchConfigGenerator<Policy>(m, n, smem_size, pairwiseDistanceOpKernel<PCT>);

  pairwiseDistanceOpKernel<PCT><<<grid, blk, smem_size, stream>>>(
    x, y, _xn, _yn, m, n, k, lda, ldb, ldd, dOutput, distance_op, fin_op);

  RAFT_CUDA_TRY(cudaGetLastError());
}

// Determine the largest number of elements that can be loaded in one
// instruction without causing misalignment errors.
template <typename DataT>
int max_aligned_load(const DataT* x, const DataT* y, int ldx, int ldy)
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
                              int m_,
                              int n_,
                              int k_,
                              const DataT* x_,
                              const DataT* y_,
                              OutT* out,
                              FinOpT fin_op,
                              cudaStream_t stream,
                              bool is_row_major)
{
  // Determine leading dimensions and possibly flip order of passing x and y if
  // column_major.
  //
  // ldx, ldy, and ld_out are the leading dimensions of x, y, and out
  const DataT* x;
  const DataT* y;
  int ldx, ldy, ld_out;
  int m, n, k;
  if (is_row_major) {
    // Pass x, y, m, n, k in order
    x = x_,   y = y_;
    m = m_,   n = n_,   k = k_;
    ldx = k_, ldy = k_, ld_out = n_;
  } else {
    // Flip x, y, and m, n, k.
    x = y_,   y = x_;
    m = n_,   n = m_,   k = k_;
    ldx = n_, ldy = m_, ld_out = m_;
  }

  int vectorized_load_num_elem = max_aligned_load(x, y, ldx, ldy);

  // We dispatch based on
  // - vectorized_load_num_elem
  // - is_row_major

  // Create run-time parameter struct that does the dispatching
  using PRT = params_RT<DataT, AccT, OutT, IdxT, decltype(distance_op), FinOpT>;
  PRT run_time_params{vectorized_load_num_elem, is_row_major};

  // Turn run-time parameters into compile-time parameters.
  bool dispatch_success = run_time_params.dispatch_with_compile_time_params(
    // We pass a lambda that receives the compile-time parameters and can use these
    // to call the correct kernel.
    [&](auto compile_time_params) {
      // compile_time_params is an empty struct that we can convert back to a type
      // using decltype.
      return distance_matrix_launch<decltype(compile_time_params)>(
        distance_op,
        fin_op,
        x,
        y,
        nullptr,
        nullptr,  // TODO: use _xn, _yn for non-l1 distances
        m,
        n,
        k,
        ldx,
        ldy,
        ld_out,
        out,
        stream);
    });
}

template <typename DataT, typename AccT, typename OutT, typename FinOpT, typename IdxT = int>
void l1Impl(int m,
            int n,
            int k,
            const DataT* x,
            const DataT* y,
            OutT* out,
            FinOpT fin_op,
            cudaStream_t stream,
            bool is_row_major)
{
  l1_distance_op distance_op{};

  distance_matrix_dispatch<l1_distance_op, DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, out, fin_op, stream, is_row_major);
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
