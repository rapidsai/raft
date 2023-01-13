/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <raft/util/cuda_utils.cuh>
#include <raft/linalg/unary_op.cuh>

#include "distance_ops/kl_divergence.cuh"
#include "pairwise_matrix/dispatch.cuh"

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the KL Divergence distance matrix calculation
 *  It computes the following equation:
      Cij = 0.5 * sum(x * log (x / y));
 * This distance computation modifies A or B by computing a log(x)
 * and then performing a `pow(e, log(x))` to convert it back. Because of this,
 * it is possible that the values in A or B might differ slightly
 * after this is invoked.
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ Index type
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param fin_op the final element-wise epilogue lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void klDivergenceImpl(int m,
                      int n,
                      int k,
                      const DataT* x,
                      const DataT* y,
                      OutT* out,
                      FinOpT fin_op,
                      cudaStream_t stream,
                      bool is_row_major)
{
  auto unaryOp_lambda = [] __device__(DataT input) {
  const bool x_zero = (input == 0);
  return (!x_zero) * raft::myLog(input + x_zero);  };

  auto unaryOp_lambda_reverse = [] __device__(DataT input) {
  // reverse previous log (x) back to x using (e ^ log(x))
  const bool x_zero = (input == 0);
  return (!x_zero) * raft::myExp(input);  };

  // This op takes some shortcuts when x equals y. So it behavior changes based
  // on this.
  ops::kl_divergence_op kl_divergence{is_row_major, x == y};

  if (x != y) {
    raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda), IdxT>(
      (DataT*)y, y, n * k, unaryOp_lambda, stream);
  }

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  distance_matrix_dispatch<decltype(kl_divergence), DataT, AccT, OutT, FinOpT, IdxT>(
    kl_divergence, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);

  if (x != y) {
    // Now reverse previous log (x) back to x using (e ^ log(x))
    raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda_reverse), IdxT>(
      (DataT*)y, y, n * k, unaryOp_lambda_reverse, stream);
  }
}
}  // namespace detail
}  // namespace distance
}  // namespace raft
