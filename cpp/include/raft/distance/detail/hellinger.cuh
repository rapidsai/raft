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
#include <raft/core/operators.cuh>
#include <raft/linalg/unary_op.cuh>

#include "pairwise_matrix/dispatch.cuh"
#include "distance_ops/hellinger.cuh"

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the Hellinger distance matrix calculation
 *  It computes the following equation:
    sqrt(1 - sum(sqrt(x_k * y_k))
 * This distance computation modifies A and B by computing a sqrt
 * and then performing a `pow(x, 2)` to convert it back. Because of this,
 * it is possible that the values in A and B might differ slightly
 * after this is invoked.
 *
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
void hellingerImpl(int m,
                   int n,
                   int k,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   FinOpT fin_op,
                   cudaStream_t stream,
                   bool is_row_major)
{
  // First sqrt x and y
  const auto raft_sqrt = raft::linalg::unaryOp<DataT, raft::sqrt_op, IdxT>;

  raft_sqrt((DataT*)x, x, m * k, raft::sqrt_op{}, stream);
  if (x != y) {
    raft_sqrt((DataT*)y, y, n * k, raft::sqrt_op{}, stream);
  }

  // Then calculate Hellinger distance
  ops::hellinger_distance_op distance_op{};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);

  // Finally revert sqrt of x and y
  raft_sqrt((DataT*)x, x, m * k, raft::sqrt_op{}, stream);
  if (x != y) {
    raft_sqrt((DataT*)y, y, n * k, raft::sqrt_op{}, stream);
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}
}  // namespace detail
}  // namespace distance
}  // namespace raft
