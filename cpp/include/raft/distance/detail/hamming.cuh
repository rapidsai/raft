/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include "distance_ops/hamming.cuh"
#include "pairwise_matrix/dispatch.cuh"

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the Hamming Unexpanded distance matrix calculation
 *  It computes the following equation:
    Cij = sum(x_i != y_i) / k
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
void hammingUnexpandedImpl(int m,
                           int n,
                           int k,
                           const DataT* x,
                           const DataT* y,
                           OutT* out,
                           FinOpT fin_op,
                           cudaStream_t stream,
                           bool is_row_major)
{
  ops::hamming_distance_op<IdxT> distance_op{k};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
