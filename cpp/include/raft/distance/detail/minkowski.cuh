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
#include "pairwise_matrix/dispatch.cuh"
#include "distance_ops/minkowski.cuh"

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the unexpanded minkowski distance matrix calculation
 *  It computes the following equation: cij = sum(|x - y|^p)^(1/p)
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ index type
 * @param[in] m number of rows of A and C/D
 * @param[in] n number of rows of B and cols of C/D
 * @param[in] k number of cols of A and B
 * @param[in] pA input matrix
 * @param[in] pB input matrix
 * @param[out] pD output matrix
 * @param[in] fin_op the final gemm epilogue lambda
 * @param[in] stream cuda stream to launch work
 * @param[in] isRowMajor whether the input and output matrices are row major
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances.
 */
template <typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void minkowskiImpl(IdxT m,
                   IdxT n,
                   IdxT k,
                   const DataT* x,
                   const DataT* y,
                   OutT* out,
                   FinOpT fin_op,
                   cudaStream_t stream,
                   bool is_row_major,
                   DataT metric_arg)
{
  ops::minkowski_distance_op<DataT> distance_op{metric_arg};

  const DataT* x_norm = nullptr;
  const DataT* y_norm = nullptr;

  distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
    distance_op, m, n, k, x, y, x_norm, y_norm, out, fin_op, stream, is_row_major);
}
};  // end namespace detail
};  // end namespace distance
};  // end namespace raft
