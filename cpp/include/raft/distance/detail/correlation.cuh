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

#include <raft/linalg/reduce.cuh>

#include "pairwise_matrix/dispatch.cuh"
#include "distance_ops/correlation.cuh"

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the Correlation distance matrix calculation
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
template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_ = int>
void correlationImpl(int m,
                     int n,
                     int k,
                     const InType* pA,
                     const InType* pB,
                     OutType* pD,
                     AccType* workspace,
                     size_t& worksize,
                     FinalLambda fin_op,
                     cudaStream_t stream,
                     bool isRowMajor)
{
  ASSERT(!(((pA != pB) && (worksize < 2 * (m + n) * sizeof(AccType))) ||
           (worksize < 2 * m * sizeof(AccType))),
         "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  AccType* norm_col_vec    = workspace;
  AccType* norm_row_vec    = workspace;
  AccType* sq_norm_col_vec = workspace;
  AccType* sq_norm_row_vec = workspace;
  if (pA != pB) {
    norm_row_vec += m;

    raft::linalg::reduce(norm_col_vec,
                         pA,
                         k,
                         m,
                         (AccType)0,
                         isRowMajor,
                         true,
                         stream,
                         false,
                         raft::identity_op(),
                         raft::add_op());
    raft::linalg::reduce(norm_row_vec,
                         pB,
                         k,
                         n,
                         (AccType)0,
                         isRowMajor,
                         true,
                         stream,
                         false,
                         raft::identity_op(),
                         raft::add_op());

    sq_norm_col_vec += (m + n);
    sq_norm_row_vec = sq_norm_col_vec + m;
    raft::linalg::rowNorm(sq_norm_col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor, stream);
    raft::linalg::rowNorm(sq_norm_row_vec, pB, k, n, raft::linalg::L2Norm, isRowMajor, stream);
  } else {
    raft::linalg::reduce(norm_col_vec,
                         pA,
                         k,
                         m,
                         (AccType)0,
                         isRowMajor,
                         true,
                         stream,
                         false,
                         raft::identity_op(),
                         raft::add_op());
    sq_norm_col_vec += m;
    sq_norm_row_vec = sq_norm_col_vec;
    raft::linalg::rowNorm(sq_norm_col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor, stream);
  }

  using CorrOp = ops::correlation_distance_op<InType, Index_>;
  CorrOp corr_op(isRowMajor, sq_norm_col_vec, sq_norm_row_vec, m, n, k);
  distance_matrix_dispatch<decltype(corr_op), InType, AccType, OutType, FinalLambda, Index_>(
    corr_op, m, n, k, pA, pB, norm_col_vec, norm_row_vec, pD, fin_op, stream, isRowMajor);
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
