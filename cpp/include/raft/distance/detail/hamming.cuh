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
 * @brief the Hamming distance matrix using the unexpanded form:
 *  It computes the following equation:
    Cij = sum(x_i != y_i) / k
 *
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam Veclen         number of k-elements loaded by each thread
                          for every LDG call. details in contractions.cuh
 * @tparam FinalLambda    final lambda called on final distance value
 * @tparam isRowMajor     true if input/output is row major,
                          false for column major
 * @param[in]       x input matrix
 * @param[in]       y input matrix
 * @param[in]       m number of rows of A and C/D
 * @param[in]       n number of rows of B and C/D
 * @param[in]       k number of cols of A and B
 * @param[in]       lda leading dimension of A
 * @param[in]       ldb leading dimension of B
 * @param[in]       ldd leading dimension of C/D
 * @param[output]   dOutput output matrix
 * @param[in]       fin_op the final gemm epilogue lambda
 * @param[in]       stream cuda stream to launch work
 */
template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename FinalLambda,
          bool isRowMajor>
static void hammingUnexpandedImpl(const DataT* x,
                                  const DataT* y,
                                  IdxT m,
                                  IdxT n,
                                  IdxT k,
                                  IdxT lda,
                                  IdxT ldb,
                                  IdxT ldd,
                                  OutT* dOutput,
                                  FinalLambda fin_op,
                                  cudaStream_t stream)
{
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy RowPolicy;
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::ColPolicy ColPolicy;

  typedef typename std::conditional<isRowMajor, RowPolicy, ColPolicy>::type KPolicy;

  dim3 blk(KPolicy::Nthreads);

  // Accumulation operation lambda
  auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) { acc += (x != y); };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [k] __device__(AccT acc[KPolicy::AccRowsPerTh][KPolicy::AccColsPerTh],
                                      DataT * regxn,
                                      DataT * regyn,
                                      IdxT gridStrideX,
                                      IdxT gridStrideY) {
    const DataT one_over_k = DataT(1.0) / k;
#pragma unroll
    for (int i = 0; i < KPolicy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < KPolicy::AccColsPerTh; ++j) {
        acc[i][j] *= one_over_k;
      }
    }
  };

  if (isRowMajor) {
    auto hammingUnexpandedRowMajor = pairwiseDistanceMatKernel<false,
                                                               DataT,
                                                               AccT,
                                                               OutT,
                                                               IdxT,
                                                               KPolicy,
                                                               decltype(core_lambda),
                                                               decltype(epilog_lambda),
                                                               FinalLambda,
                                                               true>;
    dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, hammingUnexpandedRowMajor);

    hammingUnexpandedRowMajor<<<grid, blk, KPolicy::SmemSize, stream>>>(
      x, y, nullptr, nullptr, m, n, k, lda, ldb, ldd, dOutput, core_lambda, epilog_lambda, fin_op);
  } else {
    auto hammingUnexpandedColMajor = pairwiseDistanceMatKernel<false,
                                                               DataT,
                                                               AccT,
                                                               OutT,
                                                               IdxT,
                                                               KPolicy,
                                                               decltype(core_lambda),
                                                               decltype(epilog_lambda),
                                                               FinalLambda,
                                                               false>;
    dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, hammingUnexpandedColMajor);
    hammingUnexpandedColMajor<<<grid, blk, KPolicy::SmemSize, stream>>>(
      x, y, nullptr, nullptr, m, n, k, lda, ldb, ldd, dOutput, core_lambda, epilog_lambda, fin_op);
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename FinalLambda,
          bool isRowMajor>
void hammingUnexpanded(IdxT m,
                       IdxT n,
                       IdxT k,
                       IdxT lda,
                       IdxT ldb,
                       IdxT ldd,
                       const DataT* x,
                       const DataT* y,
                       OutT* dOutput,
                       FinalLambda fin_op,
                       cudaStream_t stream)
{
  size_t bytesA = sizeof(DataT) * lda;
  size_t bytesB = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytesA % 16 == 0 && bytesB % 16 == 0) {
    hammingUnexpandedImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    hammingUnexpandedImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else {
    hammingUnexpandedImpl<DataT, AccT, OutT, IdxT, 1, FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  }
}

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
