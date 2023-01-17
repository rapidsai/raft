/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/distance/detail/pairwise_distance_base.cuh>

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the KL Divergence distance matrix:
 *  It computes the following equation:
    Cij = 0.5 * sum(x * log (x / y));
 * This distance computation modifies A or B by computing a log(x)
 * and then performing a `pow(e, log(x))` to convert it back. Because of this,
 * it is possible that the values in A or B might differ slightly
 * after this is invoked.
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
static void klDivergenceImpl(const DataT* x,
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
  auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) {
    if (isRowMajor) {
      const bool x_zero = (x == 0);
      acc += x * (raft::log(x + x_zero) - y);
    } else {
      const bool y_zero = (y == 0);
      acc += y * (raft::log(y + y_zero) - x);
    }
  };

  auto core_lambda_x_equal_y = [] __device__(AccT & acc, DataT & x, DataT & y) {
    if (isRowMajor) {
      const bool x_zero = (x == 0);
      const bool y_zero = (y == 0);
      acc += x * (raft::log(x + x_zero) - (!y_zero) * raft::log(y + y_zero));
    } else {
      const bool y_zero = (y == 0);
      const bool x_zero = (x == 0);
      acc += y * (raft::log(y + y_zero) - (!x_zero) * raft::log(x + x_zero));
    }
  };

  auto unaryOp_lambda = [] __device__(DataT input) {
    const bool x_zero = (input == 0);
    return (!x_zero) * raft::log(input + x_zero);
  };

  auto unaryOp_lambda_reverse = [] __device__(DataT input) {
    // reverse previous log (x) back to x using (e ^ log(x))
    const bool x_zero = (input == 0);
    return (!x_zero) * raft::exp(input);
  };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [] __device__(AccT acc[KPolicy::AccRowsPerTh][KPolicy::AccColsPerTh],
                                     DataT * regxn,
                                     DataT * regyn,
                                     IdxT gridStrideX,
                                     IdxT gridStrideY) {
#pragma unroll
    for (int i = 0; i < KPolicy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < KPolicy::AccColsPerTh; ++j) {
        acc[i][j] = (0.5f * acc[i][j]);
      }
    }
  };

  if (isRowMajor) {
    constexpr auto klDivergenceRowMajor = pairwiseDistanceMatKernel<false,
                                                                    DataT,
                                                                    AccT,
                                                                    OutT,
                                                                    IdxT,
                                                                    KPolicy,
                                                                    decltype(core_lambda),
                                                                    decltype(epilog_lambda),
                                                                    FinalLambda,
                                                                    true>;
    constexpr auto klDivergenceRowMajorXequalY =
      pairwiseDistanceMatKernel<false,
                                DataT,
                                AccT,
                                OutT,
                                IdxT,
                                KPolicy,
                                decltype(core_lambda_x_equal_y),
                                decltype(epilog_lambda),
                                FinalLambda,
                                true>;
    if (x != y) {
      raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda), IdxT>(
        (DataT*)y, y, n * k, unaryOp_lambda, stream);
      dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, klDivergenceRowMajor);
      klDivergenceRowMajor<<<grid, blk, KPolicy::SmemSize, stream>>>(x,
                                                                     y,
                                                                     nullptr,
                                                                     nullptr,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     lda,
                                                                     ldb,
                                                                     ldd,
                                                                     dOutput,
                                                                     core_lambda,
                                                                     epilog_lambda,
                                                                     fin_op);
      // Now reverse previous log (x) back to x using (e ^ log(x))
      raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda_reverse), IdxT>(
        (DataT*)y, y, n * k, unaryOp_lambda_reverse, stream);
    } else {
      dim3 grid =
        launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, klDivergenceRowMajorXequalY);
      klDivergenceRowMajorXequalY<<<grid, blk, KPolicy::SmemSize, stream>>>(x,
                                                                            y,
                                                                            nullptr,
                                                                            nullptr,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            lda,
                                                                            ldb,
                                                                            ldd,
                                                                            dOutput,
                                                                            core_lambda_x_equal_y,
                                                                            epilog_lambda,
                                                                            fin_op);
    }
  } else {
    constexpr auto klDivergenceColMajor = pairwiseDistanceMatKernel<false,
                                                                    DataT,
                                                                    AccT,
                                                                    OutT,
                                                                    IdxT,
                                                                    KPolicy,
                                                                    decltype(core_lambda),
                                                                    decltype(epilog_lambda),
                                                                    FinalLambda,
                                                                    false>;
    constexpr auto klDivergenceColMajorXequalY =
      pairwiseDistanceMatKernel<false,
                                DataT,
                                AccT,
                                OutT,
                                IdxT,
                                KPolicy,
                                decltype(core_lambda_x_equal_y),
                                decltype(epilog_lambda),
                                FinalLambda,
                                false>;
    if (x != y) {
      raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda), IdxT>(
        (DataT*)x, x, m * k, unaryOp_lambda, stream);
      dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, klDivergenceColMajor);
      klDivergenceColMajor<<<grid, blk, KPolicy::SmemSize, stream>>>(x,
                                                                     y,
                                                                     nullptr,
                                                                     nullptr,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     lda,
                                                                     ldb,
                                                                     ldd,
                                                                     dOutput,
                                                                     core_lambda,
                                                                     epilog_lambda,
                                                                     fin_op);
      // Now reverse previous log (x) back to x using (e ^ log(x))
      raft::linalg::unaryOp<DataT, decltype(unaryOp_lambda_reverse), IdxT>(
        (DataT*)x, x, m * k, unaryOp_lambda_reverse, stream);
    } else {
      dim3 grid =
        launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, klDivergenceColMajorXequalY);
      klDivergenceColMajorXequalY<<<grid, blk, KPolicy::SmemSize, stream>>>(x,
                                                                            y,
                                                                            nullptr,
                                                                            nullptr,
                                                                            m,
                                                                            n,
                                                                            k,
                                                                            lda,
                                                                            ldb,
                                                                            ldd,
                                                                            dOutput,
                                                                            core_lambda_x_equal_y,
                                                                            epilog_lambda,
                                                                            fin_op);
    }
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename FinalLambda,
          bool isRowMajor>
void klDivergence(IdxT m,
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
    klDivergenceImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    klDivergenceImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else {
    klDivergenceImpl<DataT, AccT, OutT, IdxT, 1, FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  }
}

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
template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_ = int>
void klDivergenceImpl(int m,
                      int n,
                      int k,
                      const InType* pA,
                      const InType* pB,
                      OutType* pD,
                      FinalLambda fin_op,
                      cudaStream_t stream,
                      bool isRowMajor)
{
  typedef std::is_same<OutType, bool> is_bool;
  typedef typename std::conditional<is_bool::value, OutType, AccType>::type klDivergenceOutType;
  Index_ lda, ldb, ldd;
  klDivergenceOutType* pDcast = reinterpret_cast<klDivergenceOutType*>(pD);
  if (isRowMajor) {
    lda = k, ldb = k, ldd = n;
    klDivergence<InType, AccType, klDivergenceOutType, Index_, FinalLambda, true>(
      m, n, k, lda, ldb, ldd, pA, pB, pDcast, fin_op, stream);

  } else {
    lda = n, ldb = m, ldd = m;
    klDivergence<InType, AccType, klDivergenceOutType, Index_, FinalLambda, false>(
      n, m, k, lda, ldb, ldd, pB, pA, pDcast, fin_op, stream);
  }
}
}  // namespace detail
}  // namespace distance
}  // namespace raft
