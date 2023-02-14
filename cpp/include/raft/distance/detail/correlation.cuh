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
#include <raft/distance/detail/pairwise_distance_base.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the Correlation distance matrix:
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
static void correlationImpl(const DataT* x,
                            const DataT* y,
                            const DataT* xn,
                            const DataT* yn,
                            const DataT* x2n,
                            const DataT* y2n,
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
  auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) { acc += x * y; };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [x2n, y2n, m, n, k] __device__(
                         AccT acc[KPolicy::AccRowsPerTh][KPolicy::AccColsPerTh],
                         DataT * regxn,
                         DataT * regyn,
                         IdxT gridStrideX,
                         IdxT gridStrideY) {
    DataT regx2n[KPolicy::AccRowsPerTh], regy2n[KPolicy::AccColsPerTh];

    extern __shared__ char smem[];
    DataT* sx2Norm =
      (DataT*)(&smem[KPolicy::SmemSize + (KPolicy::Mblk + KPolicy::Nblk) * sizeof(DataT)]);
    DataT* sy2Norm = (&sx2Norm[KPolicy::Mblk]);

    // Load x & y norms required by this threadblock in shmem buffer
    if (gridStrideX == blockIdx.x * KPolicy::Nblk) {
      for (int i = threadIdx.x; i < KPolicy::Mblk; i += KPolicy::Nthreads) {
        auto idx   = gridStrideY + i;
        sx2Norm[i] = idx < m ? x2n[idx] : 0;
      }
    }

    for (int i = threadIdx.x; i < KPolicy::Nblk; i += KPolicy::Nthreads) {
      auto idx   = gridStrideX + i;
      sy2Norm[i] = idx < n ? y2n[idx] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < KPolicy::AccRowsPerTh; ++i) {
      regx2n[i] = sx2Norm[i * KPolicy::AccThRows + (threadIdx.x / KPolicy::AccThCols)];
    }
#pragma unroll
    for (int i = 0; i < KPolicy::AccColsPerTh; ++i) {
      regy2n[i] = sy2Norm[i * KPolicy::AccThCols + (threadIdx.x % KPolicy::AccThCols)];
    }

#pragma unroll
    for (int i = 0; i < KPolicy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < KPolicy::AccColsPerTh; ++j) {
        auto numer   = k * acc[i][j] - (regxn[i] * regyn[j]);
        auto Q_denom = k * regx2n[i] - (regxn[i] * regxn[i]);
        auto R_denom = k * regy2n[j] - (regyn[j] * regyn[j]);

        acc[i][j] = 1 - (numer / raft::sqrt(Q_denom * R_denom));
      }
    }
  };

  constexpr size_t shmemSize =
    KPolicy::SmemSize + (2 * (KPolicy::Mblk + KPolicy::Nblk) * sizeof(DataT));
  if (isRowMajor) {
    constexpr auto correlationRowMajor = pairwiseDistanceMatKernel<true,
                                                                   DataT,
                                                                   AccT,
                                                                   OutT,
                                                                   IdxT,
                                                                   KPolicy,
                                                                   decltype(core_lambda),
                                                                   decltype(epilog_lambda),
                                                                   FinalLambda,
                                                                   true>;
    dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, correlationRowMajor);
    correlationRowMajor<<<grid, blk, shmemSize, stream>>>(
      x, y, xn, yn, m, n, k, lda, ldb, ldd, dOutput, core_lambda, epilog_lambda, fin_op);
  } else {
    constexpr auto correlationColMajor = pairwiseDistanceMatKernel<true,
                                                                   DataT,
                                                                   AccT,
                                                                   OutT,
                                                                   IdxT,
                                                                   KPolicy,
                                                                   decltype(core_lambda),
                                                                   decltype(epilog_lambda),
                                                                   FinalLambda,
                                                                   false>;
    dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, correlationColMajor);
    correlationColMajor<<<grid, blk, shmemSize, stream>>>(
      x, y, xn, yn, m, n, k, lda, ldb, ldd, dOutput, core_lambda, epilog_lambda, fin_op);
  }

  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename FinalLambda,
          bool isRowMajor>
void correlation(IdxT m,
                 IdxT n,
                 IdxT k,
                 IdxT lda,
                 IdxT ldb,
                 IdxT ldd,
                 const DataT* x,
                 const DataT* y,
                 const DataT* xn,
                 const DataT* yn,
                 const DataT* x2n,
                 const DataT* y2n,
                 OutT* dOutput,
                 FinalLambda fin_op,
                 cudaStream_t stream)
{
  size_t bytesA = sizeof(DataT) * lda;
  size_t bytesB = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytesA % 16 == 0 && bytesB % 16 == 0) {
    correlationImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, xn, yn, x2n, y2n, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    correlationImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, xn, yn, x2n, y2n, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else {
    correlationImpl<DataT, AccT, OutT, IdxT, 1, FinalLambda, isRowMajor>(
      x, y, xn, yn, x2n, y2n, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  }
}

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
  typedef std::is_same<OutType, bool> is_bool;
  typedef typename std::conditional<is_bool::value, OutType, AccType>::type correlationOutType;
  Index_ lda, ldb, ldd;
  correlationOutType* pDcast = reinterpret_cast<correlationOutType*>(pD);

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

  if (isRowMajor) {
    lda = k, ldb = k, ldd = n;
    correlation<InType, AccType, correlationOutType, Index_, FinalLambda, true>(m,
                                                                                n,
                                                                                k,
                                                                                lda,
                                                                                ldb,
                                                                                ldd,
                                                                                pA,
                                                                                pB,
                                                                                norm_col_vec,
                                                                                norm_row_vec,
                                                                                sq_norm_col_vec,
                                                                                sq_norm_row_vec,
                                                                                pDcast,
                                                                                fin_op,
                                                                                stream);
  } else {
    lda = n, ldb = m, ldd = m;
    correlation<InType, AccType, correlationOutType, Index_, FinalLambda, false>(n,
                                                                                 m,
                                                                                 k,
                                                                                 lda,
                                                                                 ldb,
                                                                                 ldd,
                                                                                 pB,
                                                                                 pA,
                                                                                 norm_row_vec,
                                                                                 norm_col_vec,
                                                                                 sq_norm_row_vec,
                                                                                 sq_norm_col_vec,
                                                                                 pDcast,
                                                                                 fin_op,
                                                                                 stream);
  }
}

}  // namespace detail
}  // namespace distance
}  // namespace raft
