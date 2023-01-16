/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <raft/distance/detail/pairwise_distance_cutlass_base.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace distance {
namespace detail {

template <typename DataT, typename AccT>
struct L2ExpandedOp {
  bool sqrt;

  __device__ L2ExpandedOp() noexcept : sqrt(false) {}
  __device__ L2ExpandedOp(bool isSqrt) noexcept : sqrt(isSqrt) {}
  __device__ AccT operator()(DataT& aNorm, const DataT& bNorm, DataT& accVal) const noexcept
  {
    AccT outVal = aNorm + bNorm - DataT(2.0) * accVal;
    return sqrt ? raft::sqrt(outVal) : outVal;
  }

  __device__ AccT operator()(DataT aData) const noexcept { return aData; }
};

/**
 * @brief the expanded euclidean distance matrix calculation implementer
 *  It computes the following equation: C = op(A^2 + B^2 - 2AB)
 * @tparam DataT input data-type (for A and B matrices)
 * @tparam AccT   accumulation data-type
 * @tparam OutT   output data-type (for C and D matrices)
 * @tparam IdxT   index data-type
 * @tparam Veclen number of k-elements loaded by each thread for every LDG call
 *                it makes. check contractions.cuh for details.
 * @tparam FinalLambda the final lambda called on final distance value
 * @tparam isRowMajor  true if input/output is row major,
                       false for column major
 * @param[in]     x input matrix
 * @param[in]     y input matrix
 * @param[in]     xn row norms of input matrix A.
 * @param[in]     yn row norms of input matrix B.
 * @param[in]     m number of rows of A and C/D
 * @param[in]     n number of columns of B and C/D
 * @param[in]     k number of cols of A and rows of B
 * @param[in]     lda leading dimension of A
 * @param[in]     ldb leading dimension of B
 * @param[in]     ldd leading dimension of C/D
 * @param[in]     sqrt if the square root is computed or not
 * @param[output] pD output matrix
 * @param fin_op  the final gemm epilogue lambda
*  @param stream  cuda stream to launch cuda operations.
 */
template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename FinalLambda,
          bool isRowMajor>
void euclideanExpImpl(const DataT* x,
                      const DataT* y,
                      const DataT* xn,
                      const DataT* yn,
                      IdxT m,
                      IdxT n,
                      IdxT k,
                      IdxT lda,
                      IdxT ldb,
                      IdxT ldd,
                      bool sqrt,
                      OutT* dOutput,
                      FinalLambda fin_op,
                      cudaStream_t stream)
{
#if (__CUDACC_VER_MAJOR__ < 12)
  const auto deviceVersion = getComputeCapability();
  if (deviceVersion.first >= 8) {
    using L2Op = L2ExpandedOp<DataT, AccT>;
    L2Op L2_dist_op(sqrt);

    cutlassDistanceKernel<DataT, AccT, OutT, IdxT, VecLen, FinalLambda, L2Op, isRowMajor>(
      x, y, xn, yn, m, n, k, lda, ldb, ldd, dOutput, fin_op, L2_dist_op, stream);

  } else
#endif
  {

    typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy RowPolicy;
    typedef typename raft::linalg::Policy4x4<DataT, VecLen>::ColPolicy ColPolicy;

    typedef typename std::conditional<isRowMajor, RowPolicy, ColPolicy>::type KPolicy;

    dim3 blk(KPolicy::Nthreads);

    // Accumulation operation lambda
    auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) { acc += x * y; };

    // epilogue operation lambda for final value calculation
    auto epilog_lambda = [sqrt] __device__(AccT acc[KPolicy::AccRowsPerTh][KPolicy::AccColsPerTh],
                                           DataT * regxn,
                                           DataT * regyn,
                                           IdxT gridStrideX,
                                           IdxT gridStrideY) {
#pragma unroll
      for (int i = 0; i < KPolicy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < KPolicy::AccColsPerTh; ++j) {
          acc[i][j] = regxn[i] + regyn[j] - (DataT)2.0 * acc[i][j];
        }
      }
      if (sqrt) {
#pragma unroll
        for (int i = 0; i < KPolicy::AccRowsPerTh; ++i) {
#pragma unroll
          for (int j = 0; j < KPolicy::AccColsPerTh; ++j) {
            acc[i][j] = raft::sqrt(acc[i][j]);
          }
        }
      }
    };

    constexpr size_t shmemSize =
      KPolicy::SmemSize + ((KPolicy::Mblk + KPolicy::Nblk) * sizeof(DataT));
    if (isRowMajor) {
      auto euclideanExpRowMajor = pairwiseDistanceMatKernelPriorToAmpere<true,
                                                                         DataT,
                                                                         AccT,
                                                                         OutT,
                                                                         IdxT,
                                                                         KPolicy,
                                                                         decltype(core_lambda),
                                                                         decltype(epilog_lambda),
                                                                         FinalLambda,
                                                                         true>;
      dim3 grid = launchConfigGenerator<KPolicy>(m, n, shmemSize, euclideanExpRowMajor);

      euclideanExpRowMajor<<<grid, blk, shmemSize, stream>>>(
        x, y, xn, yn, m, n, k, lda, ldb, ldd, dOutput, core_lambda, epilog_lambda, fin_op);
    } else {
      auto euclideanExpColMajor = pairwiseDistanceMatKernelPriorToAmpere<true,
                                                                         DataT,
                                                                         AccT,
                                                                         OutT,
                                                                         IdxT,
                                                                         KPolicy,
                                                                         decltype(core_lambda),
                                                                         decltype(epilog_lambda),
                                                                         FinalLambda,
                                                                         false>;
      dim3 grid = launchConfigGenerator<KPolicy>(m, n, shmemSize, euclideanExpColMajor);
      euclideanExpColMajor<<<grid, blk, shmemSize, stream>>>(
        x, y, xn, yn, m, n, k, lda, ldb, ldd, dOutput, core_lambda, epilog_lambda, fin_op);
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
void euclideanExp(IdxT m,
                  IdxT n,
                  IdxT k,
                  IdxT lda,
                  IdxT ldb,
                  IdxT ldd,
                  const DataT* x,
                  const DataT* y,
                  const DataT* xn,
                  const DataT* yn,
                  bool sqrt,
                  OutT* dOutput,
                  FinalLambda fin_op,
                  cudaStream_t stream)
{
  size_t bytesA = sizeof(DataT) * lda;
  size_t bytesB = sizeof(DataT) * ldb;

  if (16 % sizeof(DataT) == 0 && bytesA % 16 == 0 && bytesB % 16 == 0) {
    euclideanExpImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, xn, yn, m, n, k, lda, ldb, ldd, sqrt, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    euclideanExpImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, xn, yn, m, n, k, lda, ldb, ldd, sqrt, dOutput, fin_op, stream);
  } else {
    euclideanExpImpl<DataT, AccT, OutT, IdxT, 1, FinalLambda, isRowMajor>(
      x, y, xn, yn, m, n, k, lda, ldb, ldd, sqrt, dOutput, fin_op, stream);
  }
}

/**
 * @brief the expanded euclidean distance matrix calculation
 *  It computes the following equation: C = op(A^2 + B^2 - 2AB)
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam FinalLambda the final lambda called by FragmentMultiplyAdd_
 * @tparam Index_ index type
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param enable_sqrt if the square root is computed or not
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_ = int>
void euclideanAlgo1(Index_ m,
                    Index_ n,
                    Index_ k,
                    const InType* pA,
                    const InType* pB,
                    OutType* pD,
                    bool enable_sqrt,
                    AccType* workspace,
                    size_t& worksize,
                    FinalLambda fin_op,
                    cudaStream_t stream,
                    bool isRowMajor)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutType) > 1) && (sizeof(AccType) != sizeof(OutType))),
                "OutType can be uint8_t, float, double,"
                "if sizeof(OutType) > 1 then sizeof(AccType) == sizeof(OutType).");
  typedef typename std::conditional<sizeof(OutType) == 1, OutType, AccType>::type ExpOutType;
  ExpOutType* pDcast = reinterpret_cast<ExpOutType*>(pD);

  ASSERT(
    !(((pA != pB) && (worksize < (m + n) * sizeof(AccType))) || (worksize < m * sizeof(AccType))),
    "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  Index_ lda, ldb, ldd;
  InType* col_vec = workspace;
  InType* row_vec = workspace;
  if (pA != pB) {
    row_vec += m;
    raft::linalg::rowNorm(
      col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor, stream, raft::identity_op{});
    raft::linalg::rowNorm(
      row_vec, pB, k, n, raft::linalg::L2Norm, isRowMajor, stream, raft::identity_op{});
  } else {
    raft::linalg::rowNorm(
      col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor, stream, raft::identity_op{});
  }

  if (isRowMajor) {
    lda = k, ldb = k, ldd = n;
    euclideanExp<InType, AccType, ExpOutType, Index_, FinalLambda, true>(
      m, n, k, lda, ldb, ldd, pA, pB, col_vec, row_vec, enable_sqrt, pDcast, fin_op, stream);
  } else {
    lda = n, ldb = m, ldd = m;
    euclideanExp<InType, AccType, ExpOutType, Index_, FinalLambda, false>(
      n, m, k, lda, ldb, ldd, pB, pA, row_vec, col_vec, enable_sqrt, pDcast, fin_op, stream);
  }
}

/**
 * @brief the unexpanded euclidean distance matrix calculation
 *  It computes the following equation: cij = op((ai-bj)^2)
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam FinalLambda    final lambda called on final distance value
 *
 * @param[in]       x input matrix
 * @param[in]       y input matrix
 * @param[in]       m number of rows of A and C/D
 * @param[in]       n number of columns of B and C/D
 * @param[in]       k number of cols of A and rows of B
 * @param[in]       lda leading dimension of A
 * @param[in]       ldb leading dimension of B
 * @param[in]       ldd leading dimension of C/D
 * @param[in]       sqrt if the square root is computed or not
 * @param[output]   pD output matrix
 * @param fin_op    the final gemm epilogue lambda
 */
template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          int VecLen,
          typename FinalLambda,
          bool isRowMajor>
void euclideanUnExpImpl(const DataT* x,
                        const DataT* y,
                        IdxT m,
                        IdxT n,
                        IdxT k,
                        IdxT lda,
                        IdxT ldb,
                        IdxT ldd,
                        bool sqrt,
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
    const auto diff = x - y;
    acc += diff * diff;
  };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [sqrt] __device__(AccT acc[KPolicy::AccRowsPerTh][KPolicy::AccColsPerTh],
                                         DataT * regxn,
                                         DataT * regyn,
                                         IdxT gridStrideX,
                                         IdxT gridStrideY) {
    if (sqrt) {
#pragma unroll
      for (int i = 0; i < KPolicy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < KPolicy::AccColsPerTh; ++j) {
          acc[i][j] = raft::sqrt(acc[i][j]);
        }
      }
    }
  };

  if (isRowMajor) {
    auto euclideanUnExpRowMajor = pairwiseDistanceMatKernel<false,
                                                            DataT,
                                                            AccT,
                                                            OutT,
                                                            IdxT,
                                                            KPolicy,
                                                            decltype(core_lambda),
                                                            decltype(epilog_lambda),
                                                            FinalLambda,
                                                            true>;
    dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, euclideanUnExpRowMajor);

    euclideanUnExpRowMajor<<<grid, blk, KPolicy::SmemSize, stream>>>(
      x, y, nullptr, nullptr, m, n, k, lda, ldb, ldd, dOutput, core_lambda, epilog_lambda, fin_op);

  } else {
    auto euclideanUnExpColMajor = pairwiseDistanceMatKernel<false,
                                                            DataT,
                                                            AccT,
                                                            OutT,
                                                            IdxT,
                                                            KPolicy,
                                                            decltype(core_lambda),
                                                            decltype(epilog_lambda),
                                                            FinalLambda,
                                                            false>;
    dim3 grid = launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, euclideanUnExpColMajor);

    euclideanUnExpColMajor<<<grid, blk, KPolicy::SmemSize, stream>>>(
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
void euclideanUnExp(IdxT m,
                    IdxT n,
                    IdxT k,
                    IdxT lda,
                    IdxT ldb,
                    IdxT ldd,
                    const DataT* x,
                    const DataT* y,
                    bool sqrt,
                    OutT* dOutput,
                    FinalLambda fin_op,
                    cudaStream_t stream)
{
  size_t bytesA = sizeof(DataT) * lda;
  size_t bytesB = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytesA % 16 == 0 && bytesB % 16 == 0) {
    euclideanUnExpImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, sqrt, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    euclideanUnExpImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, sqrt, dOutput, fin_op, stream);
  } else {
    euclideanUnExpImpl<DataT, AccT, OutT, IdxT, 1, FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, sqrt, dOutput, fin_op, stream);
  }
}

/**
 * @brief the unexpanded euclidean distance matrix calculation
 *  It computes the following equation: cij = op((ai-bj)^2)
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ index type
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param enable_sqrt if the square root is computed or not
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType,
          typename AccType,
          typename OutType,
          typename FinalLambda,
          typename Index_ = int>
void euclideanAlgo2(Index_ m,
                    Index_ n,
                    Index_ k,
                    const InType* pA,
                    const InType* pB,
                    OutType* pD,
                    bool enable_sqrt,
                    FinalLambda fin_op,
                    cudaStream_t stream,
                    bool isRowMajor)
{
  typedef std::is_same<OutType, bool> is_bool;
  typedef typename std::conditional<is_bool::value, OutType, AccType>::type UnExpOutType;
  UnExpOutType* pDcast = reinterpret_cast<UnExpOutType*>(pD);
  Index_ lda, ldb, ldd;

  if (isRowMajor) {
    lda = k, ldb = k, ldd = n;
    euclideanUnExp<InType, AccType, UnExpOutType, Index_, FinalLambda, true>(
      m, n, k, lda, ldb, ldd, pA, pB, enable_sqrt, pDcast, fin_op, stream);
  } else {
    lda = n, ldb = m, ldd = m;
    euclideanUnExp<InType, AccType, UnExpOutType, Index_, FinalLambda, false>(
      n, m, k, lda, ldb, ldd, pB, pA, enable_sqrt, pDcast, fin_op, stream);
  }
}

};  // end namespace detail
};  // end namespace distance
};  // end namespace raft
