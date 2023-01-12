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
#include <raft/distance/detail/pairwise_distance_base.cuh>
#include <raft/distance/detail/pairwise_distance_op.cuh>
#include <raft/distance/detail/distance_operators.cuh>

namespace raft {
namespace distance {
namespace detail {


template <typename PCT>
static void distance_matrix_launch(
    typename PCT::opT distance_op,
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
  dim3 grid       = launchConfigGenerator<Policy>(m, n, smem_size, pairwiseDistanceOpKernel<PCT>);

  pairwiseDistanceOpKernel<PCT><<<grid, blk, smem_size, stream>>>(
    x, y, _xn, _yn, m, n, k, lda, ldb, ldd, dOutput, distance_op, fin_op);

  RAFT_CUDA_TRY(cudaGetLastError());

}

/**
 * @brief the L1 distance matrix calculation implementer
 *  It computes the following equation: cij = op(ai-bj)
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type

 * @tparam FinalLambda    final lambda called on final distance value
 * @tparam isRowMajor     true if input/output is row major,
                          false for column major
 * @param[in]       x input matrix
 * @param[in]       y input matrix
 * @param[in]       m number of rows of A and C/D
 * @param[in]       n number of columns of B and C/D
 * @param[in]       k number of cols of A and rows of B
 * @param[in]       lda leading dimension of A
 * @param[in]       ldb leading dimension of B
 * @param[in]       ldd leading dimension of C/D
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
static void l1Impl(const DataT* x,
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

  l1_distance_op distance_op{};

  using PCT = params_CT<DataT, AccT, OutT, IdxT, KPolicy, raft::distance::detail::l1_distance_op, FinalLambda, isRowMajor>;

  distance_matrix_launch<PCT>(
    distance_op, fin_op,        // Operations
    x, y, nullptr, nullptr,     // Input data
    m, n, k, lda, ldb, ldd,     // Dimensions
    dOutput,                    // Output data
    stream);                    // CUDA stream
}

template <typename DataT,
          typename AccT,
          typename OutT,
          typename IdxT,
          typename FinalLambda,
          bool isRowMajor>
void l1(IdxT m,
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
    l1Impl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    l1Impl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  } else {
    l1Impl<DataT, AccT, OutT, IdxT, 1, FinalLambda, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, dOutput, fin_op, stream);
  }
}

/**
 * @brief the L1 distance matrix calculation
 *  It computes the following equation: cij = op(ai-bj)
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
void l1Impl(int m,
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
  typedef typename std::conditional<is_bool::value, OutType, AccType>::type L1OutType;
  Index_ lda, ldb, ldd;
  L1OutType* pDcast = reinterpret_cast<L1OutType*>(pD);
  if (isRowMajor) {
    lda = k, ldb = k, ldd = n;
    l1<InType, AccType, L1OutType, Index_, FinalLambda, true>(
      m, n, k, lda, ldb, ldd, pA, pB, pDcast, fin_op, stream);

  } else {
    lda = n, ldb = m, ldd = m;
    l1<InType, AccType, L1OutType, Index_, FinalLambda, false>(
      n, m, k, lda, ldb, ldd, pB, pA, pDcast, fin_op, stream);
  }
}
}  // namespace detail
}  // namespace distance
}  // namespace raft
