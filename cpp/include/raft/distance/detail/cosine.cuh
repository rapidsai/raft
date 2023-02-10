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
#include <raft/linalg/norm.cuh>

#include "pairwise_matrix/dispatch.cuh"
#include "distance_ops/cosine.cuh"

namespace raft {
namespace distance {
namespace detail {

/**
 * @brief the expanded cosine distance matrix calculation
 *  It computes the following equation:
 *              C = 1 - op(A * B / sqrt(A^2) * sqrt(B^2)))
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ Index type
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @tparam in_params user-defined input parameter
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename DataT,
          typename AccT,
          typename OutT,
          typename FinOpT,
          typename IdxT = int>
void cosineAlgo1(IdxT m,
                    IdxT n,
                    IdxT k,
                    const DataT* pA,
                    const DataT* pB,
                    OutT* pD,
                    AccT* workspace,
                    size_t worksize,
                    FinOpT fin_op,
                    cudaStream_t stream,
                    bool isRowMajor)
{
  // raft distance support inputs as float/double and output as uint8_t/float/double.
  static_assert(!((sizeof(OutT) > 1) && (sizeof(AccT) != sizeof(OutT))),
                "OutT can be uint8_t, float, double,"
                "if sizeof(OutT) > 1 then sizeof(AccT) == sizeof(OutT).");

  ASSERT(
    !(((pA != pB) && (worksize < (m + n) * sizeof(AccT))) || (worksize < m * sizeof(AccT))),
    "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");


  DataT* norm_A = workspace;
  DataT* norm_B = workspace;
  if (pA != pB) {
    norm_B += m;
    raft::linalg::rowNorm(
      norm_A, pA, k, m, raft::linalg::L2Norm, isRowMajor, stream, raft::sqrt_op{});
    raft::linalg::rowNorm(
      norm_B, pB, k, n, raft::linalg::L2Norm, isRowMajor, stream, raft::sqrt_op{});
  } else {
    raft::linalg::rowNorm(
      norm_A, pA, k, m, raft::linalg::L2Norm, isRowMajor, stream, raft::sqrt_op{});
  }

  // On CUDA 12:
  // - always execute normal kernel
  //
  // On CUDA 11 and below:
  // - execute CUTLASS-based kernel on SM_80 and above
  // - execute normal kernel otherwise.

  if constexpr (__CUDACC_VER_MAJOR__ == 12) {
    // Always execute legacy kernels on CUDA 12
    ops::cosine_distance_op distance_op{};
    distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
      distance_op, m, n, k, pA, pB, norm_A, norm_B, pD, fin_op, stream, isRowMajor);
  } else {
    const auto deviceVersion = getComputeCapability();
    if (deviceVersion.first >= 8) {
      // If device is SM_80 or later, use CUTLASS-based kernel.
      using Op = ops::cosine_cutlass_op<DataT, AccT>;
      Op distance_op{};

      distance_matrix_cutlass_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
        distance_op, m, n, k, pA, pB, norm_A, norm_B, pD, fin_op, stream, isRowMajor);
    } else {
      // Else use "legacy" L2
      ops::cosine_distance_op distance_op{};
      distance_matrix_dispatch<decltype(distance_op), DataT, AccT, OutT, FinOpT, IdxT>(
        distance_op, m, n, k, pA, pB, norm_A, norm_B, pD, fin_op, stream, isRowMajor);
    }
  }
}

};  // end namespace detail
};  // end namespace distance
};  // end namespace raft
