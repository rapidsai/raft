/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <raft/linalg/cublas_wrappers.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/linalg/binary_op.cuh>

namespace raft {
namespace linalg {

/**
 * @brief Rank 1 update of Cholesky decomposition.
 *
 * This method is useful if an algorithm iteratively builds up matrix A, and
 * the Cholesky decomposition of A is required at each step.
 *
 * On entry, L is the Cholesky decomposition of matrix A, where both A and L
 * have size n-1 x n-1. We are interested in the Cholesky decomposition of a new
 * matrix A', which we get by adding a row and column to A. In Python notation:
 * - A'[0:n-1, 0:n-1] = A;
 * - A'[:,n-1] = A[n-1,:] = A_new
 *
 * On entry, the new column A_new, is stored as the n-th column of L if uplo ==
 * CUBLAS_FILL_MODE_UPPER, else A_new is stored as the n-th row of L.
 *
 * On exit L contains the Cholesky decomposition of A'. In practice the elements
 * of A_new are overwritten with new row/column of the L matrix.
 *
 * The uplo paramater is used to select the matrix layout.
 * If (uplo != CUBLAS_FILL_MODE_UPPER) then the input arg L stores the
 * lower triangular matrix L, so that A = L * L.T. Otherwise the input arg L
 * stores an upper triangular matrix U: A = U.T * U.
 *
 * On exit L will be updated to store the Cholesky decomposition of A'.
 *
 * If the matrix is not positive definit, or very ill conditioned then the new
 * diagonal element of L would be NaN. In such a case an exception is thrown.
 * The eps argument can be used to override this behavior: if eps >= 0 then
 * the diagonal element is replaced by eps in case the diagonal is NaN or
 * smaller than eps. Note: for an iterative solver it is probably better to
 * stop early in case of error, rather than relying on the eps parameter.
 *
 * Examples:
 *
 * - Lower triangular factorization:
 * @code{.cpp}
 * // Initialize arrays
 * int ld_L = n_rows;
 * rmm::device_uvector<math_t> L(ld_L * n_rows, stream);
 * MLCommon::LinAlg::choleskyRank1Update(handle, L, n_rows, ld_L, nullptr,
 *                                       &n_bytes, CUBLAS_FILL_MODE_LOWER,
 *                                       stream);
 * rmm::device_uvector<char> workspace(n_bytes, stream);
 *
 * for (n=1; n<=n_rows; rank++) {
 *   // Calculate a new row/column of matrix A into A_new
 *   // ...
 *   // Copy new row to L[rank-1,:]
 *   RAFT_CUBLAS_TRY(cublasCopy(handle.get_cublas_handle(), n - 1, A_new, 1,
 *                           L + n - 1, ld_L, stream));
 *   // Update Cholesky factorization
 *   MLCommon::LinAlg::choleskyRank1Update(
 *       handle, L, rank, ld_L, workspace, &n_bytes, CUBLAS_FILL_MODE_LOWER,
 *       stream);
 * }
 * Now L stores the Cholesky decomposition of A: A = L * L.T
 * @endcode
 *
 * - Upper triangular factorization:
 * @code{.cpp}
 * // Initialize arrays
 * int ld_U = n_rows;
 * rmm::device_uvector<math_t> U(ld_U * n_rows, stream);
 * MLCommon::LinAlg::choleskyRank1Update(handle, L, n_rows, ld_U, nullptr,
 *                                       &n_bytes, CUBLAS_FILL_MODE_UPPER,
 *                                       stream);
 * rmm::device_uvector<char> workspace(stream, n_bytes, stream);
 *
 * for (n=1; n<=n_rows; n++) {
 *   // Calculate a new row/column of matrix A into array A_new
 *   // ...
 *   // Copy new row to U[:,n-1] (column major layout)
 *   raft::copy(U + ld_U * (n-1), A_new, n-1, stream);
 *   //
 *   // Update Cholesky factorization
 *   MLCommon::LinAlg::choleskyRank1Update(
 *       handle, U, n, ld_U, workspace, &n_bytes, CUBLAS_FILL_MODE_UPPER,
 *       stream);
 * }
 * // Now U stores the Cholesky decomposition of A: A = U.T * U
 * @endcode
 *
 * @param handle RAFT handle (used to retrive cuBLAS handles).
 * @param L device array for to store the triangular matrix L, and the new
 *     column of A in column major format, size [n*n]
 * @param n number of elements in the new row.
 * @param ld stride of colums in L
 * @param workspace device pointer to workspace shall be nullptr ar an array
 *    of size [n_bytes].
 * @param n_bytes size of workspace is returned here if workspace==nullptr.
 * @param stream CUDA stream
 * @param uplo indicates whether L is stored as an upper or lower triangular
 *    matrix (CUBLAS_FILL_MODE_UPPER or CUBLAS_FILL_MODE_LOWER)
 * @param eps numerical parameter that can act as a regularizer for ill
 *    conditioned systems. Negative values mean no regularizaton.
 */
template <typename math_t>
void choleskyRank1Update(const raft::handle_t& handle,
                         math_t* L,
                         int n,
                         int ld,
                         void* workspace,
                         int* n_bytes,
                         cublasFillMode_t uplo,
                         cudaStream_t stream,
                         math_t eps = -1)
{
  // The matrix A' is defined as:
  // A' = [[A_11, A_12]
  //       [A_21, A_22]]
  // where:
  // - A_11 = A, matrix of size (n-1)x(n-1)
  // - A_21[j] = A_12.T[j] = A_new[j] j=0..n-2, vector with (n-1) elements
  // - A_22 = A_new[n-1] scalar.
  //
  // Instead of caclulating the Cholelsky decomposition of A' from scratch,
  // we just update L with the new row. The new Cholesky decomposition will be
  // calculated as:
  // L' = [[L_11,    0]
  //       [L_12, L_22]]
  // where L_11 is the Cholesky decomposition of A (size [n-1 x n-1]), and
  // L_12 and L_22 are the new quantities that we need to calculate.

  // We need a workspace in device memory to store a scalar. Additionally, in
  // CUBLAS_FILL_MODE_LOWER we need space for n-1 floats.
  const int align = 256;
  int offset =
    (uplo == CUBLAS_FILL_MODE_LOWER) ? raft::alignTo<int>(sizeof(math_t) * (n - 1), align) : 0;
  if (workspace == nullptr) {
    *n_bytes = offset + 1 * sizeof(math_t);
    return;
  }
  math_t* s    = reinterpret_cast<math_t*>(((char*)workspace) + offset);
  math_t* L_22 = L + (n - 1) * ld + n - 1;

  math_t* A_new;
  math_t* A_row;
  if (uplo == CUBLAS_FILL_MODE_UPPER) {
    // A_new is stored as the n-1 th column of L
    A_new = L + (n - 1) * ld;
  } else {
    // If the input is lower triangular, then the new elements of A are stored
    // as the n-th row of L. Since the matrix is column major, this is non
    // contiguous. We copy elements from A_row to a contiguous workspace A_new.
    A_row = L + n - 1;
    A_new = reinterpret_cast<math_t*>(workspace);
    RAFT_CUBLAS_TRY(
      raft::linalg::cublasCopy(handle.get_cublas_handle(), n - 1, A_row, ld, A_new, 1, stream));
  }
  cublasOperation_t op = (uplo == CUBLAS_FILL_MODE_UPPER) ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (n > 1) {
    // Calculate L_12 = x by solving equation L_11 x = A_12
    math_t alpha = 1;
    RAFT_CUBLAS_TRY(raft::linalg::cublastrsm(handle.get_cublas_handle(),
                                             CUBLAS_SIDE_LEFT,
                                             uplo,
                                             op,
                                             CUBLAS_DIAG_NON_UNIT,
                                             n - 1,
                                             1,
                                             &alpha,
                                             L,
                                             ld,
                                             A_new,
                                             n - 1,
                                             stream));

    // A_new now stores L_12, we calculate s = L_12 * L_12
    RAFT_CUBLAS_TRY(
      raft::linalg::cublasdot(handle.get_cublas_handle(), n - 1, A_new, 1, A_new, 1, s, stream));

    if (uplo == CUBLAS_FILL_MODE_LOWER) {
      // Copy back the L_12 elements as the n-th row of L
      RAFT_CUBLAS_TRY(
        raft::linalg::cublasCopy(handle.get_cublas_handle(), n - 1, A_new, 1, A_row, ld, stream));
    }
  } else {  // n == 1 case
    RAFT_CUDA_TRY(cudaMemsetAsync(s, 0, sizeof(math_t), stream));
  }

  // L_22 = sqrt(A_22 - L_12 * L_12)
  math_t s_host;
  math_t L_22_host;
  raft::update_host(&s_host, s, 1, stream);
  raft::update_host(&L_22_host, L_22, 1, stream);  // L_22 stores A_22
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  L_22_host = std::sqrt(L_22_host - s_host);

  // Check for numeric error with sqrt. If the matrix is not positive definit or
  // the system is very ill conditioned then the A_22 - L_12 * L_12 can be
  // negative, which would result L_22 = NaN. A small positive eps parameter
  // can be used to prevent this.
  if (eps >= 0 && (std::isnan(L_22_host) || L_22_host < eps)) { L_22_host = eps; }
  ASSERT(!std::isnan(L_22_host), "Error during Cholesky rank one update");
  raft::update_device(L_22, &L_22_host, 1, stream);
}
};  // namespace linalg
};  // namespace raft
