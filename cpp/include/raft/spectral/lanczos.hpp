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

//for cmath:
#define _USE_MATH_DEFINES

#include <cmath>
#include <vector>

#include <cuda.h>
#include <curand.h>

#include <raft/linalg/cublas_wrappers.h>
#include <raft/handle.hpp>
#include <raft/spectral/error_temp.hpp>
#include <raft/spectral/matrix_wrappers.hpp>

// =========================================================
// Useful macros
// =========================================================

// Get index of matrix entry
#define IDX(i, j, lda) ((i) + (j) * (lda))

namespace raft {

namespace {

using namespace matrix;

// =========================================================
// Helper functions
// =========================================================

/// Perform Lanczos iteration
/** Lanczos iteration is performed on a shifted matrix A+shift*I.
 *
 *  @param A Matrix.
 *  @param iter Pointer to current Lanczos iteration. On exit, the
 *    variable is set equal to the final Lanczos iteration.
 *  @param maxIter Maximum Lanczos iteration. This function will
 *    perform a maximum of maxIter-*iter iterations.
 *  @param shift Matrix shift.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm (i.e. entry in beta_host) is
 *    less than tol.
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param alpha_host (Output, host memory, maxIter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Output, host memory, maxIter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param lanczosVecs_dev (Input/output, device memory,
 *    n*(maxIter+1) entries) Lanczos vectors. Vectors are stored as
 *    columns of a column-major matrix with dimensions
 *    n x (maxIter+1).
 *  @param work_dev (Output, device memory, maxIter entries)
 *    Workspace. Not needed if full reorthogonalization is disabled.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename IndexType_, typename ValueType_>
int performLanczosIteration(
  handle_t handle, sparse_matrix_t<IndexType_, ValueType_> const *A,
  IndexType_ *iter, IndexType_ maxIter, ValueType_ shift, ValueType_ tol,
  bool reorthogonalize, ValueType_ *__restrict__ alpha_host,
  ValueType_ *__restrict__ beta_host, ValueType_ *__restrict__ lanczosVecs_dev,
  ValueType_ *__restrict__ work_dev) {
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Useful variables
  const ValueType_ one = 1;
  const ValueType_ negOne = -1;
  const ValueType_ zero = 0;

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  RAFT_EXPECT(A != nullptr, "Null matrix pointer.");

  IndexType_ n = A->nrows;

  // -------------------------------------------------------
  // Compute second Lanczos vector
  // -------------------------------------------------------
  if (*iter <= 0) {
    *iter = 1;

    // Apply matrix
    if (shift != 0)
      CUDA_TRY(cudaMemcpyAsync(lanczosVecs_dev + n, lanczosVecs_dev,
                               n * sizeof(ValueType_), cudaMemcpyDeviceToDevice,
                               stream));
    A->mv(1, lanczosVecs_dev, shift, lanczosVecs_dev + n);

    // Orthogonalize Lanczos vector
    CUBLAS_CHECK(cublasdot(cublas_h, n, lanczosVecs_dev, 1,
                           lanczosVecs_dev + IDX(0, 1, n), 1, alpha_host,
                           stream));

    auto alpha = -alpha_host[0];
    CUBLAS_CHECK(cublasaxpy(cublas_h, n, &alpha, lanczosVecs_dev, 1,
                            lanczosVecs_dev + IDX(0, 1, n), 1, stream));
    CUBLAS_CHECK(cublasnrm2(cublas_h, n, lanczosVecs_dev + IDX(0, 1, n), 1,
                            beta_host, stream));

    // Check if Lanczos has converged
    if (beta_host[0] <= tol) return 0;

    // Normalize Lanczos vector
    alpha = 1 / beta_host[0];
    CUBLAS_CHECK(cublasscal(cublas_h, n, &alpha, lanczosVecs_dev + IDX(0, 1, n),
                            1, stream));
  }

  // -------------------------------------------------------
  // Compute remaining Lanczos vectors
  // -------------------------------------------------------

  while (*iter < maxIter) {
    ++(*iter);

    // Apply matrix
    if (shift != 0)
      CUDA_TRY(cudaMemcpyAsync(
        lanczosVecs_dev + (*iter) * n, lanczosVecs_dev + (*iter - 1) * n,
        n * sizeof(ValueType_), cudaMemcpyDeviceToDevice, stream));
    A->mv(1, lanczosVecs_dev + IDX(0, *iter - 1, n), shift,
          lanczosVecs_dev + IDX(0, *iter, n));

    // Full reorthogonalization
    //   "Twice is enough" algorithm per Kahan and Parlett
    if (reorthogonalize) {
      CUBLAS_CHECK(cublasgemv(
        cublas_h, CUBLAS_OP_T, n, *iter, &one, lanczosVecs_dev, n,
        lanczosVecs_dev + IDX(0, *iter, n), 1, &zero, work_dev, 1, stream));

      CUBLAS_CHECK(cublasgemv(cublas_h, CUBLAS_OP_N, n, *iter, &negOne,
                              lanczosVecs_dev, n, work_dev, 1, &one,
                              lanczosVecs_dev + IDX(0, *iter, n), 1, stream));

      CUDA_TRY(cudaMemcpyAsync(alpha_host + (*iter - 1), work_dev + (*iter - 1),
                               sizeof(ValueType_), cudaMemcpyDeviceToHost,
                               stream));

      CUBLAS_CHECK(cublasgemv(
        cublas_h, CUBLAS_OP_T, n, *iter, &one, lanczosVecs_dev, n,
        lanczosVecs_dev + IDX(0, *iter, n), 1, &zero, work_dev, 1, stream));

      CUBLAS_CHECK(cublasgemv(cublas_h, CUBLAS_OP_N, n, *iter, &negOne,
                              lanczosVecs_dev, n, work_dev, 1, &one,
                              lanczosVecs_dev + IDX(0, *iter, n), 1, stream));
    }

    // Orthogonalization with 3-term recurrence relation
    else {
      CUBLAS_CHECK(cublasdot(cublas_h, n,
                             lanczosVecs_dev + IDX(0, *iter - 1, n), 1,
                             lanczosVecs_dev + IDX(0, *iter, n), 1,
                             alpha_host + (*iter - 1), stream));

      auto alpha = -alpha_host[*iter - 1];
      CUBLAS_CHECK(cublasaxpy(cublas_h, n, &alpha,
                              lanczosVecs_dev + IDX(0, *iter - 1, n), 1,
                              lanczosVecs_dev + IDX(0, *iter, n), 1, stream));

      alpha = -beta_host[*iter - 2];
      CUBLAS_CHECK(cublasaxpy(cublas_h, n, &alpha,
                              lanczosVecs_dev + IDX(0, *iter - 2, n), 1,
                              lanczosVecs_dev + IDX(0, *iter, n), 1, stream));
    }

    // Compute residual
    CUBLAS_CHECK(cublasnrm2(cublas_h, n, lanczosVecs_dev + IDX(0, *iter, n), 1,
                            beta_host + *iter - 1, stream));

    // Check if Lanczos has converged
    if (beta_host[*iter - 1] <= tol) break;

    // Normalize Lanczos vector
    alpha = 1 / beta_host[*iter - 1];
    CUBLAS_CHECK(cublasscal(cublas_h, n, &alpha,
                            lanczosVecs_dev + IDX(0, *iter, n), 1, stream));
  }

  CUDA_TRY(cudaDeviceSynchronize());

  return 0;
}

/// Find Householder transform for 3-dimensional system
/** Given an input vector v=[x,y,z]', this function finds a
 *  Householder transform P such that P*v is a multiple of
 *  e_1=[1,0,0]'. The input vector v is overwritten with the
 *  Householder vector such that P=I-2*v*v'.
 *
 *  @param v (Input/output, host memory, 3 entries) Input
 *    3-dimensional vector. On exit, the vector is set to the
 *    Householder vector.
 *  @param Pv (Output, host memory, 1 entry) First entry of P*v
 *    (here v is the input vector). Either equal to ||v||_2 or
 *    -||v||_2.
 *  @param P (Output, host memory, 9 entries) Householder transform
 *    matrix. Matrix dimensions are 3 x 3.
 */
template <typename IndexType_, typename ValueType_>
static void findHouseholder3(ValueType_ *v, ValueType_ *Pv, ValueType_ *P) {
  // Compute norm of vector
  *Pv = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

  // Choose whether to reflect to e_1 or -e_1
  //   This choice avoids catastrophic cancellation
  if (v[0] >= 0) *Pv = -(*Pv);
  v[0] -= *Pv;

  // Normalize Householder vector
  ValueType_ normHouseholder =
    std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (normHouseholder != 0) {
    v[0] /= normHouseholder;
    v[1] /= normHouseholder;
    v[2] /= normHouseholder;
  } else {
    v[0] = 0;
    v[1] = 0;
    v[2] = 0;
  }

  // Construct Householder matrix
  IndexType_ i, j;
  for (j = 0; j < 3; ++j)
    for (i = 0; i < 3; ++i) P[IDX(i, j, 3)] = -2 * v[i] * v[j];
  for (i = 0; i < 3; ++i) P[IDX(i, i, 3)] += 1;
}

/// Apply 3-dimensional Householder transform to 4 x 4 matrix
/** The Householder transform is pre-applied to the top three rows
 *  of the matrix and post-applied to the left three columns. The
 *  4 x 4 matrix is intended to contain the bulge that is produced
 *  in the Francis QR algorithm.
 *
 *  @param v (Input, host memory, 3 entries) Householder vector.
 *  @param A (Input/output, host memory, 16 entries) 4 x 4 matrix.
 */
template <typename IndexType_, typename ValueType_>
static void applyHouseholder3(const ValueType_ *v, ValueType_ *A) {
  // Loop indices
  IndexType_ i, j;
  // Dot product between Householder vector and matrix row/column
  ValueType_ vDotA;

  // Pre-apply Householder transform
  for (j = 0; j < 4; ++j) {
    vDotA = 0;
    for (i = 0; i < 3; ++i) vDotA += v[i] * A[IDX(i, j, 4)];
    for (i = 0; i < 3; ++i) A[IDX(i, j, 4)] -= 2 * v[i] * vDotA;
  }

  // Post-apply Householder transform
  for (i = 0; i < 4; ++i) {
    vDotA = 0;
    for (j = 0; j < 3; ++j) vDotA += A[IDX(i, j, 4)] * v[j];
    for (j = 0; j < 3; ++j) A[IDX(i, j, 4)] -= 2 * vDotA * v[j];
  }
}

/// Perform one step of Francis QR algorithm
/** Equivalent to two steps of the classical QR algorithm on a
 *  tridiagonal matrix.
 *
 *  @param n Matrix dimension.
 *  @param shift1 QR algorithm shift.
 *  @param shift2 QR algorithm shift.
 *  @param alpha (Input/output, host memory, n entries) Diagonal
 *    entries of tridiagonal matrix.
 *  @param beta (Input/output, host memory, n-1 entries)
 *    Off-diagonal entries of tridiagonal matrix.
 *  @param V (Input/output, host memory, n*n entries) Orthonormal
 *    transforms from previous steps of QR algorithm. Matrix
 *    dimensions are n x n. On exit, the orthonormal transform from
 *    this Francis QR step is post-applied to the matrix.
 *  @param work (Output, host memory, 3*n entries) Workspace.
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename IndexType_, typename ValueType_>
static int francisQRIteration(IndexType_ n, ValueType_ shift1,
                              ValueType_ shift2, ValueType_ *alpha,
                              ValueType_ *beta, ValueType_ *V,
                              ValueType_ *work) {
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Temporary storage of 4x4 bulge and Householder vector
  ValueType_ bulge[16];

  // Householder vector
  ValueType_ householder[3];
  // Householder matrix
  ValueType_ householderMatrix[3 * 3];

  // Shifts are roots of the polynomial p(x)=x^2+b*x+c
  ValueType_ b = -shift1 - shift2;
  ValueType_ c = shift1 * shift2;

  // Loop indices
  IndexType_ i, j, pos;
  // Temporary variable
  ValueType_ temp;

  // -------------------------------------------------------
  // Implementation
  // -------------------------------------------------------

  // Compute initial Householder transform
  householder[0] = alpha[0] * alpha[0] + beta[0] * beta[0] + b * alpha[0] + c;
  householder[1] = beta[0] * (alpha[0] + alpha[1] + b);
  householder[2] = beta[0] * beta[1];
  findHouseholder3<IndexType_, ValueType_>(householder, &temp,
                                           householderMatrix);

  // Apply initial Householder transform to create bulge
  memset(bulge, 0, 16 * sizeof(ValueType_));
  for (i = 0; i < 4; ++i) bulge[IDX(i, i, 4)] = alpha[i];
  for (i = 0; i < 3; ++i) {
    bulge[IDX(i + 1, i, 4)] = beta[i];
    bulge[IDX(i, i + 1, 4)] = beta[i];
  }
  applyHouseholder3<IndexType_, ValueType_>(householder, bulge);
  Lapack<ValueType_>::gemm(false, false, n, 3, 3, 1, V, n, householderMatrix, 3,
                           0, work, n);
  memcpy(V, work, 3 * n * sizeof(ValueType_));

  // Chase bulge to bottom-right of matrix with Householder transforms
  for (pos = 0; pos < n - 4; ++pos) {
    // Move to next position
    alpha[pos] = bulge[IDX(0, 0, 4)];
    householder[0] = bulge[IDX(1, 0, 4)];
    householder[1] = bulge[IDX(2, 0, 4)];
    householder[2] = bulge[IDX(3, 0, 4)];
    for (j = 0; j < 3; ++j)
      for (i = 0; i < 3; ++i) bulge[IDX(i, j, 4)] = bulge[IDX(i + 1, j + 1, 4)];
    bulge[IDX(3, 0, 4)] = 0;
    bulge[IDX(3, 1, 4)] = 0;
    bulge[IDX(3, 2, 4)] = beta[pos + 3];
    bulge[IDX(0, 3, 4)] = 0;
    bulge[IDX(1, 3, 4)] = 0;
    bulge[IDX(2, 3, 4)] = beta[pos + 3];
    bulge[IDX(3, 3, 4)] = alpha[pos + 4];

    // Apply Householder transform
    findHouseholder3<IndexType_, ValueType_>(householder, beta + pos,
                                             householderMatrix);
    applyHouseholder3<IndexType_, ValueType_>(householder, bulge);
    Lapack<ValueType_>::gemm(false, false, n, 3, 3, 1, V + IDX(0, pos + 1, n),
                             n, householderMatrix, 3, 0, work, n);
    memcpy(V + IDX(0, pos + 1, n), work, 3 * n * sizeof(ValueType_));
  }

  // Apply penultimate Householder transform
  //   Values in the last row and column are zero
  alpha[n - 4] = bulge[IDX(0, 0, 4)];
  householder[0] = bulge[IDX(1, 0, 4)];
  householder[1] = bulge[IDX(2, 0, 4)];
  householder[2] = bulge[IDX(3, 0, 4)];
  for (j = 0; j < 3; ++j)
    for (i = 0; i < 3; ++i) bulge[IDX(i, j, 4)] = bulge[IDX(i + 1, j + 1, 4)];
  bulge[IDX(3, 0, 4)] = 0;
  bulge[IDX(3, 1, 4)] = 0;
  bulge[IDX(3, 2, 4)] = 0;
  bulge[IDX(0, 3, 4)] = 0;
  bulge[IDX(1, 3, 4)] = 0;
  bulge[IDX(2, 3, 4)] = 0;
  bulge[IDX(3, 3, 4)] = 0;
  findHouseholder3<IndexType_, ValueType_>(householder, beta + n - 4,
                                           householderMatrix);
  applyHouseholder3<IndexType_, ValueType_>(householder, bulge);
  Lapack<ValueType_>::gemm(false, false, n, 3, 3, 1, V + IDX(0, n - 3, n), n,
                           householderMatrix, 3, 0, work, n);
  memcpy(V + IDX(0, n - 3, n), work, 3 * n * sizeof(ValueType_));

  // Apply final Householder transform
  //   Values in the last two rows and columns are zero
  alpha[n - 3] = bulge[IDX(0, 0, 4)];
  householder[0] = bulge[IDX(1, 0, 4)];
  householder[1] = bulge[IDX(2, 0, 4)];
  householder[2] = 0;
  for (j = 0; j < 3; ++j)
    for (i = 0; i < 3; ++i) bulge[IDX(i, j, 4)] = bulge[IDX(i + 1, j + 1, 4)];
  findHouseholder3<IndexType_, ValueType_>(householder, beta + n - 3,
                                           householderMatrix);
  applyHouseholder3<IndexType_, ValueType_>(householder, bulge);
  Lapack<ValueType_>::gemm(false, false, n, 2, 2, 1, V + IDX(0, n - 2, n), n,
                           householderMatrix, 3, 0, work, n);
  memcpy(V + IDX(0, n - 2, n), work, 2 * n * sizeof(ValueType_));

  // Bulge has been eliminated
  alpha[n - 2] = bulge[IDX(0, 0, 4)];
  alpha[n - 1] = bulge[IDX(1, 1, 4)];
  beta[n - 2] = bulge[IDX(1, 0, 4)];

  return 0;
}

/// Perform implicit restart of Lanczos algorithm
/** Shifts are Chebyshev nodes of unwanted region of matrix spectrum.
 *
 *  @param n Matrix dimension.
 *  @param iter Current Lanczos iteration.
 *  @param iter_new Lanczos iteration after restart.
 *  @param shiftUpper Pointer to upper bound for unwanted
 *    region. Value is ignored if less than *shiftLower. If a
 *    stronger upper bound has been found, the value is updated on
 *    exit.
 *  @param shiftLower Pointer to lower bound for unwanted
 *    region. Value is ignored if greater than *shiftUpper. If a
 *    stronger lower bound has been found, the value is updated on
 *    exit.
 *  @param alpha_host (Input/output, host memory, iter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Input/output, host memory, iter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param V_host (Output, host memory, iter*iter entries)
 *    Orthonormal transform used to obtain restarted system. Matrix
 *    dimensions are iter x iter.
 *  @param work_host (Output, host memory, 4*iter entries)
 *    Workspace.
 *  @param lanczosVecs_dev (Input/output, device memory, n*(iter+1)
 *    entries) Lanczos vectors. Vectors are stored as columns of a
 *    column-major matrix with dimensions n x (iter+1).
 *  @param work_dev (Output, device memory, (n+iter)*iter entries)
 *    Workspace.
 */
template <typename IndexType_, typename ValueType_>
static int lanczosRestart(
  handle_t handle, IndexType_ n, IndexType_ iter, IndexType_ iter_new,
  ValueType_ *shiftUpper, ValueType_ *shiftLower,
  ValueType_ *__restrict__ alpha_host, ValueType_ *__restrict__ beta_host,
  ValueType_ *__restrict__ V_host, ValueType_ *__restrict__ work_host,
  ValueType_ *__restrict__ lanczosVecs_dev, ValueType_ *__restrict__ work_dev,
  bool smallest_eig) {
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Useful constants
  const ValueType_ zero = 0;
  const ValueType_ one = 1;

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // Loop index
  IndexType_ i;

  // Number of implicit restart steps
  //   Assumed to be even since each call to Francis algorithm is
  //   equivalent to two calls of QR algorithm
  IndexType_ restartSteps = iter - iter_new;

  // Ritz values from Lanczos method
  ValueType_ *ritzVals_host = work_host + 3 * iter;
  // Shifts for implicit restart
  ValueType_ *shifts_host;

  // Orthonormal matrix for similarity transform
  ValueType_ *V_dev = work_dev + n * iter;

  // -------------------------------------------------------
  // Implementation
  // -------------------------------------------------------

  // Compute Ritz values
  memcpy(ritzVals_host, alpha_host, iter * sizeof(ValueType_));
  memcpy(work_host, beta_host, (iter - 1) * sizeof(ValueType_));
  Lapack<ValueType_>::sterf(iter, ritzVals_host, work_host);

  // Debug: Print largest eigenvalues
  // for (int i = iter-iter_new; i < iter; ++i)
  //  std::cout <<*(ritzVals_host+i)<< " ";
  // std::cout <<std::endl;

  // Initialize similarity transform with identity matrix
  memset(V_host, 0, iter * iter * sizeof(ValueType_));
  for (i = 0; i < iter; ++i) V_host[IDX(i, i, iter)] = 1;

  // Determine interval to suppress eigenvalues
  if (smallest_eig) {
    if (*shiftLower > *shiftUpper) {
      *shiftUpper = ritzVals_host[iter - 1];
      *shiftLower = ritzVals_host[iter_new];
    } else {
      *shiftUpper = max(*shiftUpper, ritzVals_host[iter - 1]);
      *shiftLower = min(*shiftLower, ritzVals_host[iter_new]);
    }
  } else {
    if (*shiftLower > *shiftUpper) {
      *shiftUpper = ritzVals_host[iter - iter_new - 1];
      *shiftLower = ritzVals_host[0];
    } else {
      *shiftUpper = max(*shiftUpper, ritzVals_host[iter - iter_new - 1]);
      *shiftLower = min(*shiftLower, ritzVals_host[0]);
    }
  }

  // Calculate Chebyshev nodes as shifts
  shifts_host = ritzVals_host;
  for (i = 0; i < restartSteps; ++i) {
    shifts_host[i] =
      cos((i + 0.5) * static_cast<ValueType_>(M_PI) / restartSteps);
    shifts_host[i] *= 0.5 * ((*shiftUpper) - (*shiftLower));
    shifts_host[i] += 0.5 * ((*shiftUpper) + (*shiftLower));
  }

  // Apply Francis QR algorithm to implicitly restart Lanczos
  for (i = 0; i < restartSteps; i += 2)
    if (francisQRIteration(iter, shifts_host[i], shifts_host[i + 1], alpha_host,
                           beta_host, V_host, work_host))
      WARNING("error in implicitly shifted QR algorithm");

  // Obtain new residual
  CUDA_TRY(cudaMemcpyAsync(V_dev, V_host, iter * iter * sizeof(ValueType_),
                           cudaMemcpyHostToDevice, stream));

  beta_host[iter - 1] =
    beta_host[iter - 1] * V_host[IDX(iter - 1, iter_new - 1, iter)];
  CUBLAS_CHECK(cublasgemv(
    cublas_h, CUBLAS_OP_N, n, iter, beta_host + iter_new - 1, lanczosVecs_dev,
    n, V_dev + IDX(0, iter_new, iter), 1, beta_host + iter - 1,
    lanczosVecs_dev + IDX(0, iter, n), 1, stream));

  // Obtain new Lanczos vectors
  CUBLAS_CHECK(cublasgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, iter_new, iter,
                          &one, lanczosVecs_dev, n, V_dev, iter, &zero,
                          work_dev, n, stream));

  CUDA_TRY(cudaMemcpyAsync(lanczosVecs_dev, work_dev,
                           n * iter_new * sizeof(ValueType_),
                           cudaMemcpyDeviceToDevice, stream));

  // Normalize residual to obtain new Lanczos vector
  CUDA_TRY(cudaMemcpyAsync(
    lanczosVecs_dev + IDX(0, iter_new, n), lanczosVecs_dev + IDX(0, iter, n),
    n * sizeof(ValueType_), cudaMemcpyDeviceToDevice, stream));

  CUBLAS_CHECK(cublasnrm2(cublas_h, n, lanczosVecs_dev + IDX(0, iter_new, n), 1,
                          beta_host + iter_new - 1, stream));

  auto h_beta = 1 / beta_host[iter_new - 1];
  CUBLAS_CHECK(cublasscal(cublas_h, n, &h_beta,
                          lanczosVecs_dev + IDX(0, iter_new, n), 1, stream));

  return 0;
}

}  // namespace

// =========================================================
// Eigensolver
// =========================================================

/// Compute smallest eigenvectors of symmetric matrix
/** Computes eigenvalues and eigenvectors that are least
 *  positive. If matrix is positive definite or positive
 *  semidefinite, the computed eigenvalues are smallest in
 *  magnitude.
 *
 *  The largest eigenvalue is estimated by performing several
 *  Lanczos iterations. An implicitly restarted Lanczos method is
 *  then applied to A+s*I, where s is negative the largest
 *  eigenvalue.
 *
 *  @param A Matrix.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter Maximum number of Lanczos steps. Does not include
 *    Lanczos steps used to estimate largest eigenvalue.
 *  @param restartIter Maximum size of Lanczos system before
 *    performing an implicit restart. Should be at least 4.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm is less than tol*theta, where
 *    theta is an estimate for the smallest unwanted eigenvalue
 *    (i.e. the (nEigVecs+1)th smallest eigenvalue).
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param effIter On exit, pointer to final size of Lanczos system.
 *  @param totalIter On exit, pointer to total number of Lanczos
 *    iterations performed. Does not include Lanczos steps used to
 *    estimate largest eigenvalue.
 *  @param shift On exit, pointer to matrix shift (estimate for
 *    largest eigenvalue).
 *  @param alpha_host (Output, host memory, restartIter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Output, host memory, restartIter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param lanczosVecs_dev (Output, device memory, n*(restartIter+1)
 *    entries) Lanczos vectors. Vectors are stored as columns of a
 *    column-major matrix with dimensions n x (restartIter+1).
 *  @param work_dev (Output, device memory,
 *    (n+restartIter)*restartIter entries) Workspace.
 *  @param eigVals_dev (Output, device memory, nEigVecs entries)
 *    Largest eigenvalues of matrix.
 *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
 *    Eigenvectors corresponding to smallest eigenvalues of
 *    matrix. Vectors are stored as columns of a column-major matrix
 *    with dimensions n x nEigVecs.
 *  @return error flag.
 */
template <typename IndexType_, typename ValueType_>
int computeSmallestEigenvectors(
  handle_t handle, sparse_matrix_t<IndexType_, ValueType_> const *A,
  IndexType_ nEigVecs, IndexType_ maxIter, IndexType_ restartIter,
  ValueType_ tol, bool reorthogonalize, IndexType_ *effIter,
  IndexType_ *totalIter, ValueType_ *shift, ValueType_ *__restrict__ alpha_host,
  ValueType_ *__restrict__ beta_host, ValueType_ *__restrict__ lanczosVecs_dev,
  ValueType_ *__restrict__ work_dev, ValueType_ *__restrict__ eigVals_dev,
  ValueType_ *__restrict__ eigVecs_dev, unsigned long long seed) {
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Useful constants
  const ValueType_ one = 1;
  const ValueType_ zero = 0;

  // Matrix dimension
  IndexType_ n = A->nrows;

  // Shift for implicit restart
  ValueType_ shiftUpper;
  ValueType_ shiftLower;

  // Lanczos iteration counters
  IndexType_ maxIter_curr = restartIter;  // Maximum size of Lanczos system

  // Status flags
  int status;

  // Loop index
  IndexType_ i;

  // Host memory
  ValueType_ *Z_host;     // Eigenvectors in Lanczos basis
  ValueType_ *work_host;  // Workspace

  // -------------------------------------------------------
  // Check that LAPACK is enabled
  // -------------------------------------------------------
  // Lapack<ValueType_>::check_lapack_enabled();

  // -------------------------------------------------------
  // Check that parameters are valid
  // -------------------------------------------------------
  RAFT_EXPECT(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECT(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECT(tol > 0, "Invalid tolerance.");
  RAFT_EXPECT(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECT(restartIter >= nEigVecs, "Invalid restartIter.");

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // -------------------------------------------------------
  // Variable initialization
  // -------------------------------------------------------

  // Total number of Lanczos iterations
  *totalIter = 0;

  // Allocate host memory
  std::vector<ValueType_> Z_host_v(restartIter * restartIter);
  std::vector<ValueType_> work_host_v(4 * restartIter);

  Z_host = Z_host_v.data();
  work_host = work_host_v.data();

  // Initialize cuBLAS
  CUBLAS_CHECK(cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST,
                                    stream));  // ????? TODO: check / remove

  // -------------------------------------------------------
  // Compute largest eigenvalue to determine shift
  // -------------------------------------------------------

  // Random number generator
  curandGenerator_t randGen;
  // Initialize random number generator
  CUDA_TRY(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_PHILOX4_32_10));

  // FIXME: This is hard coded, which is good for unit testing...
  //        but should really be a parameter so it could be
  //        "random" for real runs and "fixed" for tests
  CUDA_TRY(curandSetPseudoRandomGeneratorSeed(randGen, seed /*time(NULL)*/));
  // CUDA_TRY(curandSetPseudoRandomGeneratorSeed(randGen, time(NULL)));
  // Initialize initial Lanczos vector
  CUDA_TRY(
    curandGenerateNormalX(randGen, lanczosVecs_dev, n + n % 2, zero, one));
  ValueType_ normQ1;
  CUBLAS_CHECK(cublasnrm2(cublas_h, n, lanczosVecs_dev, 1, &normQ1, stream));

  auto h_val = 1 / normQ1;
  CUBLAS_CHECK(cublasscal(cublas_h, n, &h_val, lanczosVecs_dev, 1, stream));

  // Estimate number of Lanczos iterations
  //   See bounds in Kuczynski and Wozniakowski (1992).
  // const ValueType_ relError = 0.25;  // Relative error
  // const ValueType_ failProb = 1e-4;  // Probability of failure
  // maxIter_curr = log(n/pow(failProb,2))/(4*std::sqrt(relError)) + 1;
  // maxIter_curr = min(maxIter_curr, restartIter);

  // Obtain tridiagonal matrix with Lanczos
  *effIter = 0;
  *shift = 0;
  status = performLanczosIteration<IndexType_, ValueType_>(
    handle, A, effIter, maxIter_curr, *shift, 0.0, reorthogonalize, alpha_host,
    beta_host, lanczosVecs_dev, work_dev);
  if (status) WARNING("error in Lanczos iteration");

  // Determine largest eigenvalue

  Lapack<ValueType_>::sterf(*effIter, alpha_host, beta_host);
  *shift = -alpha_host[*effIter - 1];
  // std::cout <<  *shift <<std::endl;
  // -------------------------------------------------------
  // Compute eigenvectors of shifted matrix
  // -------------------------------------------------------

  // Obtain tridiagonal matrix with Lanczos
  *effIter = 0;
  // maxIter_curr = min(maxIter, restartIter);
  status = performLanczosIteration<IndexType_, ValueType_>(
    handle, A, effIter, maxIter_curr, *shift, 0, reorthogonalize, alpha_host,
    beta_host, lanczosVecs_dev, work_dev);
  if (status) WARNING("error in Lanczos iteration");
  *totalIter += *effIter;

  // Apply Lanczos method until convergence
  shiftLower = 1;
  shiftUpper = -1;
  while (*totalIter < maxIter && beta_host[*effIter - 1] > tol * shiftLower) {
    // Determine number of restart steps
    // Number of steps must be even due to Francis algorithm
    IndexType_ iter_new = nEigVecs + 1;
    if (restartIter - (maxIter - *totalIter) > nEigVecs + 1)
      iter_new = restartIter - (maxIter - *totalIter);
    if ((restartIter - iter_new) % 2) iter_new -= 1;
    if (iter_new == *effIter) break;

    // Implicit restart of Lanczos method
    status = lanczosRestart<IndexType_, ValueType_>(
      handle, n, *effIter, iter_new, &shiftUpper, &shiftLower, alpha_host,
      beta_host, Z_host, work_host, lanczosVecs_dev, work_dev, true);
    if (status) WARNING("error in Lanczos implicit restart");
    *effIter = iter_new;

    // Check for convergence
    if (beta_host[*effIter - 1] <= tol * fabs(shiftLower)) break;

    // Proceed with Lanczos method
    // maxIter_curr = min(restartIter, maxIter-*totalIter+*effIter);
    status = performLanczosIteration<IndexType_, ValueType_>(
      handle, A, effIter, maxIter_curr, *shift, tol * fabs(shiftLower),
      reorthogonalize, alpha_host, beta_host, lanczosVecs_dev, work_dev);
    if (status) WARNING("error in Lanczos iteration");
    *totalIter += *effIter - iter_new;
  }

  // Warning if Lanczos has failed to converge
  if (beta_host[*effIter - 1] > tol * fabs(shiftLower)) {
    WARNING("implicitly restarted Lanczos failed to converge");
  }

  // Solve tridiagonal system
  memcpy(work_host + 2 * (*effIter), alpha_host,
         (*effIter) * sizeof(ValueType_));
  memcpy(work_host + 3 * (*effIter), beta_host,
         (*effIter - 1) * sizeof(ValueType_));
  Lapack<ValueType_>::steqr('I', *effIter, work_host + 2 * (*effIter),
                            work_host + 3 * (*effIter), Z_host, *effIter,
                            work_host);

  // Obtain desired eigenvalues by applying shift
  for (i = 0; i < *effIter; ++i) work_host[i + 2 * (*effIter)] -= *shift;
  for (i = *effIter; i < nEigVecs; ++i) work_host[i + 2 * (*effIter)] = 0;

  // Copy results to device memory
  CUDA_TRY(cudaMemcpy(eigVals_dev, work_host + 2 * (*effIter),
                      nEigVecs * sizeof(ValueType_), cudaMemcpyHostToDevice));
  // for (int i = 0; i < nEigVecs; ++i)
  //{
  //  std::cout <<*(work_host+(2*(*effIter)+i))<< std::endl;
  //}
  CUDA_TRY(cudaMemcpy(work_dev, Z_host,
                      (*effIter) * nEigVecs * sizeof(ValueType_),
                      cudaMemcpyHostToDevice));

  // Convert eigenvectors from Lanczos basis to standard basis
  CUBLAS_CHECK(cublasgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, nEigVecs,
                          *effIter, &one, lanczosVecs_dev, n, work_dev,
                          *effIter, &zero, eigVecs_dev, n, stream));

  // Clean up and exit
  CUDA_TRY(curandDestroyGenerator(randGen));
  return 0;
}

/// Compute smallest eigenvectors of symmetric matrix
/** Computes eigenvalues and eigenvectors that are least
 *  positive. If matrix is positive definite or positive
 *  semidefinite, the computed eigenvalues are smallest in
 *  magnitude.
 *
 *  The largest eigenvalue is estimated by performing several
 *  Lanczos iterations. An implicitly restarted Lanczos method is
 *  then applied to A+s*I, where s is negative the largest
 *  eigenvalue.
 *
 *  CNMEM must be initialized before calling this function.
 *
 *  @param A Matrix.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter Maximum number of Lanczos steps. Does not include
 *    Lanczos steps used to estimate largest eigenvalue.
 *  @param restartIter Maximum size of Lanczos system before
 *    performing an implicit restart. Should be at least 4.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm is less than tol*theta, where
 *    theta is an estimate for the smallest unwanted eigenvalue
 *    (i.e. the (nEigVecs+1)th smallest eigenvalue).
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param iter On exit, pointer to total number of Lanczos
 *    iterations performed. Does not include Lanczos steps used to
 *    estimate largest eigenvalue.
 *  @param eigVals_dev (Output, device memory, nEigVecs entries)
 *    Smallest eigenvalues of matrix.
 *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
 *    Eigenvectors corresponding to smallest eigenvalues of
 *    matrix. Vectors are stored as columns of a column-major matrix
 *    with dimensions n x nEigVecs.
 *  @return error flag.
 */
template <typename IndexType_, typename ValueType_>
int computeSmallestEigenvectors(
  handle_t handle, sparse_matrix_t<IndexType_, ValueType_> const &A,
  IndexType_ nEigVecs, IndexType_ maxIter, IndexType_ restartIter,
  ValueType_ tol, bool reorthogonalize, IndexType_ &iter,
  ValueType_ *__restrict__ eigVals_dev, ValueType_ *__restrict__ eigVecs_dev,
  unsigned long long seed = 1234567) {
  // Matrix dimension
  IndexType_ n = A.nrows;

  // Check that parameters are valid
  RAFT_EXPECT(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECT(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECT(tol > 0, "Invalid tolerance.");
  RAFT_EXPECT(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECT(restartIter >= nEigVecs, "Invalid restartIter.");

  // Allocate memory
  std::vector<ValueType_> alpha_host_v(restartIter);
  std::vector<ValueType_> beta_host_v(restartIter);

  ValueType_ *alpha_host = alpha_host_v.data();
  ValueType_ *beta_host = beta_host_v.data();

  //TODO: replace and fix allocation via RAFT handle
  vector_t<ValueType_> lanczosVecs_dev(handle, n * (restartIter + 1));
  vector_t<ValueType_> work_dev(handle, (n + restartIter) * restartIter);

  // Perform Lanczos method
  IndexType_ effIter;
  ValueType_ shift;
  int status = computeSmallestEigenvectors(
    handle, &A, nEigVecs, maxIter, restartIter, tol, reorthogonalize, &effIter,
    &iter, &shift, alpha_host, beta_host, lanczosVecs_dev.raw(), work_dev.raw(),
    eigVals_dev, eigVecs_dev, seed);

  // Clean up and return
  return status;
}

// =========================================================
// Eigensolver
// =========================================================

/// Compute largest eigenvectors of symmetric matrix
/** Computes eigenvalues and eigenvectors that are least
 *  positive. If matrix is positive definite or positive
 *  semidefinite, the computed eigenvalues are largest in
 *  magnitude.
 *
 *  The largest eigenvalue is estimated by performing several
 *  Lanczos iterations. An implicitly restarted Lanczos method is
 *  then applied.
 *
 *  @param A Matrix.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter Maximum number of Lanczos steps.
 *  @param restartIter Maximum size of Lanczos system before
 *    performing an implicit restart. Should be at least 4.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm is less than tol*theta, where
 *    theta is an estimate for the largest unwanted eigenvalue
 *    (i.e. the (nEigVecs+1)th largest eigenvalue).
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param effIter On exit, pointer to final size of Lanczos system.
 *  @param totalIter On exit, pointer to total number of Lanczos
 *    iterations performed.
 *  @param alpha_host (Output, host memory, restartIter entries)
 *    Diagonal entries of Lanczos system.
 *  @param beta_host (Output, host memory, restartIter entries)
 *    Off-diagonal entries of Lanczos system.
 *  @param lanczosVecs_dev (Output, device memory, n*(restartIter+1)
 *    entries) Lanczos vectors. Vectors are stored as columns of a
 *    column-major matrix with dimensions n x (restartIter+1).
 *  @param work_dev (Output, device memory,
 *    (n+restartIter)*restartIter entries) Workspace.
 *  @param eigVals_dev (Output, device memory, nEigVecs entries)
 *    Largest eigenvalues of matrix.
 *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
 *    Eigenvectors corresponding to largest eigenvalues of
 *    matrix. Vectors are stored as columns of a column-major matrix
 *    with dimensions n x nEigVecs.
 *  @return error flag.
 */
template <typename IndexType_, typename ValueType_>
int computeLargestEigenvectors(
  handle_t handle, sparse_matrix_t<IndexType_, ValueType_> const *A,
  IndexType_ nEigVecs, IndexType_ maxIter, IndexType_ restartIter,
  ValueType_ tol, bool reorthogonalize, IndexType_ *effIter,
  IndexType_ *totalIter, ValueType_ *__restrict__ alpha_host,
  ValueType_ *__restrict__ beta_host, ValueType_ *__restrict__ lanczosVecs_dev,
  ValueType_ *__restrict__ work_dev, ValueType_ *__restrict__ eigVals_dev,
  ValueType_ *__restrict__ eigVecs_dev, unsigned long long seed) {
  // -------------------------------------------------------
  // Variable declaration
  // -------------------------------------------------------

  // Useful constants
  const ValueType_ one = 1;
  const ValueType_ zero = 0;

  // Matrix dimension
  IndexType_ n = A->nrows;

  // Lanczos iteration counters
  IndexType_ maxIter_curr = restartIter;  // Maximum size of Lanczos system

  // Status flags
  int status;

  // Loop index
  IndexType_ i;

  // Host memory
  ValueType_ *Z_host;     // Eigenvectors in Lanczos basis
  ValueType_ *work_host;  // Workspace

  // -------------------------------------------------------
  // Check that LAPACK is enabled
  // -------------------------------------------------------
  // Lapack<ValueType_>::check_lapack_enabled();

  // -------------------------------------------------------
  // Check that parameters are valid
  // -------------------------------------------------------
  RAFT_EXPECT(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECT(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECT(tol > 0, "Invalid tolerance.");
  RAFT_EXPECT(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECT(restartIter >= nEigVecs, "Invalid restartIter.");

  auto cublas_h = handle.get_cublas_handle();
  auto stream = handle.get_stream();

  // -------------------------------------------------------
  // Variable initialization
  // -------------------------------------------------------

  // Total number of Lanczos iterations
  *totalIter = 0;

  // Allocate host memory
  std::vector<ValueType_> Z_host_v(restartIter * restartIter);
  std::vector<ValueType_> work_host_v(4 * restartIter);

  Z_host = Z_host_v.data();
  work_host = work_host_v.data();

  // Initialize cuBLAS
  CUBLAS_CHECK(cublassetpointermode(cublas_h, CUBLAS_POINTER_MODE_HOST,
                                    stream));  // ????? TODO: check / remove

  // -------------------------------------------------------
  // Compute largest eigenvalue
  // -------------------------------------------------------

  // Random number generator
  curandGenerator_t randGen;
  // Initialize random number generator
  CUDA_TRY(curandCreateGenerator(&randGen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
  CUDA_TRY(curandSetPseudoRandomGeneratorSeed(randGen, seed));
  // Initialize initial Lanczos vector
  CUDA_TRY(
    curandGenerateNormalX(randGen, lanczosVecs_dev, n + n % 2, zero, one));
  ValueType_ normQ1;
  CUBLAS_CHECK(cublasnrm2(cublas_h, n, lanczosVecs_dev, 1, &normQ1, stream));

  auto h_val = 1 / normQ1;
  CUBLAS_CHECK(cublasscal(cublas_h, n, &h_val, lanczosVecs_dev, 1, stream));

  // Estimate number of Lanczos iterations
  //   See bounds in Kuczynski and Wozniakowski (1992).
  // const ValueType_ relError = 0.25;  // Relative error
  // const ValueType_ failProb = 1e-4;  // Probability of failure
  // maxIter_curr = log(n/pow(failProb,2))/(4*std::sqrt(relError)) + 1;
  // maxIter_curr = min(maxIter_curr, restartIter);

  // Obtain tridiagonal matrix with Lanczos
  *effIter = 0;
  ValueType_ shift_val = 0.0;
  ValueType_ *shift = &shift_val;
  // maxIter_curr = min(maxIter, restartIter);
  status = performLanczosIteration<IndexType_, ValueType_>(
    handle, A, effIter, maxIter_curr, *shift, 0, reorthogonalize, alpha_host,
    beta_host, lanczosVecs_dev, work_dev);
  if (status) WARNING("error in Lanczos iteration");
  *totalIter += *effIter;

  // Apply Lanczos method until convergence
  ValueType_ shiftLower = 1;
  ValueType_ shiftUpper = -1;
  while (*totalIter < maxIter && beta_host[*effIter - 1] > tol * shiftLower) {
    // Determine number of restart steps
    //   Number of steps must be even due to Francis algorithm
    IndexType_ iter_new = nEigVecs + 1;
    if (restartIter - (maxIter - *totalIter) > nEigVecs + 1)
      iter_new = restartIter - (maxIter - *totalIter);
    if ((restartIter - iter_new) % 2) iter_new -= 1;
    if (iter_new == *effIter) break;

    // Implicit restart of Lanczos method
    status = lanczosRestart<IndexType_, ValueType_>(
      handle, n, *effIter, iter_new, &shiftUpper, &shiftLower, alpha_host,
      beta_host, Z_host, work_host, lanczosVecs_dev, work_dev, false);
    if (status) WARNING("error in Lanczos implicit restart");
    *effIter = iter_new;

    // Check for convergence
    if (beta_host[*effIter - 1] <= tol * fabs(shiftLower)) break;

    // Proceed with Lanczos method
    // maxIter_curr = min(restartIter, maxIter-*totalIter+*effIter);
    status = performLanczosIteration<IndexType_, ValueType_>(
      handle, A, effIter, maxIter_curr, *shift, tol * fabs(shiftLower),
      reorthogonalize, alpha_host, beta_host, lanczosVecs_dev, work_dev);
    if (status) WARNING("error in Lanczos iteration");
    *totalIter += *effIter - iter_new;
  }

  // Warning if Lanczos has failed to converge
  if (beta_host[*effIter - 1] > tol * fabs(shiftLower)) {
    WARNING("implicitly restarted Lanczos failed to converge");
  }
  for (int i = 0; i < restartIter; ++i) {
    for (int j = 0; j < restartIter; ++j) Z_host[i * restartIter + j] = 0;
  }
  // Solve tridiagonal system
  memcpy(work_host + 2 * (*effIter), alpha_host,
         (*effIter) * sizeof(ValueType_));
  memcpy(work_host + 3 * (*effIter), beta_host,
         (*effIter - 1) * sizeof(ValueType_));
  Lapack<ValueType_>::steqr('I', *effIter, work_host + 2 * (*effIter),
                            work_host + 3 * (*effIter), Z_host, *effIter,
                            work_host);

  // note: We need to pick the top nEigVecs eigenvalues
  // but effItter can be larger than nEigVecs
  // hence we add an offset for that case, because we want to access top nEigVecs eigenpairs in the
  // matrix of size effIter. remember the array is sorted, so it is not needed for smallest
  // eigenvalues case because the first ones are the smallest ones

  IndexType_ top_eigenparis_idx_offset = *effIter - nEigVecs;

  // Debug : print nEigVecs largest eigenvalues
  // for (int i = top_eigenparis_idx_offset; i < *effIter; ++i)
  //  std::cout <<*(work_host+(2*(*effIter)+i))<< " ";
  // std::cout <<std::endl;

  // Debug : print nEigVecs largest eigenvectors
  // for (int i = top_eigenparis_idx_offset; i < *effIter; ++i)
  //{
  //  for (int j = 0; j < *effIter; ++j)
  //    std::cout <<Z_host[i*(*effIter)+j]<< " ";
  //  std::cout <<std::endl;
  //}

  // Obtain desired eigenvalues by applying shift
  for (i = 0; i < *effIter; ++i) work_host[i + 2 * (*effIter)] -= *shift;

  for (i = 0; i < top_eigenparis_idx_offset; ++i)
    work_host[i + 2 * (*effIter)] = 0;

  // Copy results to device memory
  // skip smallest eigenvalue if needed
  CUDA_TRY(cudaMemcpy(eigVals_dev,
                      work_host + 2 * (*effIter) + top_eigenparis_idx_offset,
                      nEigVecs * sizeof(ValueType_), cudaMemcpyHostToDevice));

  // skip smallest eigenvector if needed
  CUDA_TRY(cudaMemcpy(
    work_dev, Z_host + (top_eigenparis_idx_offset * (*effIter)),
    (*effIter) * nEigVecs * sizeof(ValueType_), cudaMemcpyHostToDevice));

  // Convert eigenvectors from Lanczos basis to standard basis
  CUBLAS_CHECK(cublasgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N, n, nEigVecs,
                          *effIter, &one, lanczosVecs_dev, n, work_dev,
                          *effIter, &zero, eigVecs_dev, n, stream));

  // Clean up and exit
  CUDA_TRY(curandDestroyGenerator(randGen));
  return 0;
}

/// Compute largest eigenvectors of symmetric matrix
/** Computes eigenvalues and eigenvectors that are least
 *  positive. If matrix is positive definite or positive
 *  semidefinite, the computed eigenvalues are largest in
 *  magnitude.
 *
 *  The largest eigenvalue is estimated by performing several
 *  Lanczos iterations. An implicitly restarted Lanczos method is
 *  then applied to A+s*I, where s is negative the largest
 *  eigenvalue.
 *
 *  CNMEM must be initialized before calling this function.
 *
 *  @param A Matrix.
 *  @param nEigVecs Number of eigenvectors to compute.
 *  @param maxIter Maximum number of Lanczos steps. Does not include
 *    Lanczos steps used to estimate largest eigenvalue.
 *  @param restartIter Maximum size of Lanczos system before
 *    performing an implicit restart. Should be at least 4.
 *  @param tol Convergence tolerance. Lanczos iteration will
 *    terminate when the residual norm is less than tol*theta, where
 *    theta is an estimate for the largest unwanted eigenvalue
 *    (i.e. the (nEigVecs+1)th largest eigenvalue).
 *  @param reorthogonalize Whether to reorthogonalize Lanczos
 *    vectors.
 *  @param iter On exit, pointer to total number of Lanczos
 *    iterations performed. Does not include Lanczos steps used to
 *    estimate largest eigenvalue.
 *  @param eigVals_dev (Output, device memory, nEigVecs entries)
 *    Largest eigenvalues of matrix.
 *  @param eigVecs_dev (Output, device memory, n*nEigVecs entries)
 *    Eigenvectors corresponding to largest eigenvalues of
 *    matrix. Vectors are stored as columns of a column-major matrix
 *    with dimensions n x nEigVecs.
 *  @return error flag.
 */
template <typename IndexType_, typename ValueType_>
int computeLargestEigenvectors(handle_t handle,
                               sparse_matrix_t<IndexType_, ValueType_> const &A,
                               IndexType_ nEigVecs, IndexType_ maxIter,
                               IndexType_ restartIter, ValueType_ tol,
                               bool reorthogonalize, IndexType_ &iter,
                               ValueType_ *__restrict__ eigVals_dev,
                               ValueType_ *__restrict__ eigVecs_dev,
                               unsigned long long seed = 123456) {
  // Matrix dimension
  IndexType_ n = A.nrows;

  // Check that parameters are valid
  RAFT_EXPECT(nEigVecs > 0 && nEigVecs <= n, "Invalid number of eigenvectors.");
  RAFT_EXPECT(restartIter > 0, "Invalid restartIter.");
  RAFT_EXPECT(tol > 0, "Invalid tolerance.");
  RAFT_EXPECT(maxIter >= nEigVecs, "Invalid maxIter.");
  RAFT_EXPECT(restartIter >= nEigVecs, "Invalid restartIter.");

  // Allocate memory
  std::vector<ValueType_> alpha_host_v(restartIter);
  std::vector<ValueType_> beta_host_v(restartIter);

  ValueType_ *alpha_host = alpha_host_v.data();
  ValueType_ *beta_host = beta_host_v.data();

  //TODO: replace and fix allocation via RAFT handle
  vector_t<ValueType_> lanczosVecs_dev(handle, n * (restartIter + 1));
  vector_t<ValueType_> work_dev(handle, (n + restartIter) * restartIter);

  // Perform Lanczos method
  IndexType_ effIter;
  int status = computeLargestEigenvectors(
    handle, &A, nEigVecs, maxIter, restartIter, tol, reorthogonalize, &effIter,
    &iter, alpha_host, beta_host, lanczosVecs_dev.raw(), work_dev.raw(),
    eigVals_dev, eigVecs_dev, seed);

  // Clean up and return
  return status;
}

}  // namespace raft
