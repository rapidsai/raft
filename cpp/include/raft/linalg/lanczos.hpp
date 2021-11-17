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

#include "detail/lanczos.hpp"

namespace raft {

// =========================================================
// Eigensolver
// =========================================================

/**  
 * @brief  Compute smallest eigenvectors of symmetric matrix
 *    Computes eigenvalues and eigenvectors that are least
 *    positive. If matrix is positive definite or positive
 *    semidefinite, the computed eigenvalues are smallest in
 *    magnitude.
 *    The largest eigenvalue is estimated by performing several
 *    Lanczos iterations. An implicitly restarted Lanczos method is
 *    then applied to A+s*I, where s is negative the largest
 *    eigenvalue.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
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
 *  @param seed random seed.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
int computeSmallestEigenvectors(
  handle_t const &handle, sparse_matrix_t<index_type_t, value_type_t> const *A,
  index_type_t nEigVecs, index_type_t maxIter, index_type_t restartIter,
  value_type_t tol, bool reorthogonalize, index_type_t *effIter,
  index_type_t *totalIter, value_type_t *shift,
  value_type_t *__restrict__ alpha_host, value_type_t *__restrict__ beta_host,
  value_type_t *__restrict__ lanczosVecs_dev,
  value_type_t *__restrict__ work_dev, value_type_t *__restrict__ eigVals_dev,
  value_type_t *__restrict__ eigVecs_dev, unsigned long long seed) {
  return raft::detail::computeSmallestEigenvectors(handle, A, nEigVecs, maxIter, restartIter, tol, reorthogonalize, effIter, totalIter, shift,
    alpha_host, beta_host, lanczosVecs_dev, work_dev, eigVals_dev, eigVecs_dev, seed);
}

/**  
 *  @brief  Compute smallest eigenvectors of symmetric matrix
 *    Computes eigenvalues and eigenvectors that are least
 *    positive. If matrix is positive definite or positive
 *    semidefinite, the computed eigenvalues are smallest in
 *    magnitude.
 *    The largest eigenvalue is estimated by performing several
 *    Lanczos iterations. An implicitly restarted Lanczos method is
 *    then applied to A+s*I, where s is negative the largest
 *    eigenvalue.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
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
 *  @param seed random seed.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
int computeSmallestEigenvectors(
  handle_t const &handle, sparse_matrix_t<index_type_t, value_type_t> const &A,
  index_type_t nEigVecs, index_type_t maxIter, index_type_t restartIter,
  value_type_t tol, bool reorthogonalize, index_type_t &iter,
  value_type_t *__restrict__ eigVals_dev,
  value_type_t *__restrict__ eigVecs_dev, unsigned long long seed = 1234567) {
  return raft::detail::computeSmallestEigenvectors(handle, A, nEigVecs, maxIter, restartIter, tol, reorthogonalize, iter, eigVals_dev, eigVecs_dev, seed);
}

// =========================================================
// Eigensolver
// =========================================================

/**  
 *  @brief Compute largest eigenvectors of symmetric matrix
 *    Computes eigenvalues and eigenvectors that are least
 *    positive. If matrix is positive definite or positive
 *    semidefinite, the computed eigenvalues are largest in
 *    magnitude.
 *    The largest eigenvalue is estimated by performing several
 *    Lanczos iterations. An implicitly restarted Lanczos method is
 *    then applied.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
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
 *  @param seed random seed.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
int computeLargestEigenvectors(
  handle_t const &handle, sparse_matrix_t<index_type_t, value_type_t> const *A,
  index_type_t nEigVecs, index_type_t maxIter, index_type_t restartIter,
  value_type_t tol, bool reorthogonalize, index_type_t *effIter,
  index_type_t *totalIter, value_type_t *__restrict__ alpha_host,
  value_type_t *__restrict__ beta_host,
  value_type_t *__restrict__ lanczosVecs_dev,
  value_type_t *__restrict__ work_dev, value_type_t *__restrict__ eigVals_dev,
  value_type_t *__restrict__ eigVecs_dev, unsigned long long seed) {
  return raft::detail::computeLargestEigenvectors(handle, A, nEigVecs, maxIter, restartIter, tol, reorthogonalize, effIter, totalIter, alpha_host, beta_host,
    lanczosVecs_dev, work_dev, eigVals_dev, eigVecs_dev, seed);
}

/**  
 *  @brief  Compute largest eigenvectors of symmetric matrix
 *    Computes eigenvalues and eigenvectors that are least
 *    positive. If matrix is positive definite or positive
 *    semidefinite, the computed eigenvalues are largest in
 *    magnitude.
 *    The largest eigenvalue is estimated by performing several
 *    Lanczos iterations. An implicitly restarted Lanczos method is
 *    then applied to A+s*I, where s is negative the largest
 *    eigenvalue.
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
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
 *  @param seed random seed.
 *  @return error flag.
 */
template <typename index_type_t, typename value_type_t>
int computeLargestEigenvectors(
  handle_t const &handle, sparse_matrix_t<index_type_t, value_type_t> const &A,
  index_type_t nEigVecs, index_type_t maxIter, index_type_t restartIter,
  value_type_t tol, bool reorthogonalize, index_type_t &iter,
  value_type_t *__restrict__ eigVals_dev,
  value_type_t *__restrict__ eigVecs_dev, unsigned long long seed = 123456) {
  return raft::detail::computeLargestEigenvectors(handle, A, nEigVecs, maxIter, restartIter, tol, reorthogonalize, iter, eigVals_dev, eigVecs_dev, seed);
}

}  // namespace raft
