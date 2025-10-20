/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#ifndef __LANCZOS_H
#define __LANCZOS_H

#pragma once

#include <raft/sparse/solver/detail/lanczos.cuh>
#include <raft/sparse/solver/lanczos_types.hpp>
#include <raft/spectral/matrix_wrappers.hpp>

namespace raft::sparse::solver {

// =========================================================
// Eigensolver
// =========================================================

/**
 *  @brief Find the eigenpairs using lanczos solver
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param config lanczos config used to set hyperparameters
 *  @param A Sparse matrix in CSR format.
 *  @param v0 Optional Initial lanczos vector
 *  @param eigenvalues output eigenvalues
 *  @param eigenvectors output eigenvectors
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename IndexTypeT, typename ValueTypeT>
auto lanczos_compute_smallest_eigenvectors(
  raft::resources const& handle,
  lanczos_solver_config<ValueTypeT> const& config,
  raft::device_csr_matrix_view<ValueTypeT, IndexTypeT, IndexTypeT, IndexTypeT> A,
  std::optional<raft::device_vector_view<ValueTypeT, uint32_t, raft::row_major>> v0,
  raft::device_vector_view<ValueTypeT, uint32_t, raft::col_major> eigenvalues,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> eigenvectors) -> int
{
  return detail::lanczos_compute_smallest_eigenvectors<IndexTypeT, ValueTypeT>(
    handle, config, A, v0, eigenvalues, eigenvectors);
}

/**
 *  @brief Find the eigenpairs using lanczos solver
 *  @tparam index_type_t the type of data used for indexing.
 *  @tparam value_type_t the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param config lanczos config used to set hyperparameters
 *  @param rows Vector view of the rows of the sparse matrix.
 *  @param cols Vector view of the cols of the sparse matrix.
 *  @param vals Vector view of the vals of the sparse matrix.
 *  @param v0 Optional Initial lanczos vector
 *  @param eigenvalues output eigenvalues
 *  @param eigenvectors output eigenvectors
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename IndexTypeT, typename ValueTypeT>
auto lanczos_compute_smallest_eigenvectors(
  raft::resources const& handle,
  lanczos_solver_config<ValueTypeT> const& config,
  raft::device_vector_view<IndexTypeT, uint32_t, raft::row_major> rows,
  raft::device_vector_view<IndexTypeT, uint32_t, raft::row_major> cols,
  raft::device_vector_view<ValueTypeT, uint32_t, raft::row_major> vals,
  std::optional<raft::device_vector_view<ValueTypeT, uint32_t, raft::row_major>> v0,
  raft::device_vector_view<ValueTypeT, uint32_t, raft::col_major> eigenvalues,
  raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> eigenvectors) -> int
{
  IndexTypeT ncols = rows.extent(0) - 1;
  IndexTypeT nrows = rows.extent(0) - 1;
  IndexTypeT nnz   = cols.extent(0);

  auto csr_structure =
    raft::make_device_compressed_structure_view<IndexTypeT, IndexTypeT, IndexTypeT>(
      const_cast<IndexTypeT*>(rows.data_handle()),
      const_cast<IndexTypeT*>(cols.data_handle()),
      ncols,
      nrows,
      nnz);

  auto csr_matrix =
    raft::make_device_csr_matrix_view<ValueTypeT, IndexTypeT, IndexTypeT, IndexTypeT>(
      const_cast<ValueTypeT*>(vals.data_handle()), csr_structure);

  return lanczos_compute_smallest_eigenvectors<IndexTypeT, ValueTypeT>(
    handle, config, csr_matrix, v0, eigenvalues, eigenvectors);
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
template <typename index_type_t, typename value_type_t, typename nnz_type_t>
int computeSmallestEigenvectors(
  raft::resources const& handle,
  raft::spectral::matrix::sparse_matrix_t<index_type_t, value_type_t, nnz_type_t> const& A,
  index_type_t nEigVecs,
  index_type_t maxIter,
  index_type_t restartIter,
  value_type_t tol,
  bool reorthogonalize,
  index_type_t& iter,
  value_type_t* __restrict__ eigVals_dev,
  value_type_t* __restrict__ eigVecs_dev,
  unsigned long long seed = 1234567)
{
  return detail::computeSmallestEigenvectors(handle,
                                             A,
                                             nEigVecs,
                                             maxIter,
                                             restartIter,
                                             tol,
                                             reorthogonalize,
                                             iter,
                                             eigVals_dev,
                                             eigVecs_dev,
                                             seed);
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
template <typename index_type_t, typename value_type_t, typename nnz_type_t>
int computeLargestEigenvectors(
  raft::resources const& handle,
  raft::spectral::matrix::sparse_matrix_t<index_type_t, value_type_t, nnz_type_t> const& A,
  index_type_t nEigVecs,
  index_type_t maxIter,
  index_type_t restartIter,
  value_type_t tol,
  bool reorthogonalize,
  index_type_t& iter,
  value_type_t* __restrict__ eigVals_dev,
  value_type_t* __restrict__ eigVecs_dev,
  unsigned long long seed = 123456)
{
  return detail::computeLargestEigenvectors(handle,
                                            A,
                                            nEigVecs,
                                            maxIter,
                                            restartIter,
                                            tol,
                                            reorthogonalize,
                                            iter,
                                            eigVals_dev,
                                            eigVecs_dev,
                                            seed);
}

}  // namespace raft::sparse::solver

#endif
