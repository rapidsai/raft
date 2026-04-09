/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/spmm.hpp>
#include <raft/sparse/solver/svds_config.hpp>

namespace raft::sparse::solver {

/**
 * @brief Linear operator wrapping a CSR sparse matrix for use with sparse SVD solvers.
 *
 * Provides apply() (Y = A @ X) and apply_transpose() (Z = A^T @ X) using cuSPARSE SpMM.
 *
 * @note The cuSPARSE spmm wrapper requires int for indptr/indices types, so this operator
 *       is currently limited to int-indexed CSR matrices.
 *
 * @tparam ValueTypeT Data type of matrix values
 * @tparam NNZTypeT Type for number of non-zeros
 */
template <typename ValueTypeT, typename NNZTypeT = int>
struct csr_linear_operator {
  raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> A_;
  int m_;
  int n_;

  /**
   * @brief Construct from a const CSR matrix view
   */
  csr_linear_operator(raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> A,
                      int m,
                      int n)
    : A_(A), m_(m), n_(n)
  {
  }

  /**
   * @brief Construct from a mutable CSR matrix view (converts to const)
   */
  csr_linear_operator(raft::device_csr_matrix_view<ValueTypeT, int, int, NNZTypeT> A, int m, int n)
    : A_(raft::make_device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT>(
        A.get_elements().data(), A.structure_view())),
      m_(m),
      n_(n)
  {
  }

  int rows() const { return m_; }
  int cols() const { return n_; }

  /** @brief Access the underlying const CSR matrix view (for SpMV operations) */
  raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> csr_view() const
  {
    return A_;
  }

  /**
   * @brief Compute Y = A @ X
   * @param handle raft resources handle
   * @param X input dense matrix of shape (n, k) col-major
   * @param Y output dense matrix of shape (m, k) col-major
   */
  void apply(raft::resources const& handle,
             raft::device_matrix_view<const ValueTypeT, uint32_t, raft::col_major> X,
             raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> Y) const
  {
    ValueTypeT alpha = 1;
    ValueTypeT beta  = 0;
    raft::sparse::linalg::spmm(handle, false, false, &alpha, A_, X, &beta, Y);
  }

  /**
   * @brief Compute Z = A^T @ X
   * @param handle raft resources handle
   * @param X input dense matrix of shape (m, k) col-major
   * @param Z output dense matrix of shape (n, k) col-major
   */
  void apply_transpose(raft::resources const& handle,
                       raft::device_matrix_view<const ValueTypeT, uint32_t, raft::col_major> X,
                       raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> Z) const
  {
    ValueTypeT alpha = 1;
    ValueTypeT beta  = 0;
    raft::sparse::linalg::spmm(handle, true, false, &alpha, A_, X, &beta, Z);
  }
};

/**
 * @defgroup SparseLinearOperator concept
 *
 * Any type satisfying this interface can be used with the sparse SVD solvers.
 * Required methods:
 *   - int rows() const;
 *   - int cols() const;
 *   - void apply(raft::resources const& handle,
 *                device_matrix_view<const T, uint32_t, col_major> in,
 *                device_matrix_view<T, uint32_t, col_major> out) const;
 *   - void apply_transpose(raft::resources const& handle,
 *                          device_matrix_view<const T, uint32_t, col_major> in,
 *                          device_matrix_view<T, uint32_t, col_major> out) const;
 * @{
 */
/** @} */

}  // namespace raft::sparse::solver
