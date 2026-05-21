/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/linalg/spmm.hpp>

namespace raft::sparse::solver::detail {

/**
 * @brief CSR matrix wrapper providing apply() / apply_transpose() for the sparse SVD solver.
 *
 * This is NOT a general-purpose LinearOperator (in the sense of scipy's
 * `scipy.sparse.linalg.LinearOperator`): it only exposes the two products the
 * randomized SVD inner loop needs (Y = A @ X and Z = A^T @ X) via cuSPARSE SpMM.
 * It intentionally lives in `detail/` and is not part of the public API.
 *
 * @note The cuSPARSE spmm wrapper requires `int` for indptr/indices types, so this wrapper
 *       is currently limited to int-indexed CSR matrices.
 *
 * @tparam ValueTypeT Data type of matrix values
 * @tparam NNZTypeT Type for number of non-zeros
 */
template <typename ValueTypeT, typename NNZTypeT = int>
struct csr_linear_operator {
  /**
   * @brief Construct from a const CSR matrix view
   *
   * @note `m_`/`n_` are cached at construction because
   *       `raft::device_csr_matrix_view::structure_view()` is not const-qualified
   *       and cannot be invoked from a const member function.
   */
  explicit csr_linear_operator(raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> A)
    : A_(A), m_(A.structure_view().get_n_rows()), n_(A.structure_view().get_n_cols())
  {
  }

  int rows() const { return m_; }
  int cols() const { return n_; }

  /** @brief Access the underlying const CSR matrix view */
  raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> csr_view() const { return A_; }

  /**
   * @brief Compute Y = A @ X
   * @param[in] handle raft resources handle
   * @param[in] X input dense matrix of shape (n, k) col-major
   * @param[out] Y output dense matrix of shape (m, k) col-major
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
   * @param[in] handle raft resources handle
   * @param[in] X input dense matrix of shape (m, k) col-major
   * @param[out] Z output dense matrix of shape (n, k) col-major
   */
  void apply_transpose(raft::resources const& handle,
                       raft::device_matrix_view<const ValueTypeT, uint32_t, raft::col_major> X,
                       raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> Z) const
  {
    ValueTypeT alpha = 1;
    ValueTypeT beta  = 0;
    raft::sparse::linalg::spmm(handle, true, false, &alpha, A_, X, &beta, Z);
  }

 private:
  raft::device_csr_matrix_view<const ValueTypeT, int, int, NNZTypeT> A_;
  int m_;
  int n_;
};

}  // namespace raft::sparse::solver::detail
