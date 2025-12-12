/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
 *  @tparam IndexTypeT the type of data used for indexing.
 *  @tparam ValueTypeT the type of data used for weights, distances.
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
 *  @tparam IndexTypeT the type of data used for indexing.
 *  @tparam ValueTypeT the type of data used for weights, distances.
 *  @param handle the raft handle.
 *  @param config lanczos config used to set hyperparameters
 *  @param A Sparse matrix in COO format.
 *  @param v0 Optional Initial lanczos vector
 *  @param eigenvalues output eigenvalues
 *  @param eigenvectors output eigenvectors
 *  @return Zero if successful. Otherwise non-zero.
 */
template <typename IndexTypeT, typename ValueTypeT>
auto lanczos_compute_smallest_eigenvectors(
  raft::resources const& handle,
  lanczos_solver_config<ValueTypeT> const& config,
  raft::device_coo_matrix_view<ValueTypeT, IndexTypeT, IndexTypeT, IndexTypeT> A,
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
 *  @param rows Vector view of the rows of the sparse CSR matrix.
 *  @param cols Vector view of the cols of the sparse CSR matrix.
 *  @param vals Vector view of the vals of the sparse CSR matrix.
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

}  // namespace raft::sparse::solver

#endif
