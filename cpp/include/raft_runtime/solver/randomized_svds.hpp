/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/svds_config.hpp>

#include <cstdint>

namespace raft::runtime::solver {

/**
 * @defgroup sparse_randomized_svd_runtime Sparse Randomized SVD Runtime API
 * @{
 */

void sparse_randomized_svd_float(
  const raft::resources& handle,
  const raft::sparse::solver::sparse_svd_config<float>& config,
  raft::device_vector_view<int, uint32_t, raft::row_major> indptr,
  raft::device_vector_view<int, uint32_t, raft::row_major> indices,
  raft::device_vector_view<float, uint32_t, raft::row_major> data,
  int n_rows,
  int n_cols,
  int nnz,
  raft::device_vector_view<float, uint32_t> singular_values,
  raft::device_matrix_view<float, uint32_t, raft::col_major> U,
  raft::device_matrix_view<float, uint32_t, raft::col_major> Vt);

void sparse_randomized_svd_double(
  const raft::resources& handle,
  const raft::sparse::solver::sparse_svd_config<double>& config,
  raft::device_vector_view<int, uint32_t, raft::row_major> indptr,
  raft::device_vector_view<int, uint32_t, raft::row_major> indices,
  raft::device_vector_view<double, uint32_t, raft::row_major> data,
  int n_rows,
  int n_cols,
  int nnz,
  raft::device_vector_view<double, uint32_t> singular_values,
  raft::device_matrix_view<double, uint32_t, raft::col_major> U,
  raft::device_matrix_view<double, uint32_t, raft::col_major> Vt);

/** @} */

}  // namespace raft::runtime::solver
