/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/sparse/solver/lanczos.cuh>

#define FUNC_DEF(IndexType, ValueType)                                                 \
  void lanczos_solver(                                                                 \
    const raft::resources& handle,                                                     \
    raft::sparse::solver::lanczos_solver_config<ValueType> config,                     \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> rows,               \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> cols,               \
    raft::device_vector_view<ValueType, uint32_t, raft::row_major> vals,               \
    std::optional<raft::device_vector_view<ValueType, uint32_t, raft::row_major>> v0,  \
    raft::device_vector_view<ValueType, uint32_t, raft::col_major> eigenvalues,        \
    raft::device_matrix_view<ValueType, uint32_t, raft::col_major> eigenvectors)       \
  {                                                                                    \
    raft::sparse::solver::lanczos_compute_smallest_eigenvectors<IndexType, ValueType>( \
      handle, config, rows, cols, vals, v0, eigenvalues, eigenvectors);                \
  }
