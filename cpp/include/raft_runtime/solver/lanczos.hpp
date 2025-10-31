/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/solver/lanczos_types.hpp>

#include <cstdint>

namespace raft::runtime::solver {

/**
 * @defgroup lanczos_runtime lanczos Runtime API
 * @{
 */

#define FUNC_DECL(IndexType, ValueType)                                               \
  void lanczos_solver(                                                                \
    const raft::resources& handle,                                                    \
    raft::sparse::solver::lanczos_solver_config<ValueType> config,                    \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> rows,              \
    raft::device_vector_view<IndexType, uint32_t, raft::row_major> cols,              \
    raft::device_vector_view<ValueType, uint32_t, raft::row_major> vals,              \
    std::optional<raft::device_vector_view<ValueType, uint32_t, raft::row_major>> v0, \
    raft::device_vector_view<ValueType, uint32_t, raft::col_major> eigenvalues,       \
    raft::device_matrix_view<ValueType, uint32_t, raft::col_major> eigenvectors)

FUNC_DECL(int, float);
FUNC_DECL(int64_t, float);
FUNC_DECL(int, double);
FUNC_DECL(int64_t, double);

#undef FUNC_DECL

/** @} */  // end group lanczos_runtime

}  // namespace raft::runtime::solver
