/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>

namespace raft::sparse::solver {

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
