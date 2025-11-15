/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/spectral/detail/matrix_wrappers.hpp>

// =========================================================
// Useful macros
// =========================================================

namespace raft {
namespace spectral {
namespace matrix {

using size_type = int;  // for now; TODO: move it in appropriate header

// specifies type of algorithm used
// for SpMv:
//
using detail::sparse_mv_alg_t;

// Vector "view"-like aggregate for linear algebra purposes
//
using detail::vector_view_t;

using detail::vector_t;

using detail::sparse_matrix_t;

using detail::laplacian_matrix_t;

using detail::modularity_matrix_t;

}  // namespace matrix
}  // namespace spectral
}  // namespace raft
