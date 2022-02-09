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

#include <raft/spectral/detail/matrix_wrappers.cuh>

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
using sparse_mv_alt_t = detail::sparse_mv_alg_t;

// Vector "view"-like aggregate for linear algebra purposes
//
template <typename value_type>
using vector_view_t = detail::vector_view_t<value_type>;

template <typename value_type>
using vector_t = detail::vector_t<value_type>;

template <typename index_type, typename value_type>
using sparse_matrix_t = detail::sparse_matrix_t<index_type, value_type>;

template <typename index_type, typename value_type>
using laplacian_matrix_t = detail::laplacian_matrix_t<index_type, value_type>;

template <typename index_type, typename value_type>
using modularity_matrix_t = detail::modularity_matrix_t<index_type, value_type>;

}  // namespace matrix
}  // namespace spectral
}  // namespace raft
