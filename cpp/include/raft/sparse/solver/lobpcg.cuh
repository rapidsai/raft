/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/sparse/solver/detail/lobpcg.cuh>

namespace raft::sparse::solver {

template <typename value_t, typename index_t>
void lobpcg(
  const raft::handle_t& handle,
  // IN
  raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A,    // shape=(n,n)
  raft::device_matrix_view<value_t, index_t, raft::col_major> X,  // shape=(n,k) IN OUT Eigvectors
  raft::device_vector_view<value_t, index_t> W,                   // shape=(k) OUT Eigvals
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B =
    std::nullopt,  // shape=(n,n)
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> M =
    std::nullopt,  // shape=(n,n)
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> Y =
    std::nullopt,  // Constraint matrix shape=(n,Y)
  value_t tol            = 0,
  std::uint32_t max_iter = 20,
  bool largest           = true)
{
  detail::lobpcg(handle, A, X, W, B, M, Y, tol, max_iter, largest);
}
};  // namespace raft::sparse::solver