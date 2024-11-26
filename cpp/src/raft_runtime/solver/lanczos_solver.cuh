/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
