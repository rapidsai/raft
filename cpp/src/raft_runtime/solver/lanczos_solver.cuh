/*
 * Copyright (c) 2024-2024, NVIDIA CORPORATION.
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
#include <raft/spectral/cluster_solvers.cuh>
#include <raft/spectral/eigen_solvers.cuh>
#include <raft/spectral/matrix_wrappers.hpp>

#include <raft_runtime/solver/lanczos.hpp>

template <typename IndexType, typename ValueType>
void run_lanczos_solver(const raft::resources& handle,
                        IndexType* rows,
                        IndexType* cols,
                        ValueType* vals,
                        int nnz,
                        int n,
                        int n_components,
                        int max_iterations,
                        int ncv,
                        ValueType tolerance,
                        uint64_t seed,
                        ValueType* v0,
                        ValueType* eigenvalues,
                        ValueType* eigenvectors)
{
  auto stream = raft::resource::get_cuda_stream(handle);
  raft::device_vector_view<IndexType, uint32_t, raft::row_major> rows_view =
    raft::make_device_vector_view<IndexType, uint32_t, raft::row_major>(rows, n + 1);
  raft::device_vector_view<IndexType, uint32_t, raft::row_major> cols_view =
    raft::make_device_vector_view<IndexType, uint32_t, raft::row_major>(cols, nnz);
  raft::device_vector_view<ValueType, uint32_t, raft::row_major> vals_view =
    raft::make_device_vector_view<ValueType, uint32_t, raft::row_major>(vals, nnz);
  raft::device_vector_view<ValueType, uint32_t, raft::row_major> v0_view =
    raft::make_device_vector_view<ValueType, uint32_t, raft::row_major>(v0, n);
  raft::device_vector_view<ValueType, uint32_t, raft::col_major> eigenvalues_view =
    raft::make_device_vector_view<ValueType, uint32_t, raft::col_major>(eigenvalues, n_components);
  raft::device_matrix_view<ValueType, uint32_t, raft::col_major> eigenvectors_view =
    raft::make_device_matrix_view<ValueType, uint32_t, raft::col_major>(
      eigenvectors, n, n_components);

  raft::spectral::matrix::sparse_matrix_t<IndexType, ValueType> const csr_m{
    handle, rows_view.data_handle(), cols_view.data_handle(), vals_view.data_handle(), n, nnz};
  raft::sparse::solver::lanczos_solver_config<IndexType, ValueType> config{
    n_components, max_iterations, ncv, tolerance, seed};
  raft::sparse::solver::lanczos_compute_smallest_eigenvectors<IndexType, ValueType>(
    handle, csr_m, config, v0_view, eigenvalues_view, eigenvectors_view);
}

#define FUNC_DEF(IndexType, ValueType)                       \
  void lanczos_solver(const raft::resources& handle,         \
                      IndexType* rows,                       \
                      IndexType* cols,                       \
                      ValueType* vals,                       \
                      int nnz,                               \
                      int n,                                 \
                      int n_components,                      \
                      int max_iterations,                    \
                      int ncv,                               \
                      ValueType tolerance,                   \
                      uint64_t seed,                         \
                      ValueType* v0,                         \
                      ValueType* eigenvalues,                \
                      ValueType* eigenvectors)               \
  {                                                          \
    run_lanczos_solver<IndexType, ValueType>(handle,         \
                                             rows,           \
                                             cols,           \
                                             vals,           \
                                             nnz,            \
                                             n,              \
                                             n_components,   \
                                             max_iterations, \
                                             ncv,            \
                                             tolerance,      \
                                             seed,           \
                                             v0,             \
                                             eigenvalues,    \
                                             eigenvectors);  \
  }
