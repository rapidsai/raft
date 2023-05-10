/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_pq-inl.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>  // raft::neighbors::ivf_pq::index

#define instantiate_raft_neighbors_ivf_pq_extend(T, IdxT)                                 \
  template raft::neighbors::ivf_pq::index<IdxT> raft::neighbors::ivf_pq::extend<T, IdxT>( \
    raft::resources const& handle,                                                        \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                       \
    std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,     \
    const raft::neighbors::ivf_pq::index<IdxT>& idx);                                     \
                                                                                          \
  template void raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::resources const& handle,                                                        \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                       \
    std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,     \
    raft::neighbors::ivf_pq::index<IdxT>* idx);                                           \
                                                                                          \
  template auto raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::resources const& handle,                                                        \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                                      \
    const T* new_vectors,                                                                 \
    const IdxT* new_indices,                                                              \
    IdxT n_rows)                                                                          \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                               \
                                                                                          \
  template void raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::resources const& handle,                                                        \
    raft::neighbors::ivf_pq::index<IdxT>* idx,                                            \
    const T* new_vectors,                                                                 \
    const IdxT* new_indices,                                                              \
    IdxT n_rows);

instantiate_raft_neighbors_ivf_pq_extend(float, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_extend
