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

#define instantiate_raft_neighbors_ivf_pq_build(T, IdxT)                                 \
  template raft::neighbors::ivf_pq::index<IdxT> raft::neighbors::ivf_pq::build<T, IdxT>( \
    raft::resources const& handle,                                                       \
    const raft::neighbors::ivf_pq::index_params& params,                                 \
    raft::device_matrix_view<const T, IdxT, row_major> dataset);                         \
                                                                                         \
  template auto raft::neighbors::ivf_pq::build(                                          \
    raft::resources const& handle,                                                       \
    const raft::neighbors::ivf_pq::index_params& params,                                 \
    const T* dataset,                                                                    \
    IdxT n_rows,                                                                         \
    uint32_t dim)                                                                        \
    ->raft::neighbors::ivf_pq::index<IdxT>;

instantiate_raft_neighbors_ivf_pq_build(float, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_build
