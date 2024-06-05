/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <rmm/resource_ref.hpp>

#define instantiate_raft_neighbors_ivf_pq_search(T, IdxT)            \
  template void raft::neighbors::ivf_pq::search<T, IdxT>(            \
    raft::resources const& handle,                                   \
    const raft::neighbors::ivf_pq::search_params& params,            \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                 \
    raft::device_matrix_view<const T, uint32_t, row_major> queries,  \
    raft::device_matrix_view<IdxT, uint32_t, row_major> neighbors,   \
    raft::device_matrix_view<float, uint32_t, row_major> distances); \
                                                                     \
  template void raft::neighbors::ivf_pq::search<T, IdxT>(            \
    raft::resources const& handle,                                   \
    const raft::neighbors::ivf_pq::search_params& params,            \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                 \
    const T* queries,                                                \
    uint32_t n_queries,                                              \
    uint32_t k,                                                      \
    IdxT* neighbors,                                                 \
    float* distances)

instantiate_raft_neighbors_ivf_pq_search(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_search
