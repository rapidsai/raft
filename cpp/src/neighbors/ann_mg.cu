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


#include <raft/neighbors/ann_mg-inl.cuh>

#define instantiate_raft_neighbors_ann_mg(T, IdxT)                                              \
    template auto raft::neighbors::mg::build<T, IdxT>(                                          \
        const std::vector<int> device_ids,                                                      \
        raft::neighbors::mg::dist_mode mode,                                                    \
        const ivf_flat::index_params& index_params,                                             \
        raft::host_matrix_view<const T, IdxT, row_major> index_dataset                          \
    ) -> raft::neighbors::mg::detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>;          \
                                                                                                \
    template auto raft::neighbors::mg::build<T>(                                                \
        const std::vector<int> device_ids,                                                      \
        raft::neighbors::mg::dist_mode mode,                                                    \
        const ivf_pq::index_params& index_params,                                               \
        raft::host_matrix_view<const T, uint32_t, row_major> index_dataset                      \
    ) -> raft::neighbors::mg::detail::ann_mg_index<ivf_pq::index<uint32_t>, T, uint32_t>;       \
                                                                                                \
    template void raft::neighbors::mg::extend<T, IdxT>(                                         \
        raft::neighbors::mg::detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,    \
        raft::host_matrix_view<const T, IdxT, row_major> new_vectors,                           \
        raft::host_matrix_view<const IdxT, IdxT, row_major> new_indices);                       \
                                                                                                \
    template void raft::neighbors::mg::extend<T>(                                               \
        raft::neighbors::mg::detail::ann_mg_index<ivf_pq::index<uint32_t>, T, uint32_t>& index, \
        raft::host_matrix_view<const T, uint32_t, row_major> new_vectors,                       \
        raft::host_matrix_view<const uint32_t, uint32_t, row_major> new_indices);               \
                                                                                                \
    template void raft::neighbors::mg::search<T, IdxT>(                                         \
        raft::neighbors::mg::detail::ann_mg_index<ivf_flat::index<T, IdxT>, T, IdxT>& index,    \
        const ivf_flat::search_params& search_params,                                           \
        raft::host_matrix_view<const T, IdxT, row_major> query_dataset,                         \
        raft::host_matrix_view<IdxT, IdxT, row_major> neighbors,                                \
        raft::host_matrix_view<float, IdxT, row_major> distances);                              \
                                                                                                \
    template void raft::neighbors::mg::search<T>(                                               \
        raft::neighbors::mg::detail::ann_mg_index<ivf_pq::index<uint32_t>, T, uint32_t>& index, \
        const ivf_pq::search_params& search_params,                                             \
        raft::host_matrix_view<const T, uint32_t, row_major> query_dataset,                     \
        raft::host_matrix_view<uint32_t, uint32_t, row_major> neighbors,                        \
        raft::host_matrix_view<float, uint32_t, row_major> distances);                          \

instantiate_raft_neighbors_ann_mg(float, uint32_t);

#undef instantiate_raft_neighbors_ann_mg
