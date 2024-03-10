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

#pragma once

#include "ivf_pq_compute_similarity_filters_test-ext.cuh"

#include <raft/core/device_mdspan.hpp>  // raft::device_matrix_view
#include <raft/core/resources.hpp>      // raft::resources
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>  // raft::neighbors::ivf_pq::index
#include <raft/neighbors/sample_filter.cuh>
#include <raft/neighbors/sample_filter_types.hpp>

#include <cstdint>  // int64_t

#define instantiate_raft_neighbors_ivf_pq_search(T, IdxT)            \
  extern template void raft::neighbors::ivf_pq::search<T, IdxT>(     \
    raft::resources const& handle,                                   \
    const raft::neighbors::ivf_pq::search_params& params,            \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                 \
    raft::device_matrix_view<const T, uint32_t, row_major> queries,  \
    raft::device_matrix_view<IdxT, uint32_t, row_major> neighbors,   \
    raft::device_matrix_view<float, uint32_t, row_major> distances); \
                                                                     \
  extern template void raft::neighbors::ivf_pq::search<T, IdxT>(     \
    raft::resources const& handle,                                   \
    const raft::neighbors::ivf_pq::search_params& params,            \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                 \
    const T* queries,                                                \
    uint32_t n_queries,                                              \
    uint32_t k,                                                      \
    IdxT* neighbors,                                                 \
    float* distances,                                                \
    rmm::mr::device_memory_resource* mr);                            \
                                                                     \
  extern template void raft::neighbors::ivf_pq::search<T, IdxT>(     \
    raft::resources const& handle,                                   \
    const raft::neighbors::ivf_pq::search_params& params,            \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                 \
    const T* queries,                                                \
    uint32_t n_queries,                                              \
    uint32_t k,                                                      \
    IdxT* neighbors,                                                 \
    float* distances)

instantiate_raft_neighbors_ivf_pq_search(float, uint32_t);

#undef instantiate_raft_neighbors_ivf_pq_search

#define instantiate_raft_neighbors_ivf_pq_search_with_filtering(T, IdxT, FilterT)        \
  extern template void raft::neighbors::ivf_pq::search_with_filtering<T, IdxT, FilterT>( \
    raft::resources const& handle,                                                       \
    const search_params& params,                                                         \
    const index<IdxT>& idx,                                                              \
    raft::device_matrix_view<const T, uint32_t, row_major> queries,                      \
    raft::device_matrix_view<IdxT, uint32_t, row_major> neighbors,                       \
    raft::device_matrix_view<float, uint32_t, row_major> distances,                      \
    FilterT sample_filter)

#define COMMA ,
instantiate_raft_neighbors_ivf_pq_search_with_filtering(
  float, uint32_t, raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>);

instantiate_raft_neighbors_ivf_pq_search_with_filtering(
  float, uint32_t, raft::neighbors::filtering::none_ivf_sample_filter);

instantiate_raft_neighbors_ivf_pq_search_with_filtering(
  float, int64_t, raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>);

instantiate_raft_neighbors_ivf_pq_search_with_filtering(
  int8_t, int64_t, raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>);

#undef COMMA
#undef instantiate_raft_neighbors_ivf_pq_search_with_filtering
