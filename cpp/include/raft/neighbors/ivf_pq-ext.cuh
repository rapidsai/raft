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

#pragma once

#include <raft/core/device_mdspan.hpp>      // raft::device_matrix_view
#include <raft/core/resources.hpp>          // raft::resources
#include <raft/neighbors/ivf_pq_types.hpp>  // raft::neighbors::ivf_pq::index
#include <raft/util/raft_explicit.hpp>      // RAFT_EXPLICIT

#include <cstdint>  // int64_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors::ivf_pq {

template <typename T, typename IdxT = uint32_t>
index<IdxT> build(raft::resources const& handle,
                  const index_params& params,
                  raft::device_matrix_view<const T, IdxT, row_major> dataset) RAFT_EXPLICIT;

template <typename T, typename IdxT>
index<IdxT> extend(raft::resources const& handle,
                   raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
                   std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,
                   const index<IdxT>& idx) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,
            index<IdxT>* idx) RAFT_EXPLICIT;

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<IdxT>& idx,
                           raft::device_matrix_view<const T, uint32_t, row_major> queries,
                           raft::device_matrix_view<IdxT, uint32_t, row_major> neighbors,
                           raft::device_matrix_view<float, uint32_t, row_major> distances,
                           IvfSampleFilterT sample_filter) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<IdxT>& idx,
            raft::device_matrix_view<const T, uint32_t, row_major> queries,
            raft::device_matrix_view<IdxT, uint32_t, row_major> neighbors,
            raft::device_matrix_view<float, uint32_t, row_major> distances) RAFT_EXPLICIT;

template <typename T, typename IdxT = uint32_t>
auto build(raft::resources const& handle,
           const index_params& params,
           const T* dataset,
           IdxT n_rows,
           uint32_t dim) -> index<IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            const index<IdxT>& idx,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            index<IdxT>* idx,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) RAFT_EXPLICIT;

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const raft::neighbors::ivf_pq::search_params& params,
                           const index<IdxT>& idx,
                           const T* queries,
                           uint32_t n_queries,
                           uint32_t k,
                           IdxT* neighbors,
                           float* distances,
                           IvfSampleFilterT sample_filter = IvfSampleFilterT{}) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const raft::neighbors::ivf_pq::search_params& params,
            const index<IdxT>& idx,
            const T* queries,
            uint32_t n_queries,
            uint32_t k,
            IdxT* neighbors,
            float* distances) RAFT_EXPLICIT;

}  // namespace raft::neighbors::ivf_pq

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_ivf_pq_build(T, IdxT)                                        \
  extern template raft::neighbors::ivf_pq::index<IdxT> raft::neighbors::ivf_pq::build<T, IdxT>( \
    raft::resources const& handle,                                                              \
    const raft::neighbors::ivf_pq::index_params& params,                                        \
    raft::device_matrix_view<const T, IdxT, row_major> dataset);                                \
                                                                                                \
  extern template auto raft::neighbors::ivf_pq::build(                                          \
    raft::resources const& handle,                                                              \
    const raft::neighbors::ivf_pq::index_params& params,                                        \
    const T* dataset,                                                                           \
    IdxT n_rows,                                                                                \
    uint32_t dim)                                                                               \
    ->raft::neighbors::ivf_pq::index<IdxT>;

instantiate_raft_neighbors_ivf_pq_build(float, int64_t);
instantiate_raft_neighbors_ivf_pq_build(half, int64_t);
instantiate_raft_neighbors_ivf_pq_build(int8_t, int64_t);
instantiate_raft_neighbors_ivf_pq_build(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_build

#define instantiate_raft_neighbors_ivf_pq_extend(T, IdxT)                                        \
  extern template raft::neighbors::ivf_pq::index<IdxT> raft::neighbors::ivf_pq::extend<T, IdxT>( \
    raft::resources const& handle,                                                               \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                              \
    std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,            \
    const raft::neighbors::ivf_pq::index<IdxT>& idx);                                            \
                                                                                                 \
  extern template void raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::resources const& handle,                                                               \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                              \
    std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,            \
    raft::neighbors::ivf_pq::index<IdxT>* idx);                                                  \
                                                                                                 \
  extern template auto raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::resources const& handle,                                                               \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                                             \
    const T* new_vectors,                                                                        \
    const IdxT* new_indices,                                                                     \
    IdxT n_rows)                                                                                 \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                                      \
                                                                                                 \
  extern template void raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::resources const& handle,                                                               \
    raft::neighbors::ivf_pq::index<IdxT>* idx,                                                   \
    const T* new_vectors,                                                                        \
    const IdxT* new_indices,                                                                     \
    IdxT n_rows);

instantiate_raft_neighbors_ivf_pq_extend(float, int64_t);
instantiate_raft_neighbors_ivf_pq_extend(half, int64_t);
instantiate_raft_neighbors_ivf_pq_extend(int8_t, int64_t);
instantiate_raft_neighbors_ivf_pq_extend(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_extend

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
    float* distances);                                               \
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

instantiate_raft_neighbors_ivf_pq_search(float, int64_t);
instantiate_raft_neighbors_ivf_pq_search(half, int64_t);
instantiate_raft_neighbors_ivf_pq_search(int8_t, int64_t);
instantiate_raft_neighbors_ivf_pq_search(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_search
