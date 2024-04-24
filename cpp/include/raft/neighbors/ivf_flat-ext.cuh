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

#include <raft/core/device_mdspan.hpp>  // raft::device_matrix_view
#include <raft/core/resources.hpp>      // raft::resources
#include <raft/neighbors/ivf_flat_serialize.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>  // raft::neighbors::ivf_flat::index
#include <raft/util/raft_explicit.hpp>        // RAFT_EXPLICIT

#include <rmm/resource_ref.hpp>

#include <cstdint>  // int64_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors::ivf_flat {

template <typename T, typename IdxT>
auto build(raft::resources const& handle,
           const index_params& params,
           const T* dataset,
           IdxT n_rows,
           uint32_t dim) -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto build(raft::resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const T, IdxT, row_major> dataset)
  -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
void build(raft::resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const T, IdxT, row_major> dataset,
           raft::neighbors::ivf_flat::index<T, IdxT>& idx) RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto build(raft::resources const& handle,
           const index_params& params,
           raft::host_matrix_view<const T, IdxT, row_major> dataset)
  -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
void build(raft::resources const& handle,
           const index_params& params,
           raft::host_matrix_view<const T, IdxT, row_major> dataset,
           raft::neighbors::ivf_flat::index<T, IdxT>& idx) RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            const index<T, IdxT>& orig_index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
            const index<T, IdxT>& orig_index) -> index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            index<T, IdxT>* index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
            index<T, IdxT>* index) RAFT_EXPLICIT;

template <typename T, typename IdxT>
auto extend(raft::resources const& handle,
            raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices,
            const raft::neighbors::ivf_flat::index<T, IdxT>& orig_index)
  -> raft::neighbors::ivf_flat::index<T, IdxT> RAFT_EXPLICIT;

template <typename T, typename IdxT>
void extend(raft::resources const& handle,
            raft::host_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices,
            index<T, IdxT>* index) RAFT_EXPLICIT;

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<T, IdxT>& index,
                           const T* queries,
                           uint32_t n_queries,
                           uint32_t k,
                           IdxT* neighbors,
                           float* distances,
                           rmm::device_async_resource_ref mr,
                           IvfSampleFilterT sample_filter = IvfSampleFilterT()) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<T, IdxT>& index,
            const T* queries,
            uint32_t n_queries,
            uint32_t k,
            IdxT* neighbors,
            float* distances,
            rmm::device_async_resource_ref mr) RAFT_EXPLICIT;

template <typename T, typename IdxT, typename IvfSampleFilterT>
void search_with_filtering(raft::resources const& handle,
                           const search_params& params,
                           const index<T, IdxT>& index,
                           raft::device_matrix_view<const T, IdxT, row_major> queries,
                           raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
                           raft::device_matrix_view<float, IdxT, row_major> distances,
                           IvfSampleFilterT sample_filter = IvfSampleFilterT()) RAFT_EXPLICIT;

template <typename T, typename IdxT>
void search(raft::resources const& handle,
            const search_params& params,
            const index<T, IdxT>& index,
            raft::device_matrix_view<const T, IdxT, row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::device_matrix_view<float, IdxT, row_major> distances) RAFT_EXPLICIT;

}  // namespace raft::neighbors::ivf_flat

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_ivf_flat_build(T, IdxT)        \
  extern template auto raft::neighbors::ivf_flat::build<T, IdxT>( \
    raft::resources const& handle,                                \
    const raft::neighbors::ivf_flat::index_params& params,        \
    const T* dataset,                                             \
    IdxT n_rows,                                                  \
    uint32_t dim)                                                 \
    ->raft::neighbors::ivf_flat::index<T, IdxT>;                  \
                                                                  \
  extern template auto raft::neighbors::ivf_flat::build<T, IdxT>( \
    raft::resources const& handle,                                \
    const raft::neighbors::ivf_flat::index_params& params,        \
    raft::device_matrix_view<const T, IdxT, row_major> dataset)   \
    ->raft::neighbors::ivf_flat::index<T, IdxT>;                  \
                                                                  \
  extern template void raft::neighbors::ivf_flat::build<T, IdxT>( \
    raft::resources const& handle,                                \
    const raft::neighbors::ivf_flat::index_params& params,        \
    raft::device_matrix_view<const T, IdxT, row_major> dataset,   \
    raft::neighbors::ivf_flat::index<T, IdxT>& idx);              \
                                                                  \
  extern template auto raft::neighbors::ivf_flat::build<T, IdxT>( \
    raft::resources const& handle,                                \
    const raft::neighbors::ivf_flat::index_params& params,        \
    raft::host_matrix_view<const T, IdxT, row_major> dataset)     \
    ->raft::neighbors::ivf_flat::index<T, IdxT>;                  \
                                                                  \
  extern template void raft::neighbors::ivf_flat::build<T, IdxT>( \
    raft::resources const& handle,                                \
    const raft::neighbors::ivf_flat::index_params& params,        \
    raft::host_matrix_view<const T, IdxT, row_major> dataset,     \
    raft::neighbors::ivf_flat::index<T, IdxT>& idx);

instantiate_raft_neighbors_ivf_flat_build(float, int64_t);
instantiate_raft_neighbors_ivf_flat_build(int8_t, int64_t);
instantiate_raft_neighbors_ivf_flat_build(uint8_t, int64_t);
#undef instantiate_raft_neighbors_ivf_flat_build

#define instantiate_raft_neighbors_ivf_flat_extend(T, IdxT)                \
  extern template auto raft::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    const raft::neighbors::ivf_flat::index<T, IdxT>& orig_index,           \
    const T* new_vectors,                                                  \
    const IdxT* new_indices,                                               \
    IdxT n_rows)                                                           \
    ->raft::neighbors::ivf_flat::index<T, IdxT>;                           \
                                                                           \
  extern template auto raft::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,        \
    std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
    const raft::neighbors::ivf_flat::index<T, IdxT>& orig_index)           \
    ->raft::neighbors::ivf_flat::index<T, IdxT>;                           \
                                                                           \
  extern template void raft::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    raft::neighbors::ivf_flat::index<T, IdxT>* index,                      \
    const T* new_vectors,                                                  \
    const IdxT* new_indices,                                               \
    IdxT n_rows);                                                          \
                                                                           \
  extern template void raft::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,        \
    std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices, \
    raft::neighbors::ivf_flat::index<T, IdxT>* index);                     \
                                                                           \
  extern template void raft::neighbors::ivf_flat::extend<T, IdxT>(         \
    raft::resources const& handle,                                         \
    raft::host_matrix_view<const T, IdxT, row_major> new_vectors,          \
    std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices,   \
    raft::neighbors::ivf_flat::index<T, IdxT>* index);                     \
                                                                           \
  extern template auto raft::neighbors::ivf_flat::extend<T, IdxT>(         \
    const raft::resources& handle,                                         \
    raft::host_matrix_view<const T, IdxT, row_major> new_vectors,          \
    std::optional<raft::host_vector_view<const IdxT, IdxT>> new_indices,   \
    const raft::neighbors::ivf_flat::index<T, IdxT>& idx)                  \
    ->raft::neighbors::ivf_flat::index<T, IdxT>;

instantiate_raft_neighbors_ivf_flat_extend(float, int64_t);
instantiate_raft_neighbors_ivf_flat_extend(int8_t, int64_t);
instantiate_raft_neighbors_ivf_flat_extend(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_flat_extend

#define instantiate_raft_neighbors_ivf_flat_search(T, IdxT)        \
  extern template void raft::neighbors::ivf_flat::search<T, IdxT>( \
    raft::resources const& handle,                                 \
    const raft::neighbors::ivf_flat::search_params& params,        \
    const raft::neighbors::ivf_flat::index<T, IdxT>& index,        \
    const T* queries,                                              \
    uint32_t n_queries,                                            \
    uint32_t k,                                                    \
    IdxT* neighbors,                                               \
    float* distances,                                              \
    rmm::device_async_resource_ref mr);                            \
                                                                   \
  extern template void raft::neighbors::ivf_flat::search<T, IdxT>( \
    raft::resources const& handle,                                 \
    const raft::neighbors::ivf_flat::search_params& params,        \
    const raft::neighbors::ivf_flat::index<T, IdxT>& index,        \
    raft::device_matrix_view<const T, IdxT, row_major> queries,    \
    raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,     \
    raft::device_matrix_view<float, IdxT, row_major> distances);

instantiate_raft_neighbors_ivf_flat_search(float, int64_t);
instantiate_raft_neighbors_ivf_flat_search(int8_t, int64_t);
instantiate_raft_neighbors_ivf_flat_search(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_flat_search
