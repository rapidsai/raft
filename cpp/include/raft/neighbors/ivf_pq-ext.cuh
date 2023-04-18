/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cstdint>  // int64_t
#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/util/raft_explicit.hpp>  // RAFT_EXPLICIT
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#ifdef RAFT_EXPLICIT_INSTANTIATE

namespace raft::neighbors::ivf_pq {

/**
 * @defgroup ivf_pq IVF PQ Algorithm
 * @{
 */

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[in] params configure the index building
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-pq index
 */
template <typename T, typename IdxT = uint32_t>
index<IdxT> build(raft::device_resources const& handle,
                  const index_params& params,
                  raft::device_matrix_view<const T, IdxT, row_major> dataset) RAFT_EXPLICIT;

/**
 * @brief Extend the index with the new data.
 * *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
template <typename T, typename IdxT>
index<IdxT> extend(raft::device_resources const& handle,
                   raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
                   std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,
                   const index<IdxT>& idx) RAFT_EXPLICIT;

/**
 * @brief Extend the index with the new data.
 * *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device vector view to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[inout] idx
 */
template <typename T, typename IdxT>
void extend(raft::device_resources const& handle,
            raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,
            index<IdxT>* idx) RAFT_EXPLICIT;

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`.
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */
template <typename T, typename IdxT>
void search(raft::device_resources const& handle,
            const search_params& params,
            const index<IdxT>& idx,
            raft::device_matrix_view<const T, IdxT, row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::device_matrix_view<float, IdxT, row_major> distances) RAFT_EXPLICIT;

/** @} */  // end group ivf_pq

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * NB: Currently, the following distance metrics are supported:
 * - L2Expanded
 * - L2Unexpanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset, N, D);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search K nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, N, K, out_inds, out_dists);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[in] params configure the index building
 * @param[in] dataset a device/host pointer to a row-major matrix [n_rows, dim]
 * @param[in] n_rows the number of samples
 * @param[in] dim the dimensionality of the data
 *
 * @return the constructed ivf-pq index
 */
template <typename T, typename IdxT = uint32_t>
auto build(raft::device_resources const& handle,
           const index_params& params,
           const T* dataset,
           IdxT n_rows,
           uint32_t dim) -> index<IdxT> RAFT_EXPLICIT;

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are unchanged.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   ivf_pq::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_pq::build(handle, index_params, dataset, N, D);
 *   // fill the index with the data
 *   auto index = ivf_pq::extend(handle, index_empty, dataset, nullptr, N);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[inout] idx original index
 * @param[in] new_vectors a device/host pointer to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device/host pointer to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] n_rows the number of samples
 *
 * @return the constructed extended ivf-pq index
 */
template <typename T, typename IdxT>
auto extend(raft::device_resources const& handle,
            const index<IdxT>& idx,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<IdxT> RAFT_EXPLICIT;

/**
 * @brief Extend the index with the new data.
 * *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[inout] idx
 * @param[in] new_vectors a device/host pointer to a row-major matrix [n_rows, idx.dim()]
 * @param[in] new_indices a device/host pointer to a vector of indices [n_rows].
 *    If the original index is empty (`idx.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] n_rows the number of samples
 */
template <typename T, typename IdxT>
void extend(raft::device_resources const& handle,
            index<IdxT>* idx,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) RAFT_EXPLICIT;

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_pq::build](#ivf_pq::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // Create a pooling memory resource with a pre-defined initial size.
 *   rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> mr(
 *     rmm::mr::get_current_device_resource(), 1024 * 1024);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_pq::search(handle, search_params, index, queries1, N1, K, out_inds1, out_dists1, &mr);
 *   ivf_pq::search(handle, search_params, index, queries2, N2, K, out_inds2, out_dists2, &mr);
 *   ivf_pq::search(handle, search_params, index, queries3, N3, K, out_inds3, out_dists3, &mr);
 *   ...
 * @endcode
 * The exact size of the temporary buffer depends on multiple factors and is an implementation
 * detail. However, you can safely specify a small initial size for the memory pool, so that only a
 * few allocations happen to grow it during the first invocations of the `search`.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-pq constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param[in] n_queries the batch size
 * @param[in] k the number of neighbors to find for each query.
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param[in] mr an optional memory resource to use across the searches (you can provide a large
 * enough memory pool here to avoid memory allocations within search).
 */
template <typename T, typename IdxT>
void search(raft::device_resources const& handle,
            const raft::neighbors::ivf_pq::search_params& params,
            const index<IdxT>& idx,
            const T* queries,
            uint32_t n_queries,
            uint32_t k,
            IdxT* neighbors,
            float* distances,
            rmm::mr::device_memory_resource* mr = nullptr) RAFT_EXPLICIT;

}  // namespace raft::neighbors::ivf_pq

#endif  // RAFT_EXPLICIT_INSTANTIATE

#define instantiate_raft_neighbors_ivf_pq_build(T, IdxT)                                        \
  extern template raft::neighbors::ivf_pq::index<IdxT> raft::neighbors::ivf_pq::build<T, IdxT>( \
    raft::device_resources const& handle,                                                       \
    const raft::neighbors::ivf_pq::index_params& params,                                        \
    raft::device_matrix_view<const T, IdxT, row_major> dataset);                                \
                                                                                                \
  extern template auto raft::neighbors::ivf_pq::build(                                          \
    raft::device_resources const& handle,                                                       \
    const raft::neighbors::ivf_pq::index_params& params,                                        \
    const T* dataset,                                                                           \
    IdxT n_rows,                                                                                \
    uint32_t dim)                                                                               \
    ->raft::neighbors::ivf_pq::index<IdxT>;

instantiate_raft_neighbors_ivf_pq_build(float, int64_t);
instantiate_raft_neighbors_ivf_pq_build(int8_t, int64_t);
instantiate_raft_neighbors_ivf_pq_build(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_build

#define instantiate_raft_neighbors_ivf_pq_extend(T, IdxT)                                        \
  extern template raft::neighbors::ivf_pq::index<IdxT> raft::neighbors::ivf_pq::extend<T, IdxT>( \
    raft::device_resources const& handle,                                                        \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                              \
    std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,            \
    const raft::neighbors::ivf_pq::index<IdxT>& idx);                                            \
                                                                                                 \
  extern template void raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::device_resources const& handle,                                                        \
    raft::device_matrix_view<const T, IdxT, row_major> new_vectors,                              \
    std::optional<raft::device_vector_view<const IdxT, IdxT, row_major>> new_indices,            \
    raft::neighbors::ivf_pq::index<IdxT>* idx);                                                  \
                                                                                                 \
  extern template auto raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::device_resources const& handle,                                                        \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,                                             \
    const T* new_vectors,                                                                        \
    const IdxT* new_indices,                                                                     \
    IdxT n_rows)                                                                                 \
    ->raft::neighbors::ivf_pq::index<IdxT>;                                                      \
                                                                                                 \
  extern template void raft::neighbors::ivf_pq::extend<T, IdxT>(                                 \
    raft::device_resources const& handle,                                                        \
    raft::neighbors::ivf_pq::index<IdxT>* idx,                                                   \
    const T* new_vectors,                                                                        \
    const IdxT* new_indices,                                                                     \
    IdxT n_rows);

instantiate_raft_neighbors_ivf_pq_extend(float, int64_t);
instantiate_raft_neighbors_ivf_pq_extend(int8_t, int64_t);
instantiate_raft_neighbors_ivf_pq_extend(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_extend

#define instantiate_raft_neighbors_ivf_pq_search(T, IdxT)        \
  extern template void raft::neighbors::ivf_pq::search<T, IdxT>( \
    raft::device_resources const& handle,                        \
    const raft::neighbors::ivf_pq::search_params& params,        \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,             \
    raft::device_matrix_view<const T, IdxT, row_major> queries,  \
    raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,   \
    raft::device_matrix_view<float, IdxT, row_major> distances); \
                                                                 \
  extern template void raft::neighbors::ivf_pq::search<T, IdxT>( \
    raft::device_resources const& handle,                        \
    const raft::neighbors::ivf_pq::search_params& params,        \
    const raft::neighbors::ivf_pq::index<IdxT>& idx,             \
    const T* queries,                                            \
    uint32_t n_queries,                                          \
    uint32_t k,                                                  \
    IdxT* neighbors,                                             \
    float* distances,                                            \
    rmm::mr::device_memory_resource* mr)

instantiate_raft_neighbors_ivf_pq_search(float, int64_t);
instantiate_raft_neighbors_ivf_pq_search(int8_t, int64_t);
instantiate_raft_neighbors_ivf_pq_search(uint8_t, int64_t);

#undef instantiate_raft_neighbors_ivf_pq_search
