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

#include <raft/neighbors/detail/ivf_pq_build.cuh>
#include <raft/neighbors/detail/ivf_pq_search.cuh>
#include <raft/neighbors/ivf_pq_serialize.cuh>
#include <raft/neighbors/ivf_pq_types.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

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
 * @param handle
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 * @param params configure the index building
 *
 * @return the constructed ivf-pq index
 */
template <typename T, typename IdxT = uint32_t>
auto build(raft::device_resources const& handle,
           raft::device_matrix_view<const T, IdxT, row_major> dataset,
           const index_params& params) -> index<IdxT>
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  return detail::build(handle, params, dataset.data_handle(), n_rows, dim);
}

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
 * @param handle
 * @param[inout] index
 * @param[in] dataset a device matrix view to a row-major matrix [n_rows, dim]
 * @param params configure the index building
 *
 * @return the constructed ivf-pq index
 */
template <typename T, typename IdxT = uint32_t>
void build(raft::device_resources const& handle,
           index<IdxT>* index,
           raft::device_matrix_view<const T, IdxT, row_major> dataset,
           const index_params& params)
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  *index      = detail::build(handle, params, dataset.data_handle(), n_rows, dim);
}

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are unchanged.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param handle
 * @param orig_index original index
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device matrix view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 *
 * @return the constructed extended ivf-pq index
 */
template <typename T, typename IdxT>
auto extend(raft::device_resources const& handle,
            const index<IdxT>& orig_index,
            raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::device_matrix_view<const IdxT, IdxT, row_major>> new_indices =
              std::nullopt) -> index<IdxT>
{
  ASSERT(new_vectors.extent(1) == orig_index.dim(),
         "new_vectors should have the same dimension as the index");

  IdxT n_rows = new_vectors.extent(0);
  if (new_indices.has_value()) {
    ASSERT(n_rows == new_indices.value().extent(0),
           "new_vectors and new_indices have different number of rows");
  }

  return detail::extend(handle,
                        orig_index,
                        new_vectors.data_handle(),
                        new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                        n_rows);
}

/**
 * @brief Extend the index with the new data.
 * *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param handle
 * @param[inout] index
 * @param[in] new_vectors a device matrix view to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device matrix view to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 */
template <typename T, typename IdxT>
void extend(
  raft::device_resources const& handle,
  index<IdxT>* index,
  raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
  std::optional<raft::device_matrix_view<const IdxT, IdxT, row_major>> new_indices = std::nullopt)
{
  ASSERT(new_vectors.extent(1) == index->dim(),
         "new_vectors should have the same dimension as the index");

  IdxT n_rows = new_vectors.extent(0);
  if (new_indices.has_value()) {
    ASSERT(n_rows == new_indices.value().extent(0),
           "new_vectors and new_indices have different number of rows");
  }

  *index = extend(handle,
                  *index,
                  new_vectors.data_handle(),
                  new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                  n_rows);
}

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
 * @param handle
 * @param index ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param params configure the search
 */
template <typename T, typename IdxT>
void search(raft::device_resources const& handle,
            const index<IdxT>& index,
            raft::device_matrix_view<const T, IdxT, row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::device_matrix_view<float, IdxT, row_major> distances,
            const search_params& params)
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");

  RAFT_EXPECTS(queries.extent(1) == index.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  std::uint32_t k = neighbors.extent(1);
  return detail::search(handle,
                        params,
                        index,
                        queries.data_handle(),
                        static_cast<std::uint32_t>(queries.extent(0)),
                        k,
                        neighbors.data_handle(),
                        distances.data_handle(),
                        handle.get_workspace_resource());
}

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
 * @param handle
 * @param params configure the index building
 * @param[in] dataset a device/host pointer to a row-major matrix [n_rows, dim]
 * @param n_rows the number of samples
 * @param dim the dimensionality of the data
 *
 * @return the constructed ivf-pq index
 */
template <typename T, typename IdxT = uint32_t>
auto build(raft::device_resources const& handle,
           const index_params& params,
           const T* dataset,
           IdxT n_rows,
           uint32_t dim) -> index<IdxT>
{
  return detail::build(handle, params, dataset, n_rows, dim);
}

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
 * @param handle
 * @param orig_index original index
 * @param[in] new_vectors a device/host pointer to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device/host pointer to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param n_rows the number of samples
 *
 * @return the constructed extended ivf-pq index
 */
template <typename T, typename IdxT>
auto extend(raft::device_resources const& handle,
            const index<IdxT>& orig_index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<IdxT>
{
  return detail::extend(handle, orig_index, new_vectors, new_indices, n_rows);
}

/**
 * @brief Extend the index with the new data.
 * *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param handle
 * @param[inout] index
 * @param[in] new_vectors a device/host pointer to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device/host pointer to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param n_rows the number of samples
 */
template <typename T, typename IdxT>
void extend(raft::device_resources const& handle,
            index<IdxT>* index,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows)
{
  detail::extend(handle, index, new_vectors, new_indices, n_rows);
}

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
 * @param handle
 * @param params configure the search
 * @param index ivf-pq constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param n_queries the batch size
 * @param k the number of neighbors to find for each query.
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param mr an optional memory resource to use across the searches (you can provide a large enough
 *           memory pool here to avoid memory allocations within search).
 */
template <typename T, typename IdxT>
void search(raft::device_resources const& handle,
            const search_params& params,
            const index<IdxT>& index,
            const T* queries,
            uint32_t n_queries,
            uint32_t k,
            IdxT* neighbors,
            float* distances,
            rmm::mr::device_memory_resource* mr = nullptr)
{
  return detail::search(handle, params, index, queries, n_queries, k, neighbors, distances, mr);
}

}  // namespace raft::neighbors::ivf_pq
