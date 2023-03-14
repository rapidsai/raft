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

#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/spatial/knn/detail/ivf_flat_build.cuh>
#include <raft/spatial/knn/detail/ivf_flat_search.cuh>

#include <raft/core/device_resources.hpp>

#include <raft/core/device_mdspan.hpp>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace raft::neighbors::ivf_flat {

namespace detail = raft::spatial::knn::ivf_flat::detail;

/**
 * @defgroup ivf_flat IVF Flat Algorithm
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
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, dataset, index_params);
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // search K nearest neighbours for each of the N queries
 *   ivf_flat::search(handle, index, queries, out_inds, out_dists, search_params, k);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[in] params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 *
 * @return the constructed ivf-flat index
 */
template <typename T, typename IdxT>
auto build(raft::device_resources const& handle,
           const index_params& params,
           raft::device_matrix_view<const T, IdxT, row_major> dataset) -> index<T, IdxT>
{
  IdxT n_rows = dataset.extent(0);
  IdxT dim    = dataset.extent(1);
  return detail::build(handle, params, dataset.data_handle(), n_rows, dim);
}

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   ivf_flat::extend(handle, index_empty, dataset);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[inout] idx
 * @param[in] new_vectors a device pointer to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device pointer to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 */
template <typename T, typename IdxT>
index<T, IdxT> extend(raft::device_resources const& handle,
                      raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
                      std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
                      index<T, IdxT>& idx)
{
  ASSERT(new_vectors.extent(1) == idx.dim(),
         "new_vectors should have the same dimension as the index");

  IdxT n_rows = new_vectors.extent(0);
  if (new_indices.has_value()) {
    ASSERT(n_rows == new_indices.value().extent(0),
           "new_vectors and new_indices have different number of rows");
  }

  return detail::extend(handle,
                        idx,
                        new_vectors.data_handle(),
                        new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                        n_rows);
}

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, dataset, index_params, dataset);
 *   // fill the index with the data
 *   ivf_flat::extend(handle, index_empty, dataset);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[inout] idx
 * @param[in] new_vectors a device pointer to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device pointer to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `std::nullopt`
 *    here to imply a continuous range `[0...n_rows)`.
 */
template <typename T, typename IdxT>
void extend(raft::device_resources const& handle,
            raft::device_matrix_view<const T, IdxT, row_major> new_vectors,
            std::optional<raft::device_vector_view<const IdxT, IdxT>> new_indices,
            index<T, IdxT>* idx)
{
  ASSERT(new_vectors.extent(1) == idx->dim(),
         "new_vectors should have the same dimension as the index");

  IdxT n_rows = new_vectors.extent(0);
  if (new_indices.has_value()) {
    ASSERT(n_rows == new_indices.value().extent(0),
           "new_vectors and new_indices have different number of rows");
  }

  *idx = detail::extend(handle,
                        *idx,
                        new_vectors.data_handle(),
                        new_indices.has_value() ? new_indices.value().data_handle() : nullptr,
                        n_rows);
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
 *
 * Note, this function requires a temporary buffer to store intermediate results between cuda kernel
 * calls, which may lead to undesirable allocations and slowdown. To alleviate the problem, you can
 * pass a pool memory resource or a large enough pre-allocated memory resource to reduce or
 * eliminate entirely allocations happening within `search`:
 * @code{.cpp}
 *   ...
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, index, queries1, out_inds1, out_dists1, search_params);
 *   ivf_flat::search(handle, index, queries2, out_inds2, out_dists2, search_params);
 *   ivf_flat::search(handle, index, queries3, out_inds3, out_dists3, search_params);
 *   ...
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-flat constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 */
template <typename T, typename IdxT>
void search(raft::device_resources const& handle,
            const search_params& params,
            const index<T, IdxT>& idx,
            raft::device_matrix_view<const T, IdxT, row_major> queries,
            raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
            raft::device_matrix_view<float, IdxT, row_major> distances)
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");

  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  IdxT n_queries = queries.extent(0);
  uint32_t k     = neighbors.extent(1);
  return detail::search(handle,
                        params,
                        idx,
                        queries.data_handle(),
                        n_queries,
                        k,
                        neighbors.data_handle(),
                        distances.data_handle(),
                        handle.get_workspace_resource());
}

/** @} */

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
 *   ivf_flat::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_flat::build(handle, index_params, dataset, N, D);
 *   // use default search parameters
 *   ivf_flat::search_params search_params;
 *   // search K nearest neighbours for each of the N queries
 *   ivf_flat::search(handle, search_params, index, queries, N, K, out_inds, out_dists);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[in] params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 * @param[in] n_rows the number of samples
 * @param[in] dim the dimensionality of the data
 *
 * @return the constructed ivf-flat index
 */
template <typename T, typename IdxT>
auto build(raft::device_resources const& handle,
           const index_params& params,
           const T* dataset,
           IdxT n_rows,
           uint32_t dim) -> index<T, IdxT>
{
  return detail::build(handle, params, dataset, n_rows, dim);
}

/** @} */  // end group ivf_flat

/**
 * @brief Build a new index containing the data of the original plus new extra vectors.
 *
 * Implementation note:
 *    The new data is clustered according to existing kmeans clusters, then the cluster
 *    centers are adjusted to match the newly labeled data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset, N, D);
 *   // fill the index with the data
 *   auto index = ivf_flat::extend(handle, index_empty, dataset, nullptr, N);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] handle
 * @param[in] idx original index
 * @param[in] new_vectors a device pointer to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device pointer to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] n_rows number of rows in `new_vectors`
 *
 * @return the constructed extended ivf-flat index
 */
template <typename T, typename IdxT>
auto extend(raft::device_resources const& handle,
            const index<T, IdxT>& idx,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows) -> index<T, IdxT>
{
  return detail::extend(handle, idx, new_vectors, new_indices, n_rows);
}

/**
 * @brief Extend the index in-place with the new data.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   ivf_flat::index_params index_params;
 *   index_params.add_data_on_build = false;      // don't populate index on build
 *   index_params.kmeans_trainset_fraction = 1.0; // use whole dataset for kmeans training
 *   // train the index from a [N, D] dataset
 *   auto index_empty = ivf_flat::build(handle, index_params, dataset, N, D);
 *   // fill the index with the data
 *   ivf_flat::extend(handle, index_empty, dataset, nullptr, N);
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param handle
 * @param[inout] idx
 * @param[in] new_vectors a device pointer to a row-major matrix [n_rows, index.dim()]
 * @param[in] new_indices a device pointer to a vector of indices [n_rows].
 *    If the original index is empty (`orig_index.size() == 0`), you can pass `nullptr`
 *    here to imply a continuous range `[0...n_rows)`.
 * @param[in] n_rows the number of samples
 */
template <typename T, typename IdxT>
void extend(raft::device_resources const& handle,
            index<T, IdxT>& idx,
            const T* new_vectors,
            const IdxT* new_indices,
            IdxT n_rows)
{
  idx = detail::extend(handle, idx, new_vectors, new_indices, n_rows);
}

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [ivf_flat::build](#ivf_flat::build) documentation for a usage example.
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
 *   ivf_flat::search_params search_params;
 *   // Use the same allocator across multiple searches to reduce the number of
 *   // cuda memory allocations
 *   ivf_flat::search(handle, search_params, index, queries1, N1, K, out_inds1, out_dists1, &mr);
 *   ivf_flat::search(handle, search_params, index, queries2, N2, K, out_inds2, out_dists2, &mr);
 *   ivf_flat::search(handle, search_params, index, queries3, N3, K, out_inds3, out_dists3, &mr);
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
 * @param idx ivf-pq constructed index
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
            const index<T, IdxT>& idx,
            const T* queries,
            uint32_t n_queries,
            uint32_t k,
            IdxT* neighbors,
            float* distances,
            rmm::mr::device_memory_resource* mr = nullptr)
{
  return detail::search(handle, params, idx, queries, n_queries, k, neighbors, distances, mr);
}

}  // namespace raft::neighbors::ivf_flat
