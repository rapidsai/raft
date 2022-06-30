/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "detail/ivf_flat_build.cuh"
#include "detail/ivf_flat_search.cuh"
#include "ivf_flat_types.hpp"

#include <raft/core/handle.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

namespace raft::spatial::knn::ivf_flat {

/**
 * @brief Build the index from the dataset for efficient search.
 *
 * @tparam T data element type
 *
 * @param handle
 * @param params configure the index building
 * @param[in] dataset a device pointer to a row-major matrix [n_rows, dim]
 * @param n_rows the number of samples
 * @param dim the dimensionality of the data
 * @param stream
 *
 * @return the constructed ivf-flat index
 */
template <typename T>
inline auto build(const handle_t& handle,
                  const index_params& params,
                  const T* dataset,
                  uint32_t n_rows,
                  uint32_t dim,
                  rmm::cuda_stream_view stream) -> const index<T>
{
  return raft::spatial::knn::detail::ivf_flat::build(handle, params, dataset, n_rows, dim, stream);
}

/**
 * @brief Search ANN using the constructed index.
 *
 * @tparam T data element type
 *
 * @param handle
 * @param params configure the search
 * @param index ivf-flat constructed index
 * @param[in] queries a device pointer to a row-major matrix [n_queries, index->dim()]
 * @param n_queries the batch size
 * @param k the number of neighbors to find for each query.
 * @param[out] neighbors a device pointer to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device pointer to the distances to the selected neighbors [n_queries, k]
 * @param stream
 * @param mr an optional memory resource to use across the searches (you can provide a large enough
 *           memory pool here to avoid memory allocations within search).
 */
template <typename T>
inline void search(const handle_t& handle,
                   const search_params& params,
                   const index<T>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   size_t* neighbors,
                   float* distances,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr = nullptr)
{
  return raft::spatial::knn::detail::ivf_flat::search(
    handle, params, index, queries, n_queries, k, neighbors, distances, stream, mr);
}

}  // namespace raft::spatial::knn::ivf_flat
