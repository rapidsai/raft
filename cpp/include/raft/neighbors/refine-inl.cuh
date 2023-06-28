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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/neighbors/detail/refine.cuh>
#include <raft/spatial/knn/detail/ann_utils.cuh>

namespace raft::neighbors {

/**
 * @defgroup ann_refine Approximate Nearest Neighbors Refinement
 * @{
 */

/**
 * @brief Refine nearest neighbor search.
 *
 * Refinement is an operation that follows an approximate NN search. The approximate search has
 * already selected n_candidates neighbor candidates for each query. We narrow it down to k
 * neighbors. For each query, we calculate the exact distance between the query and its
 * n_candidates neighbor candidate, and select the k nearest ones.
 *
 * The k nearest neighbors and distances are returned.
 *
 * Example usage
 * @code{.cpp}
 *   using namespace raft::neighbors;
 *   // use default index parameters
 *   ivf_pq::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = ivf_pq::build(handle, index_params, dataset, N, D);
 *   // use default search parameters
 *   ivf_pq::search_params search_params;
 *   // search m = 4 * k nearest neighbours for each of the N queries
 *   ivf_pq::search(handle, search_params, index, queries, N, 4 * k, neighbor_candidates,
 *                  out_dists_tmp);
 *   // refine it to the k nearest one
 *   refine(handle, dataset, queries, neighbor_candidates, out_indices, out_dists,
 *           index.metric());
 * @endcode
 *
 *
 * @param[in] handle the raft handle
 * @param[in] dataset device matrix that stores the dataset [n_rows, dims]
 * @param[in] queries device matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates indices of candidate vectors [n_queries, n_candidates], where
 *   n_candidates >= k
 * @param[out] indices device matrix that stores the refined indices [n_queries, k]
 * @param[out] distances device matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine(raft::resources const& handle,
            raft::device_matrix_view<const data_t, matrix_idx, row_major> dataset,
            raft::device_matrix_view<const data_t, matrix_idx, row_major> queries,
            raft::device_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,
            raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,
            raft::device_matrix_view<distance_t, matrix_idx, row_major> distances,
            distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
{
  detail::refine_device(handle, dataset, queries, neighbor_candidates, indices, distances, metric);
}

/** Same as above, but all input and out data is in host memory.
 * @param[in] handle the raft handle
 * @param[in] dataset host matrix that stores the dataset [n_rows, dims]
 * @param[in] queries host matrix of the queries [n_queris, dims]
 * @param[in] neighbor_candidates host matrix with indices of candidate vectors [n_queries,
 *   n_candidates], where n_candidates >= k
 * @param[out] indices host matrix that stores the refined indices [n_queries, k]
 * @param[out] distances host matrix that stores the refined distances [n_queries, k]
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 */
template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine(raft::resources const& handle,
            raft::host_matrix_view<const data_t, matrix_idx, row_major> dataset,
            raft::host_matrix_view<const data_t, matrix_idx, row_major> queries,
            raft::host_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,
            raft::host_matrix_view<idx_t, matrix_idx, row_major> indices,
            raft::host_matrix_view<distance_t, matrix_idx, row_major> distances,
            distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
{
  detail::refine_host(dataset, queries, neighbor_candidates, indices, distances, metric);
}

/** @} */  // end group ann_refine
}  // namespace raft::neighbors
