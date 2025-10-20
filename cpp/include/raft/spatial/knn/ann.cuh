/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include "ann_common.h"
#include "detail/ann_quantized.cuh"

#include <raft/core/nvtx.hpp>

namespace raft::spatial::knn {

/**
 * @brief Flat C++ API function to build an approximate nearest neighbors index
 * from an index array and a set of parameters.
 *
 * @param[in] handle RAFT handle
 * @param[out] index index to be built
 * @param[in] params parametrization of the index to be built
 * @param[in] metric distance metric to use. Euclidean (L2) is used by default
 * @param[in] metricArg metric argument
 * @param[in] index_array the index array to build the index with
 * @param[in] n number of rows in the index array
 * @param[in] D the dimensionality of the index array
 */
template <typename T = float, typename value_idx = int>
[[deprecated("Consider using new-style raft::spatial::knn::*::build functions")]] inline void
approx_knn_build_index(raft::resources& handle,
                       raft::spatial::knn::knnIndex* index,
                       knnIndexParam* params,
                       raft::distance::DistanceType metric,
                       float metricArg,
                       T* index_array,
                       value_idx n,
                       value_idx D)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "legacy approx_knn_build_index(n_rows = %u, dim = %u)", n, D);
  detail::approx_knn_build_index(handle, index, params, metric, metricArg, index_array, n, D);
}

/**
 * @brief Flat C++ API function to perform an approximate nearest neighbors
 * search from previously built index and a query array
 *
 * @param[in] handle RAFT handle
 * @param[out] distances distances of the nearest neighbors toward
 *                       their query point
 * @param[out] indices indices of the nearest neighbors
 * @param[in] index index to perform a search with
 * @param[in] k the number of nearest neighbors to search for
 * @param[in] query_array the query to perform a search with
 * @param[in] n number of rows in the query array
 */
template <typename T = float, typename value_idx = int>
[[deprecated("Consider using new-style raft::spatial::knn::*::search functions")]] inline void
approx_knn_search(raft::resources& handle,
                  float* distances,
                  int64_t* indices,
                  raft::spatial::knn::knnIndex* index,
                  value_idx k,
                  T* query_array,
                  value_idx n)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "legacy approx_knn_search(k = %u, n_queries = %u)", k, n);
  detail::approx_knn_search(handle, distances, indices, index, k, query_array, n);
}

}  // namespace raft::spatial::knn
