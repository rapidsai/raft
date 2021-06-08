/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include "common.h"
#include "detail/brute_force_knn.cuh"
#include "detail/ivf_pq_ann.cuh"

#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/GpuIndex.h>

#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace spatial {
namespace knn {

using deviceAllocator = raft::mr::device::allocator;


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
void approx_knn_build_index(raft::handle_t &handle, knnIndex *index,
                            knnIndexParam *params,
                            raft::distance::DistanceType metric,
                            float metricArg, float *index_array, IntType n,
                            IntType D) {
  detail::approx_knn_build_index(handle, index, params, metric,
                                 metricArg, index_array, n, D);
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
void approx_knn_search(raft::handle_t &handle, float *distances,
                       int64_t *indices, knnIndex *index, IntType k,
                       float *query_array, IntType n) {

  detail::approx_knn_search(handle, distances, indices, index, k,
                           query_array, n);

}

/**
 * @brief Flat C++ API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param[in] handle the cuml handle to use
 * @param[in] input vector of pointers to the input arrays
 * @param[in] sizes vector of sizes of input arrays
 * @param[in] D the dimensionality of the arrays
 * @param[in] search_items array of items to search of dimensionality D
 * @param[in] n number of rows in search_items
 * @param[out] res_I the resulting index array of size n * k
 * @param[out] res_D the resulting distance array of size n * k
 * @param[in] k the number of nearest neighbors to return
 * @param[in] rowMajorIndex are the index arrays in row-major order?
 * @param[in] rowMajorQuery are the query arrays in row-major order?
 * @param[in] metric distance metric to use. Euclidean (L2) is used by
 * 			   default
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances. This
 * 					 is ignored if the metric_type is not Minkowski.
 * @param[in] expanded should lp-based distances be returned in their expanded
 * 					 form (e.g., without raising to the 1/p power).
 */
inline void brute_force_knn(
  raft::handle_t const &handle, std::vector<float *> &input,
  std::vector<int> &sizes, int D, float *search_items, int n, int64_t *res_I,
  float *res_D, int k, bool rowMajorIndex = true, bool rowMajorQuery = true,
  std::vector<int64_t> *translations = nullptr,
  distance::DistanceType metric = distance::DistanceType::L2Unexpanded,
  float metric_arg = 2.0f) {
  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors must be the same size");

  std::vector<cudaStream_t> int_streams = handle.get_internal_streams();

  detail::brute_force_knn_impl(input, sizes, D, search_items, n, res_I, res_D,
                               k, handle.get_device_allocator(),
                               handle.get_stream(), int_streams.data(),
                               handle.get_num_internal_streams(), rowMajorIndex,
                               rowMajorQuery, translations, metric, metric_arg);
}

}  // namespace knn
}  // namespace spatial
}  // namespace raft
