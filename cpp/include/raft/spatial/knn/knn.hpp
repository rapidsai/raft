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

#include "detail/knn_brute_force_faiss.cuh"
#include "detail/selection_faiss.cuh"

#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace spatial {
namespace knn {

using deviceAllocator = raft::mr::device::allocator;

/**
 * Performs a k-select across row partitioned index/distance
 * matrices formatted like the following:
 * row1: k0, k1, k2
 * row2: k0, k1, k2
 * row3: k0, k1, k2
 * row1: k0, k1, k2
 * row2: k0, k1, k2
 * row3: k0, k1, k2
 *
 * etc...
 *
 * @tparam value_idx
 * @tparam value_t
 * @param inK
 * @param inV
 * @param outK
 * @param outV
 * @param n_samples
 * @param n_parts
 * @param k
 * @param stream
 * @param translations
 */
template <typename value_idx = int64_t, typename value_t = float>
inline void knn_merge_parts(value_t *inK, value_idx *inV, value_t *outK,
                            value_idx *outV, size_t n_samples, int n_parts,
                            int k, cudaStream_t stream,
                            value_idx *translations) {
  detail::knn_merge_parts(inK, inV, outK, outV, n_samples, n_parts, k, stream,
                          translations);
}

/**
 * Performs a k-select across column-partitioned index/distance
 * matrices formatted like the following:
 * row1: k0, k1, k2, k0, k1, k2
 * row2: k0, k1, k2, k0, k1, k2
 * row3: k0, k1, k2, k0, k1, k2
 *
 * etc...
 *
 * @tparam value_idx
 * @tparam value_t
 * @param inK
 * @param inV
 * @param n_rows
 * @param n_cols
 * @param outK
 * @param outV
 * @param select_min
 * @param k
 * @param stream
 */
template <typename value_idx = int, typename value_t = float>
inline void select_k(value_t *inK, value_idx *inV, size_t n_rows, size_t n_cols,
                     value_t *outK, value_idx *outV, bool select_min, int k,
                     cudaStream_t stream) {
  detail::select_k(inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
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

  detail::brute_force_knn_impl(handle, input, sizes, D, search_items, n, res_I,
                               res_D, k, rowMajorIndex, rowMajorQuery,
                               translations, metric, metric_arg);
}
}  // namespace knn
}  // namespace spatial
}  // namespace raft
