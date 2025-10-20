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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/matrix/detail/select_radix.cuh>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/neighbors/detail/knn_brute_force.cuh>

namespace raft::spatial::knn {

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
 * @tparam idx_t
 * @tparam value_t
 * @param in_keys
 * @param in_values
 * @param out_keys
 * @param out_values
 * @param n_samples
 * @param n_parts
 * @param k
 * @param stream
 * @param translations
 */
template <typename idx_t = int64_t, typename value_t = float>
inline void knn_merge_parts(const value_t* in_keys,
                            const idx_t* in_values,
                            value_t* out_keys,
                            idx_t* out_values,
                            size_t n_samples,
                            int n_parts,
                            int k,
                            cudaStream_t stream,
                            idx_t* translations)
{
  raft::neighbors::detail::knn_merge_parts(
    in_keys, in_values, out_keys, out_values, n_samples, n_parts, k, stream, translations);
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
 * @param[in] translations starting offsets for partitions. should be the same size
 *            as input vector.
 */
template <typename idx_t = std::int64_t, typename value_t = float, typename value_int = int>
void brute_force_knn(raft::resources const& handle,
                     std::vector<value_t*>& input,
                     std::vector<value_int>& sizes,
                     value_int D,
                     value_t* search_items,
                     value_int n,
                     idx_t* res_I,
                     value_t* res_D,
                     value_int k,
                     bool rowMajorIndex               = true,
                     bool rowMajorQuery               = true,
                     std::vector<idx_t>* translations = nullptr,
                     distance::DistanceType metric    = distance::DistanceType::L2Unexpanded,
                     float metric_arg                 = 2.0f)
{
  ASSERT(input.size() == sizes.size(), "input and sizes vectors must be the same size");

  raft::neighbors::detail::brute_force_knn_impl(handle,
                                                input,
                                                sizes,
                                                D,
                                                search_items,
                                                n,
                                                res_I,
                                                res_D,
                                                k,
                                                rowMajorIndex,
                                                rowMajorQuery,
                                                translations,
                                                metric,
                                                metric_arg);
}

}  // namespace raft::spatial::knn
