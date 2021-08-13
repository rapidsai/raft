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

#include "ball_cover_common.h"
#include "detail/ball_cover.cuh"

namespace raft {
namespace spatial {
namespace knn {

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n))
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 * @param handle
 * @param X
 * @param m
 * @param n
 * @param k
 * @param inds
 * @param dists
 * @param n_samples
 */
template <typename value_idx = int64_t, typename value_t,
  typename value_int = int, typename dist_func>
inline void rbc_all_knn_query(const raft::handle_t &handle,
                                  BallCoverIndex<value_idx, value_t> &index
                                  int k,
                                  value_idx *inds,
                                  value_t *dists,
                                  dist_func dfunc,
                                  bool perform_post_filtering = true,
                                  float weight = 1.0) {
  detail::rbc_all_knn_query(handle, index, k, inds, dists,
                                          dfunc, perform_post_filtering, weight);
  index.index_trained = true;
}

// TODO: implement functions for:
//  1. rbc_build_index() - populate a BallCoverIndex
//  2. rbc_knn_query() - given a populated index, perform query against different query array
//  3. rbc_all_knn_query() - populate a BallCoverIndex and query against training data
//  4. rbc_eps_neigh() - given a populated index, perform query against different query array
//  5. rbc_all_eps_neigh() - populate a BallCoverIndex and query against training data

}  // namespace knn
}  // namespace spatial
}  // namespace raft
