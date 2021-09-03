/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/linalg/distance_type.h>
#include "ball_cover_common.h"
#include "detail/ball_cover.cuh"
#include "detail/ball_cover/common.cuh"

namespace raft {
namespace spatial {
namespace knn {

template <typename value_idx = int64_t, typename value_t,
          typename value_int = int>
inline void rbc_build_index(const raft::handle_t &handle,
                            BallCoverIndex<value_idx, value_t> &index, int k) {
  ASSERT(index.n == 2,
         "Random ball cover currently only works in 2-dimensions");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    detail::rbc_build_index(handle, index, k, detail::HaversineFunc());
  } else if (index.metric == raft::distance::DistanceType::L2Expanded ||
             index.metric == raft::distance::DistanceType::L2Unexpanded) {
    detail::rbc_build_index(handle, index, k, detail::EuclideanFunc());
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    detail::rbc_build_index(handle, index, k, detail::EuclideanFunc());

    // TODO: Call sqrt on distances
  }

  index.set_index_trained();
}

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
          typename value_int = int>
inline void rbc_all_knn_query(const raft::handle_t &handle,
                              BallCoverIndex<value_idx, value_t> &index, int k,
                              value_idx *inds, value_t *dists,
                              bool perform_post_filtering = true,
                              float weight = 1.0) {
  ASSERT(index.n == 2,
         "Random ball cover currently only works in 2-dimensions");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    detail::rbc_all_knn_query(handle, index, k, inds, dists,
                              detail::HaversineFunc(), perform_post_filtering,
                              weight);
  } else if (index.metric == raft::distance::DistanceType::L2Expanded ||
             index.metric == raft::distance::DistanceType::L2Unexpanded) {
    detail::rbc_all_knn_query(handle, index, k, inds, dists,
                              detail::EuclideanFunc(), perform_post_filtering,
                              weight);
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    detail::rbc_all_knn_query(handle, index, k, inds, dists,
                              detail::EuclideanFunc(), perform_post_filtering,
                              weight);

    // TODO: Call sqrt on distances
  }

  index.set_index_trained();
}

template <typename value_idx = int64_t, typename value_t,
          typename value_int = int>
inline void rbc_knn_query(const raft::handle_t &handle,
                          BallCoverIndex<value_idx, value_t> &index, int k,
                          const value_t *query, value_int n_query_pts,
                          value_idx *inds, value_t *dists,
                          bool perform_post_filtering = true,
                          float weight = 1.0) {
  ASSERT(index.n == 2,
         "Random ball cover currently only works in 2-dimensions");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    detail::rbc_knn_query(handle, index, k, query, n_query_pts, inds, dists,
                          detail::HaversineFunc(), perform_post_filtering,
                          weight);
  } else if (index.metric == raft::distance::DistanceType::L2Expanded ||
             index.metric == raft::distance::DistanceType::L2Unexpanded) {
    detail::rbc_knn_query(handle, index, k, query, n_query_pts, inds, dists,
                          detail::EuclideanFunc(), perform_post_filtering,
                          weight);
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    detail::rbc_knn_query(handle, index, k, query, n_query_pts, inds, dists,
                          detail::EuclideanFunc(), perform_post_filtering,
                          weight);

    // TODO: Call sqrt on distances
  }
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
