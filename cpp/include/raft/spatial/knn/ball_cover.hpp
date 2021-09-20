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
#include <thrust/transform.h>
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
  }

  index.set_index_trained();
}

    * @tparam value_idx
    * @tparam value_t
    * @tparam value_int
    * @tparam distance_func
    * @param[in] handle raft handle for resource management
    * @param[inout] index previously untrained index
    * @param[in] k number neighbors to return
    * @param[out] inds output indices
    * @param[out] dists output distances
    * @param[in] dfunc
    * @param[in] perform_post_filtering turn off computing distances for
    *            additional landmarks outside of the closest k, if necessary.
    *            This can save a little computation time for approximate
            *            nearest neighbors and will generally return great recall.


/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * performs an all neighbors knn, which can reuse memory when
 * the index and query are the same array. This function will
 * build the index and assumes rbc_build_index() has not already
 * been called.
 * @tparam value_idx knn index type
 * @tparam value_t knn distance type
 * @tparam value_int type for integers, such as number of rows/cols
 * @param handle raft handle for resource management
 * @param index ball cover index which has not yet been built
 * @param k number of nearest neighbors to find
 * @param perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
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

    thrust::transform(handle.get_thrust_policy(), dists, dists + (index.m * k),
                      dists, [] __device__(value_t in) { return sqrt(in); });
  }

  index.set_index_trained();
}

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * function does not build the index and assumes rbc_build_index() has
 * already been called. Use this function when the index and
 * query arrays are different, otherwise use rbc_all_knn_query().
 * @tparam value_idx index type
 * @tparam value_t distances type
 * @tparam value_int integer type for size info
 * @param handle raft handle for resource management
 * @param index ball cover index which has not yet been built
 * @param k number of nearest neighbors to find
 * @param query the
 * @param perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 * @param k
 * @param inds
 * @param dists
 * @param n_samples
 */
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

    thrust::transform(handle.get_thrust_policy(), dists,
                      dists + (n_query_pts * k), dists,
                      [] __device__(value_t in) { return sqrt(in); });
  }
}

// TODO: implement functions for:
//  4. rbc_eps_neigh() - given a populated index, perform query against different query array
//  5. rbc_all_eps_neigh() - populate a BallCoverIndex and query against training data

}  // namespace knn
}  // namespace spatial
}  // namespace raft
