/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#ifndef __BALL_COVER_H
#define __BALL_COVER_H

#pragma once

#include <cstdint>

#include "ball_cover_types.hpp"
#include "detail/ball_cover.cuh"
#include "detail/ball_cover/common.cuh"
#include <raft/distance/distance_types.hpp>
#include <thrust/transform.h>

namespace raft {
namespace spatial {
namespace knn {

template <typename value_idx = std::int64_t, typename value_t, typename value_int = std::uint32_t>
void rbc_build_index(const raft::handle_t& handle,
                     BallCoverIndex<value_idx, value_t, value_int>& index)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    detail::rbc_build_index(handle, index, detail::HaversineFunc<value_t, value_int>());
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    detail::rbc_build_index(handle, index, detail::EuclideanFunc<value_t, value_int>());
  } else {
    RAFT_FAIL("Metric not support");
  }

  index.set_index_trained();
}

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
template <typename value_idx = std::int64_t, typename value_t, typename value_int = std::uint32_t>
void rbc_all_knn_query(const raft::handle_t& handle,
                       BallCoverIndex<value_idx, value_t, value_int>& index,
                       value_int k,
                       value_idx* inds,
                       value_t* dists,
                       bool perform_post_filtering = true,
                       float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    detail::rbc_all_knn_query(handle,
                              index,
                              k,
                              inds,
                              dists,
                              detail::HaversineFunc<value_t, value_int>(),
                              perform_post_filtering,
                              weight);
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    detail::rbc_all_knn_query(handle,
                              index,
                              k,
                              inds,
                              dists,
                              detail::EuclideanFunc<value_t, value_int>(),
                              perform_post_filtering,
                              weight);
  } else {
    RAFT_FAIL("Metric not supported");
  }

  index.set_index_trained();
}

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
template <typename value_idx = std::int64_t, typename value_t, typename value_int = std::uint32_t>
void rbc_all_knn_query(const raft::handle_t& handle,
                       BallCoverIndex<value_idx, value_t, value_int>& index,
                       value_int k,
                       raft::device_matrix_view<value_t, value_idx, row_major> inds,
                       raft::device_matrix_view<value_t, value_idx, row_major> dists,
                       bool perform_post_filtering = true,
                       float weight                = 1.0)
{
  RAFT_EXPECTS(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  RAFT_EXPECTS(inds.extent(1) == dists.extent(1) && dists.extent(1) == static_cast<value_idx>(k),
               "Number of columns in output indices and distances matrices must be equal to k");

  RAFT_EXPECTS(inds.extent(0) == dists.extent(0) && dists.extent(0) == index.get_X().extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in index matrix.");

  rbc_all_knn_query(
    handle, index, k, inds.data_handle(), dists.data_handle(), perform_post_filtering, weight);
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
 * @param[in] n_query_pts number of query points
 */
template <typename value_idx = std::int64_t, typename value_t, typename value_int = std::uint32_t>
void rbc_knn_query(const raft::handle_t& handle,
                   BallCoverIndex<value_idx, value_t, value_int>& index,
                   value_int k,
                   const value_t* query,
                   value_int n_query_pts,
                   value_idx* inds,
                   value_t* dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    detail::rbc_knn_query(handle,
                          index,
                          k,
                          query,
                          n_query_pts,
                          inds,
                          dists,
                          detail::HaversineFunc<value_t, value_int>(),
                          perform_post_filtering,
                          weight);
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    detail::rbc_knn_query(handle,
                          index,
                          k,
                          query,
                          n_query_pts,
                          inds,
                          dists,
                          detail::EuclideanFunc<value_t, value_int>(),
                          perform_post_filtering,
                          weight);
  } else {
    RAFT_FAIL("Metric not supported");
  }
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
 * @param[in] n_query_pts number of query points
 */
template <typename value_idx = std::int64_t, typename value_t, typename value_int = std::uint32_t>
void rbc_knn_query(const raft::handle_t& handle,
                   BallCoverIndex<value_idx, value_t, value_int>& index,
                   value_int k,
                   raft::device_matrix_view<const value_t, value_idx, row_major> query,
                   raft::device_matrix_view<value_idx, value_idx, row_major> inds,
                   raft::device_matric_view<value_t, value_idx, row_major> dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  RAFT_EXPECTS(inds.extent(1) == dists.extent(1) && dists.extent(1) == static_cast<value_idx>(k),
               "Number of columns in output indices and distances matrices must be equal to k");

  RAFT_EXPECTS(inds.extent(0) == dists.extent(0) && dists.extent(0) == query.extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in search matrix.");

  RAFT_EXPECTS(query.extent(1) == index.get_R().extent(1),
               "Number of columns in query and index matrices must match.");

  rbc_knn_query(handle,
                index,
                k,
                query.data_handle(),
                query.extent(0),
                inds.data_handle(),
                dists.data_handle(),
                perform_post_filtering,
                weight);
}

// TODO: implement functions for:
//  4. rbc_eps_neigh() - given a populated index, perform query against different query array
//  5. rbc_all_eps_neigh() - populate a BallCoverIndex and query against training data

}  // namespace knn
}  // namespace spatial
}  // namespace raft

#endif