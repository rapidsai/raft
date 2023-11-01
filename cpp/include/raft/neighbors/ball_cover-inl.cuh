/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/ball_cover_types.hpp>
#include <raft/spatial/knn/detail/ball_cover.cuh>
#include <raft/spatial/knn/detail/ball_cover/common.cuh>
#include <thrust/transform.h>

namespace raft::neighbors::ball_cover {

/**
 * @defgroup random_ball_cover Random Ball Cover algorithm
 * @{
 */

/**
 * Builds and populates a previously unbuilt BallCoverIndex
 *
 * Usage example:
 * @code{.cpp}
 *
 *  #include <raft/core/resources.hpp>
 *  #include <raft/neighbors/ball_cover.cuh>
 *  #include <raft/distance/distance_types.hpp>
 *  using namespace raft::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = raft::distance::DistanceType::L2Expanded;
 *  BallCoverIndex index(handle, X, metric);
 *
 *  ball_cover::build_index(handle, index);
 * @endcode
 *
 * @tparam idx_t knn index type
 * @tparam value_t knn value type
 * @tparam int_t integral type for knn params
 * @tparam matrix_idx_t matrix indexing type
 * @param[in] handle library resource management handle
 * @param[inout] index an empty (and not previous built) instance of BallCoverIndex
 */
template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void build_index(raft::resources const& handle,
                 BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    raft::spatial::knn::detail::rbc_build_index(
      handle, index, spatial::knn::detail::HaversineFunc<value_t, int_t>());
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    raft::spatial::knn::detail::rbc_build_index(
      handle, index, spatial::knn::detail::EuclideanFunc<value_t, int_t>());
  } else {
    RAFT_FAIL("Metric not support");
  }

  index.set_index_trained();
}

/** @} */  // end group random_ball_cover

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * performs an all neighbors knn, which can reuse memory when
 * the index and query are the same array. This function will
 * build the index and assumes rbc_build_index() has not already
 * been called.
 * @tparam idx_t knn index type
 * @tparam value_t knn distance type
 * @tparam int_t type for integers, such as number of rows/cols
 * @param[in] handle raft handle for resource management
 * @param[inout] index ball cover index which has not yet been built
 * @param[in] k number of nearest neighbors to find
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 */
template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void all_knn_query(raft::resources const& handle,
                   BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
                   int_t k,
                   idx_t* inds,
                   value_t* dists,
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    raft::spatial::knn::detail::rbc_all_knn_query(
      handle,
      index,
      k,
      inds,
      dists,
      spatial::knn::detail::HaversineFunc<value_t, int_t>(),
      perform_post_filtering,
      weight);
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    raft::spatial::knn::detail::rbc_all_knn_query(
      handle,
      index,
      k,
      inds,
      dists,
      spatial::knn::detail::EuclideanFunc<value_t, int_t>(),
      perform_post_filtering,
      weight);
  } else {
    RAFT_FAIL("Metric not supported");
  }

  index.set_index_trained();
}

/**
 * @ingroup random_ball_cover
 * @{
 */

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * performs an all neighbors knn, which can reuse memory when
 * the index and query are the same array. This function will
 * build the index and assumes rbc_build_index() has not already
 * been called.
 *
 * Usage example:
 * @code{.cpp}
 *
 *  #include <raft/core/resources.hpp>
 *  #include <raft/neighbors/ball_cover.cuh>
 *  #include <raft/distance/distance_types.hpp>
 *  using namespace raft::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = raft::distance::DistanceType::L2Expanded;
 *
 *  // Construct a ball cover index
 *  BallCoverIndex index(handle, X, metric);
 *
 *  // Perform all neighbors knn query
 *  ball_cover::all_knn_query(handle, index, inds, dists, k);
 * @endcode
 *
 * @tparam idx_t knn index type
 * @tparam value_t knn distance type
 * @tparam int_t type for integers, such as number of rows/cols
 * @tparam matrix_idx_t matrix indexing type
 *
 * @param[in] handle raft handle for resource management
 * @param[in] index ball cover index which has not yet been built
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] k number of nearest neighbors to find
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 */
template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void all_knn_query(raft::resources const& handle,
                   BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
                   raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,
                   raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,
                   int_t k,
                   bool perform_post_filtering = true,
                   float weight                = 1.0)
{
  RAFT_EXPECTS(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  RAFT_EXPECTS(k <= index.m,
               "k must be less than or equal to the number of data points in the index");
  RAFT_EXPECTS(inds.extent(1) == dists.extent(1) && dists.extent(1) == static_cast<matrix_idx_t>(k),
               "Number of columns in output indices and distances matrices must be equal to k");

  RAFT_EXPECTS(inds.extent(0) == dists.extent(0) && dists.extent(0) == index.get_X().extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in index matrix.");

  all_knn_query(
    handle, index, k, inds.data_handle(), dists.data_handle(), perform_post_filtering, weight);
}

/** @} */

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * function does not build the index and assumes rbc_build_index() has
 * already been called. Use this function when the index and
 * query arrays are different, otherwise use rbc_all_knn_query().
 * @tparam idx_t index type
 * @tparam value_t distances type
 * @tparam int_t integer type for size info
 * @param[in] handle raft handle for resource management
 * @param[inout] index ball cover index which has not yet been built
 * @param[in] k number of nearest neighbors to find
 * @param[in] query the
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 * @param[in] n_query_pts number of query points
 */
template <typename idx_t, typename value_t, typename int_t>
void knn_query(raft::resources const& handle,
               const BallCoverIndex<idx_t, value_t, int_t>& index,
               int_t k,
               const value_t* query,
               int_t n_query_pts,
               idx_t* inds,
               value_t* dists,
               bool perform_post_filtering = true,
               float weight                = 1.0)
{
  ASSERT(index.n <= 3, "only 2d and 3d vectors are supported in current implementation");
  if (index.metric == raft::distance::DistanceType::Haversine) {
    raft::spatial::knn::detail::rbc_knn_query(handle,
                                              index,
                                              k,
                                              query,
                                              n_query_pts,
                                              inds,
                                              dists,
                                              spatial::knn::detail::HaversineFunc<value_t, int_t>(),
                                              perform_post_filtering,
                                              weight);
  } else if (index.metric == raft::distance::DistanceType::L2SqrtExpanded ||
             index.metric == raft::distance::DistanceType::L2SqrtUnexpanded) {
    raft::spatial::knn::detail::rbc_knn_query(handle,
                                              index,
                                              k,
                                              query,
                                              n_query_pts,
                                              inds,
                                              dists,
                                              spatial::knn::detail::EuclideanFunc<value_t, int_t>(),
                                              perform_post_filtering,
                                              weight);
  } else {
    RAFT_FAIL("Metric not supported");
  }
}

/**
 * @ingroup random_ball_cover
 * @{
 */

/**
 * Performs a faster exact knn in metric spaces using the triangle
 * inequality with a number of landmark points to reduce the
 * number of distance computations from O(n^2) to O(sqrt(n)). This
 * function does not build the index and assumes rbc_build_index() has
 * already been called. Use this function when the index and
 * query arrays are different, otherwise use rbc_all_knn_query().
 *
 * Usage example:
 * @code{.cpp}
 *
 *  #include <raft/core/resources.hpp>
 *  #include <raft/neighbors/ball_cover.cuh>
 *  #include <raft/distance/distance_types.hpp>
 *  using namespace raft::neighbors;
 *
 *  raft::resources handle;
 *  ...
 *  auto metric = raft::distance::DistanceType::L2Expanded;
 *
 *  // Build a ball cover index
 *  BallCoverIndex index(handle, X, metric);
 *  ball_cover::build_index(handle, index);
 *
 *  // Perform all neighbors knn query
 *  ball_cover::knn_query(handle, index, inds, dists, k);
 * @endcode

 *
 * @tparam idx_t index type
 * @tparam value_t distances type
 * @tparam int_t integer type for size info
 * @tparam matrix_idx_t
 * @param[in] handle raft handle for resource management
 * @param[in] index ball cover index which has not yet been built
 * @param[in] query device matrix containing query data points
 * @param[out] inds output knn indices
 * @param[out] dists output knn distances
 * @param[in] k number of nearest neighbors to find
 * @param[in] perform_post_filtering if this is false, only the closest k landmarks
 *                               are considered (which will return approximate
 *                               results).
 * @param[in] weight a weight for overlap between the closest landmark and
 *               the radius of other landmarks when pruning distances.
 *               Setting this value below 1 can effectively turn off
 *               computing distances against many other balls, enabling
 *               approximate nearest neighbors. Recall can be adjusted
 *               based on how many relevant balls are ignored. Note that
 *               many datasets can still have great recall even by only
 *               looking in the closest landmark.
 */
template <typename idx_t, typename value_t, typename int_t, typename matrix_idx_t>
void knn_query(raft::resources const& handle,
               const BallCoverIndex<idx_t, value_t, int_t, matrix_idx_t>& index,
               raft::device_matrix_view<const value_t, matrix_idx_t, row_major> query,
               raft::device_matrix_view<idx_t, matrix_idx_t, row_major> inds,
               raft::device_matrix_view<value_t, matrix_idx_t, row_major> dists,
               int_t k,
               bool perform_post_filtering = true,
               float weight                = 1.0)
{
  RAFT_EXPECTS(k <= index.m,
               "k must be less than or equal to the number of data points in the index");
  RAFT_EXPECTS(inds.extent(1) == dists.extent(1) && dists.extent(1) == static_cast<idx_t>(k),
               "Number of columns in output indices and distances matrices must be equal to k");

  RAFT_EXPECTS(inds.extent(0) == dists.extent(0) && dists.extent(0) == query.extent(0),
               "Number of rows in output indices and distances matrices must equal number of rows "
               "in search matrix.");

  RAFT_EXPECTS(query.extent(1) == index.get_X().extent(1),
               "Number of columns in query and index matrices must match.");

  knn_query(handle,
            index,
            k,
            query.data_handle(),
            query.extent(0),
            inds.data_handle(),
            dists.data_handle(),
            perform_post_filtering,
            weight);
}

/** @} */

// TODO: implement functions for:
//  4. rbc_eps_neigh() - given a populated index, perform query against different query array
//  5. rbc_all_eps_neigh() - populate a BallCoverIndex and query against training data

}  // namespace raft::neighbors::ball_cover

#endif
