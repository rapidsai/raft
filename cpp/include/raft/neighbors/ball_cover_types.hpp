/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>

namespace raft::neighbors::ball_cover {

/**
 * @ingroup random_ball_cover
 * @{
 */

/**
 * Stores raw index data points, sampled landmarks, the 1-nns of index points
 * to their closest landmarks, and the ball radii of each landmark. This
 * class is intended to be constructed once and reused across subsequent
 * queries.
 * @tparam value_idx
 * @tparam value_t
 * @tparam value_int
 */
template <typename value_idx,
          typename value_t,
          typename value_int  = std::int64_t,
          typename matrix_idx = std::int64_t>
class BallCoverIndex {
 public:
  explicit BallCoverIndex(raft::resources const& handle_,
                          const value_t* X_,
                          value_int m_,
                          value_int n_,
                          raft::distance::DistanceType metric_)
    : handle(handle_),
      X(raft::make_device_matrix_view<const value_t, matrix_idx>(X_, m_, n_)),
      m(m_),
      n(n_),
      metric(metric_),
      /**
       * the sqrt() here makes the sqrt(m)^2 a linear-time lower bound
       *
       * Total memory footprint of index: (2 * sqrt(m)) + (n * sqrt(m)) + (2 * m)
       */
      n_landmarks(sqrt(m_)),
      R_indptr(raft::make_device_vector<value_idx, matrix_idx>(handle, sqrt(m_) + 1)),
      R_1nn_cols(raft::make_device_vector<value_idx, matrix_idx>(handle, m_)),
      R_1nn_dists(raft::make_device_vector<value_t, matrix_idx>(handle, m_)),
      R_closest_landmark_dists(raft::make_device_vector<value_t, matrix_idx>(handle, m_)),
      R(raft::make_device_matrix<value_t, matrix_idx>(handle, sqrt(m_), n_)),
      X_reordered(raft::make_device_matrix<value_t, matrix_idx>(handle, m_, n_)),
      R_radius(raft::make_device_vector<value_t, matrix_idx>(handle, sqrt(m_))),
      index_trained(false)
  {
  }

  explicit BallCoverIndex(raft::resources const& handle_,
                          raft::device_matrix_view<const value_t, matrix_idx, row_major> X_,
                          raft::distance::DistanceType metric_)
    : handle(handle_),
      X(X_),
      m(X_.extent(0)),
      n(X_.extent(1)),
      metric(metric_),
      /**
       * the sqrt() here makes the sqrt(m)^2 a linear-time lower bound
       *
       * Total memory footprint of index: (2 * sqrt(m)) + (n * sqrt(m)) + (2 * m)
       */
      n_landmarks(sqrt(X_.extent(0))),
      R_indptr(raft::make_device_vector<value_idx, matrix_idx>(handle, sqrt(X_.extent(0)) + 1)),
      R_1nn_cols(raft::make_device_vector<value_idx, matrix_idx>(handle, X_.extent(0))),
      R_1nn_dists(raft::make_device_vector<value_t, matrix_idx>(handle, X_.extent(0))),
      R_closest_landmark_dists(raft::make_device_vector<value_t, matrix_idx>(handle, X_.extent(0))),
      R(raft::make_device_matrix<value_t, matrix_idx>(handle, sqrt(X_.extent(0)), X_.extent(1))),
      X_reordered(
        raft::make_device_matrix<value_t, matrix_idx>(handle, X_.extent(0), X_.extent(1))),
      R_radius(raft::make_device_vector<value_t, matrix_idx>(handle, sqrt(X_.extent(0)))),
      index_trained(false)
  {
  }

  auto get_R_indptr() const -> raft::device_vector_view<const value_idx, matrix_idx>
  {
    return R_indptr.view();
  }
  auto get_R_1nn_cols() const -> raft::device_vector_view<const value_idx, matrix_idx>
  {
    return R_1nn_cols.view();
  }
  auto get_R_1nn_dists() const -> raft::device_vector_view<const value_t, matrix_idx>
  {
    return R_1nn_dists.view();
  }
  auto get_R_radius() const -> raft::device_vector_view<const value_t, matrix_idx>
  {
    return R_radius.view();
  }
  auto get_R() const -> raft::device_matrix_view<const value_t, matrix_idx, row_major>
  {
    return R.view();
  }
  auto get_R_closest_landmark_dists() const -> raft::device_vector_view<const value_t, matrix_idx>
  {
    return R_closest_landmark_dists.view();
  }
  auto get_X_reordered() const -> raft::device_matrix_view<const value_t, matrix_idx, row_major>
  {
    return X_reordered.view();
  }

  raft::device_vector_view<value_idx, matrix_idx> get_R_indptr() { return R_indptr.view(); }
  raft::device_vector_view<value_idx, matrix_idx> get_R_1nn_cols() { return R_1nn_cols.view(); }
  raft::device_vector_view<value_t, matrix_idx> get_R_1nn_dists() { return R_1nn_dists.view(); }
  raft::device_vector_view<value_t, matrix_idx> get_R_radius() { return R_radius.view(); }
  raft::device_matrix_view<value_t, matrix_idx, row_major> get_R() { return R.view(); }
  raft::device_vector_view<value_t, matrix_idx> get_R_closest_landmark_dists()
  {
    return R_closest_landmark_dists.view();
  }
  raft::device_matrix_view<value_t, matrix_idx, row_major> get_X_reordered()
  {
    return X_reordered.view();
  }
  raft::device_matrix_view<const value_t, matrix_idx, row_major> get_X() const { return X; }

  raft::distance::DistanceType get_metric() const { return metric; }

  value_int get_n_landmarks() const { return n_landmarks; }
  bool is_index_trained() const { return index_trained; };

  // This should only be set by internal functions
  void set_index_trained() { index_trained = true; }

  raft::resources const& handle;

  value_int m;
  value_int n;
  value_int n_landmarks;

  raft::device_matrix_view<const value_t, matrix_idx, row_major> X;

  raft::distance::DistanceType metric;

 private:
  // CSR storing the neighborhoods for each data point
  raft::device_vector<value_idx, matrix_idx> R_indptr;
  raft::device_vector<value_idx, matrix_idx> R_1nn_cols;
  raft::device_vector<value_t, matrix_idx> R_1nn_dists;
  raft::device_vector<value_t, matrix_idx> R_closest_landmark_dists;

  raft::device_vector<value_t, matrix_idx> R_radius;

  raft::device_matrix<value_t, matrix_idx, row_major> R;
  raft::device_matrix<value_t, matrix_idx, row_major> X_reordered;

 protected:
  bool index_trained;
};

/** @} */

}  // namespace raft::neighbors::ball_cover
