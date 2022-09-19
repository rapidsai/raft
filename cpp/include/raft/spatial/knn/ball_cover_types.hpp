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

#pragma once

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace spatial {
namespace knn {

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
          typename value_int  = std::uint32_t,
          typename matrix_idx = std::uint32_t>
class BallCoverIndex {
 public:
  explicit BallCoverIndex(const raft::handle_t& handle_,
                          const value_t* X_,
                          value_int m_,
                          value_int n_,
                          raft::distance::DistanceType metric_)
    : handle(handle_),
      X(std::move(raft::make_device_matrix_view<const value_t, matrix_idx>(X_, m_, n_))),
      m(m_),
      n(n_),
      metric(metric_),
      /**
       * the sqrt() here makes the sqrt(m)^2 a linear-time lower bound
       *
       * Total memory footprint of index: (2 * sqrt(m)) + (n * sqrt(m)) + (2 * m)
       */
      n_landmarks(sqrt(m_)),
      R_indptr(std::move(raft::make_device_vector<value_idx, matrix_idx>(handle, sqrt(m_) + 1))),
      R_1nn_cols(std::move(raft::make_device_vector<value_idx, matrix_idx>(handle, m_))),
      R_1nn_dists(std::move(raft::make_device_vector<value_t, matrix_idx>(handle, m_))),
      R_closest_landmark_dists(
        std::move(raft::make_device_vector<value_t, matrix_idx>(handle, m_))),
      R(std::move(raft::make_device_matrix<value_t, matrix_idx>(handle, sqrt(m_), n_))),
      R_radius(std::move(raft::make_device_vector<value_t, matrix_idx>(handle, sqrt(m_)))),
      index_trained(false)
  {
  }

  explicit BallCoverIndex(const raft::handle_t& handle_,
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
      R_indptr(
        std::move(raft::make_device_vector<value_idx, matrix_idx>(handle, sqrt(X_.extent(0)) + 1))),
      R_1nn_cols(std::move(raft::make_device_vector<value_idx, matrix_idx>(handle, X_.extent(0)))),
      R_1nn_dists(std::move(raft::make_device_vector<value_t, matrix_idx>(handle, X_.extent(0)))),
      R_closest_landmark_dists(
        std::move(raft::make_device_vector<value_t, matrix_idx>(handle, X_.extent(0)))),
      R(std::move(
        raft::make_device_matrix<value_t, matrix_idx>(handle, sqrt(X_.extent(0)), X_.extent(1)))),
      R_radius(
        std::move(raft::make_device_vector<value_t, matrix_idx>(handle, sqrt(X_.extent(0))))),
      index_trained(false)
  {
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
  raft::device_matrix_view<const value_t, matrix_idx, row_major> get_X() { return X; }

  bool is_index_trained() const { return index_trained; };

  // This should only be set by internal functions
  void set_index_trained() { index_trained = true; }

  const raft::handle_t& handle;

  const value_int m;
  const value_int n;
  const value_int n_landmarks;

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

 protected:
  bool index_trained;
};
}  // namespace knn
}  // namespace spatial
}  // namespace raft
