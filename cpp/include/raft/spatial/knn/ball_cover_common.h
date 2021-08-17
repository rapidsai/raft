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

#include <raft/linalg/distance_type.h>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace spatial {
namespace knn {

template <typename value_idx, typename value_t, typename value_int = int>
struct BallCoverIndex {
 public:
  BallCoverIndex(const raft::handle_t &handle_, const value_t *X_, value_int m_,
                 value_int n_, raft::distance::DistanceType metric_)
    : handle(handle_),
      X(X_),
      m(m_),
      n(n_),
      metric(metric_),
      /**
      * the sqrt() here makes the sqrt(m)^2 a linear-time lower bound
      *
      * Total memory footprint of index: (2 * sqrt(m)) + (n * sqrt(m)) + (2 * m)
      */
      n_landmarks(sqrt(m_)),
      R_indptr(sqrt(m_), handle.get_stream()),
      R_1nn_cols(m_, handle.get_stream()),
      R_1nn_dists(m_, handle.get_stream()),
      R(sqrt(m_) * n_, handle.get_stream()),
      R_radius(sqrt(m_), handle.get_stream()),
      index_trained(false) {}

  value_idx *get_R_indptr() { return R_indptr.data(); }
  value_idx *get_R_1nn_cols() { return R_1nn_cols.data(); }
  value_t *get_R_1nn_dists() { return R_1nn_dists.data(); }
  value_t *get_R_radius() { return R_radius.data(); }
  value_t *get_R() { return R.data(); }
  const value_t *get_X() { return X; }

  bool is_index_trained() const { return index_trained; };

  // This should only be set by internal functions
  void set_index_trained() { index_trained = true; }
  const raft::handle_t &handle;

  const value_int m;
  const value_int n;
  const int n_landmarks;

  const value_t *X;

  raft::distance::DistanceType metric;

  // CSR storing the neighborhoods for each data point
  rmm::device_uvector<value_idx> R_indptr;
  rmm::device_uvector<value_idx> R_1nn_cols;
  rmm::device_uvector<value_t> R_1nn_dists;

  rmm::device_uvector<value_t> R_radius;

  rmm::device_uvector<value_t> R;

 protected:
  bool index_trained;
};
}  // namespace knn
}  // namespace spatial
}  // namespace raft
