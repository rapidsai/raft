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

#pragma once

#include <raft/core/device_mdspan.hpp>

namespace raft::cluster::hierarchy {

/**
 * Determines the method for computing the minimum spanning tree (MST)
 */
enum LinkageDistance {

  /**
   * Use a pairwise distance matrix as input to the mst. This
   * is very fast and the best option for fairly small datasets (~50k data points)
   */
  PAIRWISE = 0,

  /**
   * Construct a KNN graph as input to the mst and provide additional
   * edges if the mst does not converge. This is slower but scales
   * to very large datasets.
   */
  KNN_GRAPH = 1
};

};  // end namespace raft::cluster::hierarchy

// The code below is now considered legacy
namespace raft::cluster {

using hierarchy::LinkageDistance;

/**
 * Simple container object for consolidating linkage results. This closely
 * mirrors the trained instance variables populated in
 * Scikit-learn's AgglomerativeClustering estimator.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename idx_t>
class linkage_output {
 public:
  idx_t m;
  idx_t n_clusters;

  idx_t n_leaves;
  idx_t n_connected_components;

  // TODO: These will be made private in a future release
  idx_t* labels;    // size: m
  idx_t* children;  // size: (m-1, 2)

  raft::device_vector_view<idx_t> get_labels()
  {
    return raft::make_device_vector_view<idx_t>(labels, m);
  }

  raft::device_matrix_view<idx_t> get_children()
  {
    return raft::make_device_matrix_view<idx_t>(children, m - 1, 2);
  }
};

class linkage_output_int : public linkage_output<int> {};
class linkage_output_int64 : public linkage_output<int64_t> {};

};  // namespace raft::cluster
