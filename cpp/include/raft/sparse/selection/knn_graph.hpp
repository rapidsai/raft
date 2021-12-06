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
#include <raft/sparse/coo.hpp>
#include <raft/sparse/selection/detail/knn_graph.cuh>

namespace raft {
namespace sparse {
namespace selection {

/**
 * Constructs a (symmetrized) knn graph edge list from
 * dense input vectors.
 *
 * Note: The resulting KNN graph is not guaranteed to be connected.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[in] X dense matrix of input data samples and observations
 * @param[in] m number of data samples (rows) in X
 * @param[in] n number of observations (columns) in X
 * @param[in] metric distance metric to use when constructing neighborhoods
 * @param[out] out output edge list
 * @param[out] out output edge list
 * @param c
 */
template <typename value_idx = int, typename value_t = float>
void knn_graph(const handle_t& handle,
               const value_t* X,
               size_t m,
               size_t n,
               raft::distance::DistanceType metric,
               raft::sparse::COO<value_t, value_idx>& out,
               int c = 15)
{
  detail::knn_graph(handle, X, m, n, metric, out, c);
}

};  // namespace selection
};  // namespace sparse
};  // end namespace raft