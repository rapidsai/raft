/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/spatial/detail/connect_components.cuh>

namespace raft::sparse::spatial {

template <typename value_idx, typename value_t>
using FixConnectivitiesRedOp = detail::FixConnectivitiesRedOp<value_idx, value_t>;

/**
 * Gets the number of unique components from array of
 * colors or labels. This does not assume the components are
 * drawn from a monotonically increasing set.
 * @tparam value_idx
 * @param[in] colors array of components
 * @param[in] n_rows size of components array
 * @param[in] stream cuda stream for which to order cuda operations
 * @return total number of components
 */
template <typename value_idx>
value_idx get_n_components(value_idx* colors, size_t n_rows, cudaStream_t stream)
{
  return detail::get_n_components(colors, n_rows, stream);
}

/**
 * Connects the components of an otherwise unconnected knn graph
 * by computing a 1-nn to neighboring components of each data point
 * (e.g. component(nn) != component(self)) and reducing the results to
 * include the set of smallest destination components for each source
 * component. The result will not necessarily contain
 * n_components^2 - n_components number of elements because many components
 * will likely not be contained in the neighborhoods of 1-nns.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[out] out output edge list containing nearest cross-component
 *             edges.
 * @param[in] X original (row-major) dense matrix for which knn graph should be constructed.
 * @param[in] orig_colors array containing component number for each row of X
 * @param[in] n_rows number of rows in X
 * @param[in] n_cols number of cols in X
 * @param[in] reduction_op
 * @param[in] metric
 */
template <typename value_idx, typename value_t, typename red_op>
void connect_components(
  const raft::handle_t& handle,
  raft::sparse::COO<value_t, value_idx>& out,
  const value_t* X,
  const value_idx* orig_colors,
  size_t n_rows,
  size_t n_cols,
  red_op reduction_op,
  raft::distance::DistanceType metric = raft::distance::DistanceType::L2SqrtExpanded)
{
  detail::connect_components(handle, out, X, orig_colors, n_rows, n_cols, reduction_op, metric);
}

};  // end namespace raft::sparse::spatial