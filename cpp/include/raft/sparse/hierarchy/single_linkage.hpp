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

#include <raft/cudart_utils.h>
#include <raft/mr/device/buffer.hpp>

#include <raft/sparse/hierarchy/common.h>
#include <raft/sparse/hierarchy/detail/agglomerative.cuh>
#include <raft/sparse/hierarchy/detail/connectivities.cuh>
#include <raft/sparse/hierarchy/detail/mst.cuh>

namespace raft {
namespace hierarchy {

static const size_t EMPTY = 0;
template <typename value_idx, typename value_t,
          LinkageDistance dist_type = LinkageDistance::PAIRWISE>
void single_linkage(const raft::handle_t &handle, const value_t *X, size_t m,
                    size_t n, raft::distance::DistanceType metric,
                    linkage_output<value_idx, value_t> *out, int c,
                    int n_clusters) {
  ASSERT(n_clusters <= m,
         "n_clusters must be less than or equal to the number of data points");

  auto stream = handle.get_stream();
  auto d_alloc = handle.get_device_allocator();

  raft::mr::device::buffer<value_idx> indptr(d_alloc, stream, EMPTY);
  raft::mr::device::buffer<value_idx> indices(d_alloc, stream, EMPTY);
  raft::mr::device::buffer<value_t> pw_dists(d_alloc, stream, EMPTY);

  /**
   * 1. Construct distance graph
   */
  detail::get_distance_graph<value_idx, value_t, dist_type>(
    handle, X, m, n, metric, indptr, indices, pw_dists, c);

  raft::mr::device::buffer<value_idx> mst_rows(d_alloc, stream, EMPTY);
  raft::mr::device::buffer<value_idx> mst_cols(d_alloc, stream, EMPTY);
  raft::mr::device::buffer<value_t> mst_data(d_alloc, stream, EMPTY);

  /**
   * 2. Construct MST, sorted by weights
   */
  detail::build_sorted_mst<value_idx, value_t>(
    handle, indptr.data(), indices.data(), pw_dists.data(), m, mst_rows,
    mst_cols, mst_data, indices.size());

  pw_dists.release();

  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = mst_rows.size();

  raft::mr::device::buffer<value_idx> children(d_alloc, stream, n_edges * 2);
  raft::mr::device::buffer<value_t> out_delta(d_alloc, stream, n_edges);
  raft::mr::device::buffer<value_idx> out_size(d_alloc, stream, n_edges);

  // Create dendrogram
  detail::build_dendrogram_host<value_idx, value_t>(
    handle, mst_rows.data(), mst_cols.data(), mst_data.data(), n_edges,
    children, out_delta, out_size);

  detail::extract_flattened_clusters(handle, out->labels, children, n_clusters,
                                     m);
}

};  // namespace hierarchy
};  // namespace raft