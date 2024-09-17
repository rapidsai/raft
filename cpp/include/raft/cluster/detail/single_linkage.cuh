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

#include <raft/cluster/detail/agglomerative.cuh>
#include <raft/cluster/detail/connectivities.cuh>
#include <raft/cluster/detail/mst.cuh>
#include <raft/cluster/single_linkage_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

namespace raft::cluster::detail {

static const size_t EMPTY = 0;

/**
 * Single-linkage clustering, capable of constructing a KNN graph to
 * scale the algorithm beyond the n^2 memory consumption of implementations
 * that use the fully-connected graph of pairwise distances by connecting
 * a knn graph when k is not large enough to connect it.

 * @tparam value_idx
 * @tparam value_t
 * @tparam dist_type method to use for constructing connectivities graph
 * @param[in] handle raft handle
 * @param[in] X dense input matrix in row-major layout
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metrix to use when constructing connectivities graph
 * @param[out] out struct containing output dendrogram and cluster assignments
 * @param[in] c a constant used when constructing connectivities from knn graph. Allows the indirect
 control
 *            of k. The algorithm will set `k = log(n) + c`
 * @param[in] n_clusters number of clusters to assign data samples
 */
template <typename value_idx, typename value_t, LinkageDistance dist_type>
void single_linkage(raft::resources const& handle,
                    const value_t* X,
                    size_t m,
                    size_t n,
                    raft::distance::DistanceType metric,
                    linkage_output<value_idx>* out,
                    int c,
                    size_t n_clusters)
{
  ASSERT(n_clusters <= m, "n_clusters must be less than or equal to the number of data points");

  auto stream = resource::get_cuda_stream(handle);

  rmm::device_uvector<value_idx> indptr(EMPTY, stream);
  rmm::device_uvector<value_idx> indices(EMPTY, stream);
  rmm::device_uvector<value_t> pw_dists(EMPTY, stream);

  /**
   * 1. Construct distance graph
   */
  detail::get_distance_graph<value_idx, value_t, dist_type>(
    handle, X, m, n, metric, indptr, indices, pw_dists, c);

  rmm::device_uvector<value_idx> mst_rows(m - 1, stream);
  rmm::device_uvector<value_idx> mst_cols(m - 1, stream);
  rmm::device_uvector<value_t> mst_data(m - 1, stream);

  /**
   * 2. Construct MST, sorted by weights
   */
  rmm::device_uvector<value_idx> color(m, stream);
  raft::sparse::neighbors::FixConnectivitiesRedOp<value_idx, value_t> op(m);
  detail::build_sorted_mst<value_idx, value_t>(handle,
                                               X,
                                               indptr.data(),
                                               indices.data(),
                                               pw_dists.data(),
                                               m,
                                               n,
                                               mst_rows.data(),
                                               mst_cols.data(),
                                               mst_data.data(),
                                               color.data(),
                                               indices.size(),
                                               op,
                                               metric);

  pw_dists.release();

  /**
   * Perform hierarchical labeling
   */
  size_t n_edges = mst_rows.size();

  rmm::device_uvector<value_t> out_delta(n_edges, stream);
  rmm::device_uvector<value_idx> out_size(n_edges, stream);
  // Create dendrogram
  detail::build_dendrogram_host<value_idx, value_t>(handle,
                                                    mst_rows.data(),
                                                    mst_cols.data(),
                                                    mst_data.data(),
                                                    n_edges,
                                                    out->children,
                                                    out_delta.data(),
                                                    out_size.data());
  detail::extract_flattened_clusters(handle, out->labels, out->children, n_clusters, m);

  out->m                      = m;
  out->n_clusters             = n_clusters;
  out->n_leaves               = m;
  out->n_connected_components = 1;
}
};  // namespace raft::cluster::detail