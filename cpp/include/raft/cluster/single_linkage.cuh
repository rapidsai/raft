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

#include <raft/cluster/detail/single_linkage.cuh>
#include <raft/cluster/single_linkage_types.hpp>
#include <raft/core/device_mdspan.hpp>

namespace raft::cluster {

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
template <typename value_idx,
          typename value_t,
          LinkageDistance dist_type = LinkageDistance::KNN_GRAPH>
void single_linkage(const raft::handle_t& handle,
                    const value_t* X,
                    size_t m,
                    size_t n,
                    raft::distance::DistanceType metric,
                    linkage_output<value_idx>* out,
                    int c,
                    size_t n_clusters)
{
  detail::single_linkage<value_idx, value_t, dist_type>(
    handle, X, m, n, metric, out, c, n_clusters);
}

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
 control of k. The algorithm will set `k = log(n) + c`
 * @param[in] n_clusters number of clusters to assign data samples
 */
template <typename value_t, typename idx_t, LinkageDistance dist_type = LinkageDistance::KNN_GRAPH>
void single_linkage(const raft::handle_t& handle,
                    raft::device_matrix_view<const value_t, idx_t, row_major> X,
                    raft::distance::DistanceType metric,
                    linkage_output<idx_t>& out,
                    int c,
                    size_t n_clusters)
{
  detail::single_linkage<idx_t, value_t, dist_type>(handle,
                                                    X.data_handle(),
                                                    static_cast<std::size_t>(X.extent(0)),
                                                    static_cast<std::size_t>(X.extent(1)),
                                                    metric,
                                                    &out,
                                                    c,
                                                    n_clusters);
}

};  // namespace raft::cluster
