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
#include <raft/cuda_utils.cuh>

#include <raft/sparse/op/sort.h>
#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>
#include <raft/sparse/selection/connect_components.cuh>
#include <rmm/device_uvector.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <rmm/exec_policy.hpp>

namespace raft {
namespace hierarchy {
namespace detail {

template <typename value_idx, typename value_t>
void merge_msts(raft::Graph_COO<value_idx, value_idx, value_t> &coo1,
                raft::Graph_COO<value_idx, value_idx, value_t> &coo2,
                cudaStream_t stream) {
  /** Add edges to existing mst **/
  int final_nnz = coo2.n_edges + coo1.n_edges;

  coo1.src.resize(final_nnz, stream);
  coo1.dst.resize(final_nnz, stream);
  coo1.weights.resize(final_nnz, stream);

  /**
   * Construct final edge list
   */
  raft::copy_async(coo1.src.data() + coo1.n_edges, coo2.src.data(),
                   coo2.n_edges, stream);
  raft::copy_async(coo1.dst.data() + coo1.n_edges, coo2.dst.data(),
                   coo2.n_edges, stream);
  raft::copy_async(coo1.weights.data() + coo1.n_edges, coo2.weights.data(),
                   coo2.n_edges, stream);

  coo1.n_edges = final_nnz;
}

/**
 * Connect an unconnected knn graph (one in which mst returns an msf). The
 * device buffers underlying the Graph_COO object are modified in-place.
 * @tparam value_idx index type
 * @tparam value_t floating-point value type
 * @param[in] handle raft handle
 * @param[in] X original dense data from which knn grpah was constructed
 * @param[inout] msf edge list containing the mst result
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[inout] color the color labels array returned from the mst invocation
 * @return updated MST edge list
 */
template <typename value_idx, typename value_t, typename red_op>
void connect_knn_graph(const raft::handle_t &handle, const value_t *X,
                       raft::Graph_COO<value_idx, value_idx, value_t> &msf,
                       size_t m, size_t n, value_idx *color,
                       red_op reduction_op,
                       raft::distance::DistanceType metric =
                         raft::distance::DistanceType::L2SqrtExpanded) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  raft::sparse::COO<value_t, value_idx> connected_edges(d_alloc, stream);

  raft::linkage::connect_components<value_idx, value_t>(
    handle, connected_edges, X, color, m, n, reduction_op);

  rmm::device_uvector<value_idx> indptr2(m + 1, stream);
  raft::sparse::convert::sorted_coo_to_csr(connected_edges.rows(),
                                           connected_edges.nnz, indptr2.data(),
                                           m + 1, d_alloc, stream);

  // On the second call, we hand the MST the original colors
  // and the new set of edges and let it restart the optimization process
  auto new_mst = raft::mst::mst<value_idx, value_idx, value_t, double>(
    handle, indptr2.data(), connected_edges.cols(), connected_edges.vals(), m,
    connected_edges.nnz, color, stream, false, false);

  merge_msts<value_idx, value_t>(msf, new_mst, stream);
}

/**
 * Constructs an MST and sorts the resulting edges in ascending
 * order by their weight.
 *
 * Hierarchical clustering heavily relies upon the ordering
 * and vertices returned in the MST. If the result of the
 * MST was actually a minimum-spanning forest, the CSR
 * being passed into the MST is not connected. In such a
 * case, this graph will be connected by performing a
 * KNN across the components.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[in] indptr CSR indptr of connectivities graph
 * @param[in] indices CSR indices array of connectivities graph
 * @param[in] pw_dists CSR weights array of connectivities graph
 * @param[in] m number of rows in X / src vertices in connectivities graph
 * @param[in] n number of columns in X
 * @param[out] mst_src output src edges
 * @param[out] mst_dst output dst edges
 * @param[out] mst_weight output weights (distances)
 * @param[in] max_iter maximum iterations to run knn graph connection. This
 *  argument is really just a safeguard against the potential for infinite loops.
 */
template <typename value_idx, typename value_t, typename red_op>
void build_sorted_mst(const raft::handle_t &handle, const value_t *X,
                      const value_idx *indptr, const value_idx *indices,
                      const value_t *pw_dists, size_t m, size_t n,
                      value_idx *mst_src, value_idx *mst_dst,
                      value_t *mst_weight, value_idx *color, size_t nnz,
                      red_op reduction_op,
                      raft::distance::DistanceType metric =
                        raft::distance::DistanceType::L2SqrtExpanded,
                      int max_iter = 10) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  // We want to have MST initialize colors on first call.
  auto mst_coo = raft::mst::mst<value_idx, value_idx, value_t, double>(
    handle, indptr, indices, pw_dists, (value_idx)m, nnz, color, stream, false,
    true);

  int iters = 1;
  int n_components = linkage::get_n_components(color, m, d_alloc, stream);

  while (n_components > 1 && iters < max_iter) {
    connect_knn_graph<value_idx, value_t>(handle, X, mst_coo, m, n, color,
                                          reduction_op);

    iters++;

    n_components = linkage::get_n_components(color, m, d_alloc, stream);
  }

  /**
   * The `max_iter` argument was introduced only to prevent the potential for an infinite loop.
   * Ideally the log2(n) guarantees of the MST should be enough to connect KNN graphs with a
   * massive number of data samples in very few iterations. If it does not, there are 3 likely
   * reasons why (in order of their likelihood):
   * 1. There is a bug in this code somewhere
   * 2. Either the given KNN graph wasn't generated from X or the same metric is not being used
   *    to generate the 1-nn (currently only L2SqrtExpanded is supported).
   * 3. max_iter was not large enough to connect the graph (less likely).
   *
   * Note that a KNN graph generated from 50 random isotropic balls (with significant overlap)
   * was able to be connected in a single iteration.
   */
  RAFT_EXPECTS(n_components == 1,
               "KNN graph could not be connected in %d iterations. "
               "Please verify that the input knn graph is generated from X "
               "(and the same distance metric used),"
               " or increase 'max_iter'",
               max_iter);

  raft::sparse::op::coo_sort_by_weight(mst_coo.src.data(), mst_coo.dst.data(),
                                       mst_coo.weights.data(), mst_coo.n_edges,
                                       stream);

  raft::copy_async(mst_src, mst_coo.src.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_dst, mst_coo.dst.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_weight, mst_coo.weights.data(), mst_coo.n_edges, stream);
}

};  // namespace detail
};  // namespace hierarchy
};  // namespace raft
