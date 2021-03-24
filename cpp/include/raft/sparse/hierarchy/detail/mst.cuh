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

/**
 * Sorts a COO by its weight
 * @tparam value_idx
 * @tparam value_t
 * @param[inout] rows source edges
 * @param[inout] cols dest edges
 * @param[inout] data edge weights
 * @param[in] nnz number of edges in edge list
 * @param[in] stream cuda stream for which to order cuda operations
 */
template <typename value_idx, typename value_t>
void sort_coo_by_data(value_idx *rows, value_idx *cols, value_t *data,
                      value_idx nnz, cudaStream_t stream) {
  thrust::device_ptr<value_idx> t_rows = thrust::device_pointer_cast(rows);
  thrust::device_ptr<value_idx> t_cols = thrust::device_pointer_cast(cols);
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(data);

  auto first = thrust::make_zip_iterator(thrust::make_tuple(rows, cols));

  thrust::sort_by_key(thrust::cuda::par.on(stream), t_data, t_data + nnz,
                      first);
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
 * @param[in] color the color labels array returned from the mst invocation
 * @return updated MST edge list
 */
template <typename value_idx, typename value_t>
raft::Graph_COO<value_idx, value_idx, value_t> connect_knn_graph(
  const raft::handle_t &handle, const value_t *X,
  raft::Graph_COO<value_idx, value_idx, value_t> &msf, size_t m, size_t n,
  value_idx *color) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  raft::sparse::COO<value_t, value_idx> connected_edges(d_alloc, stream);

  raft::linkage::connect_components<value_idx, value_t>(handle, connected_edges,
                                                        X, color, m, n);

  int final_nnz = connected_edges.nnz + msf.n_edges;

  msf.src.resize(final_nnz, stream);
  msf.dst.resize(final_nnz, stream);
  msf.weights.resize(final_nnz, stream);

  /**
   * Construct final edge list
   */
  raft::copy_async(msf.src.data() + msf.n_edges, connected_edges.rows(),
                   connected_edges.nnz, stream);
  raft::copy_async(msf.dst.data() + msf.n_edges, connected_edges.cols(),
                   connected_edges.nnz, stream);
  raft::copy_async(msf.weights.data() + msf.n_edges, connected_edges.vals(),
                   connected_edges.nnz, stream);

  printf("connected components nnz: %d\n", final_nnz);
  raft::sparse::COO<value_t, value_idx> final_coo(d_alloc, stream);
  raft::sparse::linalg::symmetrize(handle, msf.src.data(), msf.dst.data(),
                                   msf.weights.data(), m, n, final_nnz,
                                   final_coo);

  rmm::device_uvector<value_idx> indptr2(m + 1, stream);

  raft::sparse::convert::sorted_coo_to_csr(final_coo.rows(), final_coo.nnz,
                                           indptr2.data(), m, d_alloc, stream);

  value_idx max_offset = 0;
  raft::update_host(&max_offset, indptr2.data() + (m - 1), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  max_offset += (final_nnz - max_offset);

  raft::update_device(indptr2.data() + m, &max_offset, 1, stream);

  return raft::mst::mst<value_idx, value_idx, value_t>(
    handle, indptr2.data(), final_coo.cols(), final_coo.vals(), m,
    final_coo.nnz, color, stream, false, true);
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
 */
template <typename value_idx, typename value_t>
void build_sorted_mst(const raft::handle_t &handle, const value_t *X,
                      const value_idx *indptr, const value_idx *indices,
                      const value_t *pw_dists, size_t m, size_t n,
                      rmm::device_uvector<value_idx> &mst_src,
                      rmm::device_uvector<value_idx> &mst_dst,
                      rmm::device_uvector<value_t> &mst_weight,
                      const size_t nnz) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  rmm::device_uvector<value_idx> color(m, stream);

  auto mst_coo = raft::mst::mst<value_idx, value_idx, value_t>(
    handle, indptr, indices, pw_dists, (value_idx)m, nnz, color.data(), stream,
    false);

  int iters = 1;
  int n_components = linkage::get_n_components(color.data(), m, stream);
  while (n_components > 1 && iters < 100) {
    printf("Found %d components. Going to try connecting graph\n",
           n_components);
    mst_coo = connect_knn_graph<value_idx, value_t>(handle, X, mst_coo, m, n,
                                                    color.data());

    iters++;

    n_components = linkage::get_n_components(color.data(), m, stream);
    //
    //    printf("Connecting knn graph!\n");
    //
    //    RAFT_EXPECTS(
    //      mst_coo.n_edges == m - 1,
    //      "MST was not able to connect knn graph in a single iteration.");
  }
  printf("Found %d components.\n", n_components);

  sort_coo_by_data(mst_coo.src.data(), mst_coo.dst.data(),
                   mst_coo.weights.data(), mst_coo.n_edges, stream);

  // TODO: be nice if we could pass these directly into the MST
  mst_src.resize(mst_coo.n_edges, stream);
  mst_dst.resize(mst_coo.n_edges, stream);
  mst_weight.resize(mst_coo.n_edges, stream);

  raft::copy_async(mst_src.data(), mst_coo.src.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_dst.data(), mst_coo.dst.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_weight.data(), mst_coo.weights.data(), mst_coo.n_edges,
                   stream);
}

};  // namespace detail
};  // namespace hierarchy
};  // namespace raft