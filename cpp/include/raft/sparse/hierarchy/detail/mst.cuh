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

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace raft {
namespace hierarchy {
namespace detail {

/**
 * Sorts a COO by its weight
 * @tparam value_idx
 * @tparam value_t
 * @param rows
 * @param cols
 * @param data
 * @param nnz
 * @param stream
 */
template <typename value_idx, typename value_t>
void sort_coo_by_data(value_idx *rows, value_idx *cols, value_t *data,
                      value_idx nnz, cudaStream_t stream) {
  thrust::device_ptr<value_idx> t_rows = thrust::device_pointer_cast(rows);
  thrust::device_ptr<value_idx> t_cols = thrust::device_pointer_cast(cols);
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(data);

  auto first = thrust::make_zip_iterator(thrust::make_tuple(t_rows, t_cols));

  thrust::sort_by_key(thrust::cuda::par.on(stream), t_data, t_data + nnz,
                      first);
}

/**
 * Constructs an MST and sorts the resulting edges in ascending
 * order by their weight.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle
 * @param[in] pw_dists
 * @param[in] m
 * @param[out] mst_src
 * @param[out] mst_dst
 * @param[out] mst_weight
 */
template <typename value_idx, typename value_t>
void build_sorted_mst(const raft::handle_t &handle, const value_t *X,
                      const value_idx *indptr, const value_idx *indices,
                      const value_t *pw_dists, size_t m, size_t n,
                      raft::mr::device::buffer<value_idx> &mst_src,
                      raft::mr::device::buffer<value_idx> &mst_dst,
                      raft::mr::device::buffer<value_t> &mst_weight,
                      const size_t nnz) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  raft::mr::device::buffer<value_idx> color(d_alloc, stream, m);

  auto mst_coo = raft::mst::mst<value_idx, value_idx, value_t>(
    handle, indptr, indices, pw_dists, (value_idx)m, nnz, color.data(), stream,
    false);

  // TODO: Pull this into a separate function
  if (linkage::get_n_components(color.data(), m, stream) > 1) {
    raft::sparse::COO<value_t, value_idx> connected_edges(d_alloc, stream);

    raft::linkage::connect_components<value_idx, value_t>(
      handle, connected_edges, X, color.data(), m, n);

    int final_nnz = connected_edges.nnz + mst_coo.n_edges;

    mst_coo.src.resize(final_nnz, stream);
    mst_coo.dst.resize(final_nnz, stream);
    mst_coo.weights.resize(final_nnz, stream);

    /**
     * Construct final edge list
     */
    raft::copy_async(mst_coo.src.data() + mst_coo.n_edges,
                     connected_edges.rows(), connected_edges.nnz, stream);
    raft::copy_async(mst_coo.dst.data() + mst_coo.n_edges,
                     connected_edges.cols(), connected_edges.nnz, stream);
    raft::copy_async(mst_coo.weights.data() + mst_coo.n_edges,
                     connected_edges.vals(), connected_edges.nnz, stream);

    raft::sparse::COO<value_t, value_idx> final_coo(d_alloc, stream);
    raft::sparse::linalg::symmetrize(handle, mst_coo.src.data(),
                                     mst_coo.dst.data(), mst_coo.weights.data(),
                                     m, n, final_nnz, final_coo);

    raft::mr::device::buffer<value_idx> indptr2(d_alloc, stream, m + 1);

    raft::sparse::convert::sorted_coo_to_csr(
      final_coo.rows(), final_coo.nnz, indptr2.data(), m, d_alloc, stream);

    value_idx max_offset = 0;
    raft::update_host(&max_offset, indptr2.data() + (m - 1), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    max_offset += (final_nnz - max_offset);

    raft::update_device(indptr2.data() + m, &max_offset, 1, stream);

    mst_coo = raft::mst::mst<value_idx, value_idx, value_t>(
      handle, indptr2.data(), final_coo.cols(), final_coo.vals(), m,
      final_coo.nnz, color.data(), stream, false, true);

    RAFT_EXPECTS(
      mst_coo.n_edges == m - 1,
      "MST was not able to connect knn graph in a single iteration.");
  }

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