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
#include <raft/handle.hpp>

#include <raft/linalg/unary_op.cuh>

#include <raft/linalg/distance_type.h>
#include <raft/sparse/hierarchy/common.h>
#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/hierarchy/detail/knn_graph.cuh>

#include <limits>

namespace raft {
namespace hierarchy {
namespace detail {

template <raft::hierarchy::LinkageDistance dist_type, typename value_idx,
          typename value_t>
struct distance_graph_impl {
  void run(const raft::handle_t &handle, const value_t *X, size_t m, size_t n,
           raft::distance::DistanceType metric,
           raft::mr::device::buffer<value_idx> &indptr,
           raft::mr::device::buffer<value_idx> &indices,
           raft::mr::device::buffer<value_t> &data, int c);
};

//@TODO: Don't need to expose these
/**
 * Connectivities specialization to build a knn graph
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct distance_graph_impl<LinkageDistance::KNN_GRAPH, value_idx, value_t> {
  void run(const raft::handle_t &handle, const value_t *X, size_t m, size_t n,
           raft::distance::DistanceType metric,
           raft::mr::device::buffer<value_idx> &indptr,
           raft::mr::device::buffer<value_idx> &indices,
           raft::mr::device::buffer<value_t> &data, int c) {
    auto d_alloc = handle.get_device_allocator();
    auto stream = handle.get_stream();

    // Need to symmetrize knn into undirected graph
    raft::sparse::COO<value_t, value_idx> knn_graph_coo(d_alloc, stream);

    knn_graph(handle, X, m, n, metric, knn_graph_coo, c);

    indices.resize(knn_graph_coo.nnz, stream);
    data.resize(knn_graph_coo.nnz, stream);

    raft::sparse::convert::sorted_coo_to_csr(&knn_graph_coo, indptr.data(),
                                             d_alloc, stream);

    //TODO: This is a bug in the coo_to_csr prim
    value_idx max_offset = 0;
    raft::update_host(&max_offset, indptr.data() + (m - 1), 1, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    max_offset += (knn_graph_coo.nnz - max_offset);

    raft::update_device(indptr.data() + m, &max_offset, 1, stream);

    raft::copy_async(indices.data(), knn_graph_coo.cols(), knn_graph_coo.nnz,
                     stream);
    raft::copy_async(data.data(), knn_graph_coo.vals(), knn_graph_coo.nnz,
                     stream);
  }
};

template <typename value_idx, typename value_t,
          hierarchy::LinkageDistance dist_type>
void get_distance_graph(const raft::handle_t &handle, const value_t *X,
                        size_t m, size_t n, raft::distance::DistanceType metric,
                        raft::mr::device::buffer<value_idx> &indptr,
                        raft::mr::device::buffer<value_idx> &indices,
                        raft::mr::device::buffer<value_t> &data, int c) {
  auto stream = handle.get_stream();

  indptr.resize(m + 1, stream);

  distance_graph_impl<dist_type, value_idx, value_t> dist_graph;
  dist_graph.run(handle, X, m, n, metric, indptr, indices, data, c);

  // a little adjustment for distances of 0.
  // TODO: This will only need to be done when src_v==dst_v
  raft::linalg::unaryOp<value_t>(
    data.data(), data.data(), data.size(),
    [] __device__(value_t input) {
      if (input == 0)
        return std::numeric_limits<value_t>::max();
      else
        return input;
    },
    stream);
}

};  // namespace detail
};  // namespace hierarchy
};  // namespace raft