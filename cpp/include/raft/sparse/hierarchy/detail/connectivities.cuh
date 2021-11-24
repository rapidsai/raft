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
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <raft/linalg/distance_type.h>
#include <raft/sparse/hierarchy/common.h>
#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/selection/knn_graph.cuh>

#include <limits>

namespace raft {
namespace hierarchy {
namespace detail {

template <raft::hierarchy::LinkageDistance dist_type, typename value_idx,
          typename value_t>
struct distance_graph_impl {
  void run(const raft::handle_t &handle, const value_t *X, size_t m, size_t n,
           raft::distance::DistanceType metric,
           rmm::device_uvector<value_idx> &indptr,
           rmm::device_uvector<value_idx> &indices,
           rmm::device_uvector<value_t> &data, int c);
};

/**
 * Connectivities specialization to build a knn graph
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct distance_graph_impl<raft::hierarchy::LinkageDistance::KNN_GRAPH,
                           value_idx, value_t> {
  void run(const raft::handle_t &handle, const value_t *X, size_t m, size_t n,
           raft::distance::DistanceType metric,
           rmm::device_uvector<value_idx> &indptr,
           rmm::device_uvector<value_idx> &indices,
           rmm::device_uvector<value_t> &data, int c) {
    auto d_alloc = handle.get_device_allocator();
    auto stream = handle.get_stream();
    auto exec_policy = rmm::exec_policy(rmm::cuda_stream_view{stream});

    // Need to symmetrize knn into undirected graph
    raft::sparse::COO<value_t, value_idx> knn_graph_coo(d_alloc, stream);

    raft::sparse::selection::knn_graph(handle, X, m, n, metric, knn_graph_coo,
                                       c);

    indices.resize(knn_graph_coo.nnz, stream);
    data.resize(knn_graph_coo.nnz, stream);

    // self-loops get max distance
    auto transform_in = thrust::make_zip_iterator(thrust::make_tuple(
      knn_graph_coo.rows(), knn_graph_coo.cols(), knn_graph_coo.vals()));

    thrust::transform(
      exec_policy, transform_in, transform_in + knn_graph_coo.nnz,
      knn_graph_coo.vals(),
      [=] __device__(const thrust::tuple<value_idx, value_idx, value_t> &tup) {
        bool self_loop = thrust::get<0>(tup) == thrust::get<1>(tup);
        return (self_loop * std::numeric_limits<value_t>::max()) +
               (!self_loop * thrust::get<2>(tup));
      });

    raft::sparse::convert::sorted_coo_to_csr(knn_graph_coo.rows(),
                                             knn_graph_coo.nnz, indptr.data(),
                                             m + 1, d_alloc, stream);

    // TODO: Wouldn't need to copy here if we could compute knn
    // graph directly on the device uvectors
    // ref: https://github.com/rapidsai/raft/issues/227
    raft::copy_async(indices.data(), knn_graph_coo.cols(), knn_graph_coo.nnz,
                     stream);
    raft::copy_async(data.data(), knn_graph_coo.vals(), knn_graph_coo.nnz,
                     stream);
  }
};

/**
 * Returns a CSR connectivities graph based on the given linkage distance.
 * @tparam value_idx
 * @tparam value_t
 * @tparam dist_type
 * @param[in] handle raft handle
 * @param[in] X dense data for which to construct connectivites
 * @param[in] m number of rows in X
 * @param[in] n number of columns in X
 * @param[in] metric distance metric to use
 * @param[out] indptr indptr array of connectivities graph
 * @param[out] indices column indices array of connectivities graph
 * @param[out] data distances array of connectivities graph
 * @param[out] c constant 'c' used for nearest neighbors-based distances
 *             which will guarantee k <= log(n) + c
 */
template <typename value_idx, typename value_t,
          raft::hierarchy::LinkageDistance dist_type>
void get_distance_graph(const raft::handle_t &handle, const value_t *X,
                        size_t m, size_t n, raft::distance::DistanceType metric,
                        rmm::device_uvector<value_idx> &indptr,
                        rmm::device_uvector<value_idx> &indices,
                        rmm::device_uvector<value_t> &data, int c) {
  auto stream = handle.get_stream();

  indptr.resize(m + 1, stream);

  distance_graph_impl<dist_type, value_idx, value_t> dist_graph;
  dist_graph.run(handle, X, m, n, metric, indptr, indices, data, c);
}

};  // namespace detail
};  // namespace hierarchy
};  // namespace raft
