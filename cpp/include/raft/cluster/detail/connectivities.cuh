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

#include <raft/cluster/single_linkage_types.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/neighbors/knn_graph.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/tuple.h>

#include <limits>

namespace raft::cluster::detail {

template <raft::cluster::LinkageDistance dist_type, typename value_idx, typename value_t>
struct distance_graph_impl {
  void run(raft::resources const& handle,
           const value_t* X,
           size_t m,
           size_t n,
           raft::distance::DistanceType metric,
           rmm::device_uvector<value_idx>& indptr,
           rmm::device_uvector<value_idx>& indices,
           rmm::device_uvector<value_t>& data,
           int c);
};

/**
 * Connectivities specialization to build a knn graph
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct distance_graph_impl<raft::cluster::LinkageDistance::KNN_GRAPH, value_idx, value_t> {
  void run(raft::resources const& handle,
           const value_t* X,
           size_t m,
           size_t n,
           raft::distance::DistanceType metric,
           rmm::device_uvector<value_idx>& indptr,
           rmm::device_uvector<value_idx>& indices,
           rmm::device_uvector<value_t>& data,
           int c)
  {
    auto stream        = resource::get_cuda_stream(handle);
    auto thrust_policy = resource::get_thrust_policy(handle);

    // Need to symmetrize knn into undirected graph
    raft::sparse::COO<value_t, value_idx> knn_graph_coo(stream);

    raft::sparse::neighbors::knn_graph(handle, X, m, n, metric, knn_graph_coo, c);

    indices.resize(knn_graph_coo.nnz, stream);
    data.resize(knn_graph_coo.nnz, stream);

    // self-loops get max distance
    auto transform_in = thrust::make_zip_iterator(
      thrust::make_tuple(knn_graph_coo.rows(), knn_graph_coo.cols(), knn_graph_coo.vals()));

    thrust::transform(thrust_policy,
                      transform_in,
                      transform_in + knn_graph_coo.nnz,
                      knn_graph_coo.vals(),
                      [=] __device__(const thrust::tuple<value_idx, value_idx, value_t>& tup) {
                        bool self_loop = thrust::get<0>(tup) == thrust::get<1>(tup);
                        return (self_loop * std::numeric_limits<value_t>::max()) +
                               (!self_loop * thrust::get<2>(tup));
                      });

    raft::sparse::convert::sorted_coo_to_csr(
      knn_graph_coo.rows(), knn_graph_coo.nnz, indptr.data(), m + 1, stream);

    // TODO: Wouldn't need to copy here if we could compute knn
    // graph directly on the device uvectors
    // ref: https://github.com/rapidsai/raft/issues/227
    raft::copy_async(indices.data(), knn_graph_coo.cols(), knn_graph_coo.nnz, stream);
    raft::copy_async(data.data(), knn_graph_coo.vals(), knn_graph_coo.nnz, stream);
  }
};

template <typename value_idx>
RAFT_KERNEL fill_indices2(value_idx* indices, size_t m, size_t nnz)
{
  value_idx tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nnz) return;
  value_idx v  = tid % m;
  indices[tid] = v;
}

/**
 * Compute connected CSR of pairwise distances
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param X
 * @param m
 * @param n
 * @param metric
 * @param[out] indptr
 * @param[out] indices
 * @param[out] data
 */
template <typename value_idx, typename value_t>
void pairwise_distances(const raft::resources& handle,
                        const value_t* X,
                        size_t m,
                        size_t n,
                        raft::distance::DistanceType metric,
                        value_idx* indptr,
                        value_idx* indices,
                        value_t* data)
{
  auto stream      = resource::get_cuda_stream(handle);
  auto exec_policy = resource::get_thrust_policy(handle);

  value_idx nnz = m * m;

  value_idx blocks = raft::ceildiv(nnz, (value_idx)256);
  fill_indices2<value_idx><<<blocks, 256, 0, stream>>>(indices, m, nnz);

  thrust::sequence(exec_policy, indptr, indptr + m, 0, (int)m);

  raft::update_device(indptr + m, &nnz, 1, stream);

  // TODO: It would ultimately be nice if the MST could accept
  // dense inputs directly so we don't need to double the memory
  // usage to hand it a sparse array here.
  distance::pairwise_distance<value_t, value_idx>(handle, X, X, data, m, m, n, metric);
  // self-loops get max distance
  auto transform_in =
    thrust::make_zip_iterator(thrust::make_tuple(thrust::make_counting_iterator(0), data));

  thrust::transform(exec_policy,
                    transform_in,
                    transform_in + nnz,
                    data,
                    [=] __device__(const thrust::tuple<value_idx, value_t>& tup) {
                      value_idx idx  = thrust::get<0>(tup);
                      bool self_loop = idx % m == idx / m;
                      return (self_loop * std::numeric_limits<value_t>::max()) +
                             (!self_loop * thrust::get<1>(tup));
                    });
}

/**
 * Connectivities specialization for pairwise distances
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx, typename value_t>
struct distance_graph_impl<raft::cluster::LinkageDistance::PAIRWISE, value_idx, value_t> {
  void run(const raft::resources& handle,
           const value_t* X,
           size_t m,
           size_t n,
           raft::distance::DistanceType metric,
           rmm::device_uvector<value_idx>& indptr,
           rmm::device_uvector<value_idx>& indices,
           rmm::device_uvector<value_t>& data,
           int c)
  {
    auto stream = resource::get_cuda_stream(handle);

    size_t nnz = m * m;

    indices.resize(nnz, stream);
    data.resize(nnz, stream);

    pairwise_distances(handle, X, m, n, metric, indptr.data(), indices.data(), data.data());
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
template <typename value_idx, typename value_t, raft::cluster::LinkageDistance dist_type>
void get_distance_graph(raft::resources const& handle,
                        const value_t* X,
                        size_t m,
                        size_t n,
                        raft::distance::DistanceType metric,
                        rmm::device_uvector<value_idx>& indptr,
                        rmm::device_uvector<value_idx>& indices,
                        rmm::device_uvector<value_t>& data,
                        int c)
{
  auto stream = resource::get_cuda_stream(handle);

  indptr.resize(m + 1, stream);

  distance_graph_impl<dist_type, value_idx, value_t> dist_graph;
  dist_graph.run(handle, X, m, n, metric, indptr, indices, data, c);
}

};  // namespace raft::cluster::detail
