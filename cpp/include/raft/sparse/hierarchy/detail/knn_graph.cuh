/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include "connectivities.cuh"

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/coo.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>

#include <raft/spatial/knn/knn.hpp>

#include <raft/linalg/distance_type.h>
#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/hierarchy/connectivities.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <limits>

namespace raft {
namespace hierarchy {
namespace detail {

/**
 * Fills indices array of pairwise distance array
 * @tparam value_idx
 * @param indices
 * @param m
 */
template <typename value_idx>
__global__ void fill_indices(value_idx *indices, size_t m, size_t nnz) {
  value_idx tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nnz) return;
  value_idx v = tid / m;
  indices[tid] = v;
}

template <typename value_idx>
value_idx build_k(value_idx n_samples, int c) {
  // from "kNN-MST-Agglomerative: A fast & scalable graph-based data clustering
  // approach on GPU"
  return min(n_samples, (value_idx)floor(logf(n_samples)) + c);
}

template <typename in_t, typename out_t>
__global__ void conv_indices_kernel(in_t *inds, out_t *out, size_t nnz) {
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nnz) return;
  out_t v = inds[tid];
  out[tid] = v;
}

template <typename in_t, typename out_t, int tpb = 256>
void conv_indices(in_t *inds, out_t *out, size_t size, cudaStream_t stream) {
  size_t blocks = raft::ceildiv(size, (size_t)tpb);
  conv_indices_kernel<<<blocks, tpb, 0, stream>>>(inds, out, size);
}

/**
 * Constructs a (symmetrized) knn graph edge list from
 * dense input vectors.
 *
 * Note: The resulting KNN graph is not
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param X
 * @param m
 * @param n
 * @param metric
 * @param indptr
 * @param indices
 * @param data
 * @param c
 */
template <typename value_idx = int, typename value_t = float>
void knn_graph(const raft::handle_t &handle, const value_t *X, size_t m,
               size_t n, raft::distance::DistanceType metric,
               MLCommon::Sparse::COO<value_t, value_idx> &out, int c = 15) {
  int k = build_k(m, c);

  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  size_t nnz = m * k;

  raft::mr::device::buffer<value_idx> rows(d_alloc, stream, nnz);
  raft::mr::device::buffer<value_idx> indices(d_alloc, stream, nnz);
  raft::mr::device::buffer<value_t> data(d_alloc, stream, nnz);

  size_t blocks = raft::ceildiv(nnz, (size_t)256);
  fill_indices<value_idx><<<blocks, 256, 0, stream>>>(rows.data(), k, nnz);

  std::vector<value_t *> inputs;
  inputs.push_back(const_cast<value_t *>(X));

  std::vector<int> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  raft::mr::device::buffer<int64_t> int64_indices(d_alloc, stream, nnz);

  uint32_t knn_start = raft::curTimeMillis();
  raft::spatial::knn::brute_force_knn(
    handle, inputs, sizes, n, const_cast<value_t *>(X), m, int64_indices.data(),
    data.data(), k, true, true, metric);

  // convert from current knn's 64-bit to 32-bit.
  conv_indices(int64_indices.data(), indices.data(), nnz, stream);

  raft::sparse::linalg::symmetrize(handle, rows.data(), indices.data(),
                                   data.data(), m, k, nnz, out);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());
}

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

    raft::copy_async(indices.data(), knn_graph_coo.cols(), knn_graph_coo.nnz,
                     stream);
    raft::copy_async(data.data(), knn_graph_coo.vals(), knn_graph_coo.nnz,
                     stream);
  }
};

};  // namespace detail
};  // end namespace hierarchy
};  // end namespace raft