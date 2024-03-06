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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/spatial/knn/knn.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <limits>

namespace raft::sparse::neighbors::detail {

/**
 * Fills indices array of pairwise distance array
 * @tparam value_idx
 * @param indices
 * @param m
 */
template <typename value_idx>
RAFT_KERNEL fill_indices(value_idx* indices, size_t m, size_t nnz)
{
  value_idx tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (tid >= nnz) return;
  value_idx v  = tid / m;
  indices[tid] = v;
}

template <typename value_idx>
value_idx build_k(value_idx n_samples, int c)
{
  // from "kNN-MST-Agglomerative: A fast & scalable graph-based data clustering
  // approach on GPU"
  return std::min(n_samples, std::max((value_idx)2, (value_idx)floor(log2(n_samples)) + c));
}

template <typename in_t, typename out_t>
RAFT_KERNEL conv_indices_kernel(in_t* inds, out_t* out, size_t nnz)
{
  size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nnz) return;
  out_t v  = inds[tid];
  out[tid] = v;
}

template <typename in_t, typename out_t, int tpb = 256>
void conv_indices(in_t* inds, out_t* out, size_t size, cudaStream_t stream)
{
  size_t blocks = ceildiv(size, (size_t)tpb);
  conv_indices_kernel<<<blocks, tpb, 0, stream>>>(inds, out, size);
}

/**
 * Constructs a (symmetrized) knn graph edge list from
 * dense input vectors.
 *
 * Note: The resulting KNN graph is not guaranteed to be connected.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle raft handle
 * @param[in] X dense matrix of input data samples and observations
 * @param[in] m number of data samples (rows) in X
 * @param[in] n number of observations (columns) in X
 * @param[in] metric distance metric to use when constructing neighborhoods
 * @param[out] out output edge list
 * @param[out] out output edge list
 * @param c
 */
template <typename value_idx = int, typename value_t = float>
void knn_graph(raft::resources const& handle,
               const value_t* X,
               size_t m,
               size_t n,
               raft::distance::DistanceType metric,
               raft::sparse::COO<value_t, value_idx>& out,
               int c = 15)
{
  size_t k = build_k(m, c);

  auto stream = resource::get_cuda_stream(handle);

  size_t nnz = m * k;

  rmm::device_uvector<value_idx> rows(nnz, stream);
  rmm::device_uvector<value_idx> indices(nnz, stream);
  rmm::device_uvector<value_t> data(nnz, stream);

  size_t blocks = ceildiv(nnz, (size_t)256);
  fill_indices<value_idx><<<blocks, 256, 0, stream>>>(rows.data(), k, nnz);

  std::vector<value_t*> inputs;
  inputs.push_back(const_cast<value_t*>(X));

  std::vector<size_t> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  rmm::device_uvector<int64_t> int64_indices(nnz, stream);

  raft::spatial::knn::brute_force_knn<int64_t, value_t, size_t>(handle,
                                                                inputs,
                                                                sizes,
                                                                n,
                                                                const_cast<value_t*>(X),
                                                                m,
                                                                int64_indices.data(),
                                                                data.data(),
                                                                k,
                                                                true,
                                                                true,
                                                                nullptr,
                                                                metric);

  // convert from current knn's 64-bit to 32-bit.
  conv_indices(int64_indices.data(), indices.data(), nnz, stream);

  raft::sparse::linalg::symmetrize(
    handle, rows.data(), indices.data(), data.data(), m, k, nnz, out);
}

};  // namespace raft::sparse::neighbors::detail
