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

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>

#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/symmetrize.cuh>

#include <raft/spatial/knn/knn.cuh>

#include <raft/distance/distance_type.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace raft {
namespace sparse {
namespace selection {
namespace detail {

/**
 * Fills indices array of pairwise distance array
 * @tparam value_idx
 * @param indices
 * @param m
 */
template <typename value_idx>
__global__ void fill_indices(value_idx* indices, size_t m, size_t nnz)
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

/**
 * Constructs a (symmetrized) knn graph edge list from
 * dense input vectors.
 *
 * Note: The resulting KNN graph is not guaranteed to be connected.
 *
 * @tparam idx_t
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
template <typename idx_t = int, typename value_t = float>
void knn_graph(const handle_t& handle,
               const value_t* X,
               std::size_t m,
               std::size_t n,
               raft::distance::DistanceType metric,
               raft::sparse::COO<value_t, idx_t>& out,
               int c = 15)
{
  size_t k = build_k(m, c);

  auto stream = handle.get_stream();

  size_t nnz = m * k;

  rmm::device_uvector<idx_t> rows(nnz, stream);
  rmm::device_uvector<idx_t> indices(nnz, stream);
  rmm::device_uvector<value_t> distances(nnz, stream);

  size_t blocks = ceildiv(nnz, (size_t)256);
  fill_indices<idx_t><<<blocks, 256, 0, stream>>>(rows.data(), k, nnz);

  std::vector<value_t*> inputs = {const_cast<value_t*>(X)};

  std::vector<size_t> sizes = {m};

  raft::spatial::knn::brute_force_knn<int, value_t, size_t>(handle,
                                                            inputs,
                                                            sizes,
                                                            n,
                                                            const_cast<value_t*>(X),
                                                            m,
                                                            indices.data(),
                                                            distances.data(),
                                                            k,
                                                            true,
                                                            true,
                                                            nullptr,
                                                            metric);

  raft::sparse::linalg::symmetrize(
    handle, rows.data(), indices.data(), distances.data(), m, k, nnz, out);
}

};  // namespace detail
};  // namespace selection
};  // namespace sparse
};  // end namespace raft
