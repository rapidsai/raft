/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <raft/spatial/knn/faiss_mr.hpp>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/utils/Heap.h>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

template <typename key_t, typename payload_t>
constexpr int kFaissMaxK()
{
  return (sizeof(key_t) + sizeof(payload_t) > 8) ? 512 : 1024;
}

template <typename key_t, typename payload_t, bool select_min, int warp_q, int thread_q, int tpb>
__global__ void select_k_kernel(const key_t* inK,
                                const payload_t* inV,
                                size_t n_rows,
                                size_t n_cols,
                                key_t* outK,
                                payload_t* outV,
                                key_t initK,
                                payload_t initV,
                                int k)
{
  using align_warp        = Pow2<WarpSize>;
  constexpr int kNumWarps = align_warp::div(tpb);

  __shared__ key_t smemK[kNumWarps * warp_q];
  __shared__ payload_t smemV[kNumWarps * warp_q];

  faiss::gpu::
    BlockSelect<key_t, payload_t, select_min, faiss::gpu::Comparator<key_t>, warp_q, thread_q, tpb>
      heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row = blockIdx.x;
  {
    size_t i = size_t(threadIdx.x);

    inK += row * n_cols;
    if (inV != nullptr) { inV += row * n_cols; }

    // Whole warps must participate in the selection
    size_t limit = align_warp::roundDown(n_cols);

    for (; i < limit; i += tpb) {
      heap.add(inK[i], (inV != nullptr) ? inV[i] : payload_t(i));
    }

    // Handle last remainder fraction of a warp of elements
    if (i < n_cols) { heap.addThreadQ(inK[i], (inV != nullptr) ? inV[i] : payload_t(i)); }
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    outK[row * k + i] = smemK[i];
    outV[row * k + i] = smemV[i];
  }
}

template <typename payload_t = int, typename key_t = float, int warp_q, int thread_q>
inline void select_k_impl(const key_t* inK,
                          const payload_t* inV,
                          size_t n_rows,
                          size_t n_cols,
                          key_t* outK,
                          payload_t* outV,
                          bool select_min,
                          int k,
                          cudaStream_t stream)
{
  auto grid = dim3(n_rows);

  constexpr int n_threads = (warp_q <= 1024) ? 128 : 64;
  auto block              = dim3(n_threads);

  auto kInit = select_min ? upper_bound<key_t>() : lower_bound<key_t>();
  auto vInit = -1;
  if (select_min) {
    select_k_kernel<key_t, payload_t, false, warp_q, thread_q, n_threads>
      <<<grid, block, 0, stream>>>(inK, inV, n_rows, n_cols, outK, outV, kInit, vInit, k);
  } else {
    select_k_kernel<key_t, payload_t, true, warp_q, thread_q, n_threads>
      <<<grid, block, 0, stream>>>(inK, inV, n_rows, n_cols, outK, outV, kInit, vInit, k);
  }
  RAFT_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Select the k-nearest neighbors from dense
 * distance and index matrices.
 *
 * @param[in] inK partitioned knn distance matrix
 * @param[in] inV partitioned knn index matrix
 * @param[in] n_rows number of rows in distance and index matrices
 * @param[in] n_cols number of columns in distance and index matrices
 * @param[out] outK merged knn distance matrix
 * @param[out] outV merged knn index matrix
 * @param[in] select_min whether to select the min or the max distances
 * @param[in] k number of neighbors per partition (also number of merged neighbors)
 * @param[in] stream CUDA stream to use
 */
template <typename payload_t = int, typename key_t = float>
inline void select_k(const key_t* inK,
                     const payload_t* inV,
                     size_t n_rows,
                     size_t n_cols,
                     key_t* outK,
                     payload_t* outV,
                     bool select_min,
                     int k,
                     cudaStream_t stream)
{
  constexpr int max_k = kFaissMaxK<payload_t, key_t>();
  if (k == 1)
    select_k_impl<payload_t, key_t, 1, 1>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
  else if (k <= 32)
    select_k_impl<payload_t, key_t, 32, 2>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
  else if (k <= 64)
    select_k_impl<payload_t, key_t, 64, 3>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
  else if (k <= 128)
    select_k_impl<payload_t, key_t, 128, 3>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
  else if (k <= 256)
    select_k_impl<payload_t, key_t, 256, 4>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
  else if (k <= 512)
    select_k_impl<payload_t, key_t, 512, 8>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
  else if (k <= 1024 && k <= max_k)
    select_k_impl<payload_t, key_t, max_k, 8>(
      inK, inV, n_rows, n_cols, outK, outV, select_min, k, stream);
  else
    ASSERT(k <= max_k, "Current max k is %d (requested %d)", max_k, k);
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
