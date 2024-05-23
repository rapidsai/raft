/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>
#include <raft/neighbors/detail/faiss_select/DistanceUtils.h>
#include <raft/neighbors/detail/faiss_select/Select.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>

namespace raft::neighbors::detail {

template <typename value_idx = std::int64_t,
          typename value_t   = float,
          int warp_q,
          int thread_q,
          int tpb>
RAFT_KERNEL knn_merge_parts_kernel(const value_t* inK,
                                   const value_idx* inV,
                                   value_t* outK,
                                   value_idx* outV,
                                   size_t n_samples,
                                   int n_parts,
                                   value_t initK,
                                   value_idx initV,
                                   int k,
                                   value_idx* translations)
{
  constexpr int kNumWarps = tpb / WarpSize;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ value_idx smemV[kNumWarps * warp_q];

  /**
   * Uses shared memory
   */
  faiss_select::
    BlockSelect<value_t, value_idx, false, faiss_select::Comparator<value_t>, warp_q, thread_q, tpb>
      heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row     = blockIdx.x;
  int total_k = k * n_parts;

  int i = threadIdx.x;

  // Get starting pointers for cols in current thread
  int part       = i / k;
  size_t row_idx = (row * k) + (part * n_samples * k);

  int col = i % k;

  const value_t* inKStart   = inK + (row_idx + col);
  const value_idx* inVStart = inV + (row_idx + col);

  int limit             = Pow2<WarpSize>::roundDown(total_k);
  value_idx translation = 0;

  for (; i < limit; i += tpb) {
    translation = translations[part];
    heap.add(*inKStart, (*inVStart) + translation);

    part    = (i + tpb) / k;
    row_idx = (row * k) + (part * n_samples * k);

    col = (i + tpb) % k;

    inKStart = inK + (row_idx + col);
    inVStart = inV + (row_idx + col);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < total_k) {
    translation = translations[part];
    heap.addThreadQ(*inKStart, (*inVStart) + translation);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    outK[row * k + i] = smemK[i];
    outV[row * k + i] = smemV[i];
  }
}

template <typename value_idx = std::int64_t, typename value_t = float, int warp_q, int thread_q>
inline void knn_merge_parts_impl(const value_t* inK,
                                 const value_idx* inV,
                                 value_t* outK,
                                 value_idx* outV,
                                 size_t n_samples,
                                 int n_parts,
                                 int k,
                                 cudaStream_t stream,
                                 value_idx* translations)
{
  auto grid = dim3(n_samples);

  constexpr int n_threads = (warp_q < 1024) ? 128 : 64;
  auto block              = dim3(n_threads);

  auto kInit = std::numeric_limits<value_t>::max();
  auto vInit = -1;
  knn_merge_parts_kernel<value_idx, value_t, warp_q, thread_q, n_threads>
    <<<grid, block, 0, stream>>>(
      inK, inV, outK, outV, n_samples, n_parts, kInit, vInit, k, translations);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * @brief Merge knn distances and index matrix, which have been partitioned
 * by row, into a single matrix with only the k-nearest neighbors.
 *
 * @param inK partitioned knn distance matrix
 * @param inV partitioned knn index matrix
 * @param outK merged knn distance matrix
 * @param outV merged knn index matrix
 * @param n_samples number of samples per partition
 * @param n_parts number of partitions
 * @param k number of neighbors per partition (also number of merged neighbors)
 * @param stream CUDA stream to use
 * @param translations mapping of index offsets for each partition
 */
template <typename value_idx = std::int64_t, typename value_t = float>
inline void knn_merge_parts(const value_t* inK,
                            const value_idx* inV,
                            value_t* outK,
                            value_idx* outV,
                            size_t n_samples,
                            int n_parts,
                            int k,
                            cudaStream_t stream,
                            value_idx* translations)
{
  if (k == 1)
    knn_merge_parts_impl<value_idx, value_t, 1, 1>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 32)
    knn_merge_parts_impl<value_idx, value_t, 32, 2>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 64)
    knn_merge_parts_impl<value_idx, value_t, 64, 3>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 128)
    knn_merge_parts_impl<value_idx, value_t, 128, 3>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 256)
    knn_merge_parts_impl<value_idx, value_t, 256, 4>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 512)
    knn_merge_parts_impl<value_idx, value_t, 512, 8>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else if (k <= 1024)
    knn_merge_parts_impl<value_idx, value_t, 1024, 8>(
      inK, inV, outK, outV, n_samples, n_parts, k, stream, translations);
  else
    THROW("Unimplemented for k=%d, knn_merge_parts works for k<=1024", k);
}
}  // namespace raft::neighbors::detail
