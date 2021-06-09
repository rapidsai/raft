/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/utils/Heap.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>

#include <raft/linalg/distance_type.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <iostream>
#include <raft/handle.hpp>
#include <set>

#include "haversine_distance.cuh"
#include "processing.hpp"

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

template <typename value_idx = int64_t, typename value_t = float, int warp_q,
          int thread_q, int tpb>
__global__ void knn_merge_parts_kernel(value_t *inK, value_idx *inV,
                                       value_t *outK, value_idx *outV,
                                       size_t n_samples, int n_parts,
                                       value_t initK, value_idx initV, int k,
                                       value_idx *translations) {
  constexpr int kNumWarps = tpb / faiss::gpu::kWarpSize;

  __shared__ value_t smemK[kNumWarps * warp_q];
  __shared__ value_idx smemV[kNumWarps * warp_q];

  /**
   * Uses shared memory
   */
  faiss::gpu::BlockSelect<value_t, value_idx, false,
                          faiss::gpu::Comparator<value_t>, warp_q, thread_q,
                          tpb>
    heap(initK, initV, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int row = blockIdx.x;
  int total_k = k * n_parts;

  int i = threadIdx.x;

  // Get starting pointers for cols in current thread
  int part = i / k;
  size_t row_idx = (row * k) + (part * n_samples * k);

  int col = i % k;

  value_t *inKStart = inK + (row_idx + col);
  value_idx *inVStart = inV + (row_idx + col);

  int limit = faiss::gpu::utils::roundDown(total_k, faiss::gpu::kWarpSize);
  value_idx translation = 0;

  for (; i < limit; i += tpb) {
    translation = translations[part];
    heap.add(*inKStart, (*inVStart) + translation);

    part = (i + tpb) / k;
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

template <typename value_idx = int64_t, typename value_t = float, int warp_q,
          int thread_q>
inline void knn_merge_parts_impl(value_t *inK, value_idx *inV, value_t *outK,
                                 value_idx *outV, size_t n_samples, int n_parts,
                                 int k, cudaStream_t stream,
                                 value_idx *translations) {
  auto grid = dim3(n_samples);

  constexpr int n_threads = (warp_q <= 1024) ? 128 : 64;
  auto block = dim3(n_threads);

  auto kInit = faiss::gpu::Limits<value_t>::getMax();
  auto vInit = -1;
  knn_merge_parts_kernel<value_idx, value_t, warp_q, thread_q, n_threads>
    <<<grid, block, 0, stream>>>(inK, inV, outK, outV, n_samples, n_parts,
                                 kInit, vInit, k, translations);
  CUDA_CHECK(cudaPeekAtLastError());
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
template <typename value_idx = int64_t, typename value_t = float>
inline void knn_merge_parts(value_t *inK, value_idx *inV, value_t *outK,
                            value_idx *outV, size_t n_samples, int n_parts,
                            int k, cudaStream_t stream,
                            value_idx *translations) {
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
}

inline faiss::MetricType build_faiss_metric(
  raft::distance::DistanceType metric) {
  switch (metric) {
    case raft::distance::DistanceType::CosineExpanded:
      return faiss::MetricType::METRIC_INNER_PRODUCT;
    case raft::distance::DistanceType::CorrelationExpanded:
      return faiss::MetricType::METRIC_INNER_PRODUCT;
    case raft::distance::DistanceType::L2Expanded:
      return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L2Unexpanded:
      return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L2SqrtExpanded:
      return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L2SqrtUnexpanded:
      return faiss::MetricType::METRIC_L2;
    case raft::distance::DistanceType::L1:
      return faiss::MetricType::METRIC_L1;
    case raft::distance::DistanceType::InnerProduct:
      return faiss::MetricType::METRIC_INNER_PRODUCT;
    case raft::distance::DistanceType::LpUnexpanded:
      return faiss::MetricType::METRIC_Lp;
    case raft::distance::DistanceType::Linf:
      return faiss::MetricType::METRIC_Linf;
    case raft::distance::DistanceType::Canberra:
      return faiss::MetricType::METRIC_Canberra;
    case raft::distance::DistanceType::BrayCurtis:
      return faiss::MetricType::METRIC_BrayCurtis;
    case raft::distance::DistanceType::JensenShannon:
      return faiss::MetricType::METRIC_JensenShannon;
    default:
      THROW("MetricType not supported: %d", metric);
  }
}

/**
 * Search the kNN for the k-nearest neighbors of a set of query vectors
 * @param[in] input vector of device device memory array pointers to search
 * @param[in] sizes vector of memory sizes for each device array pointer in input
 * @param[in] D number of cols in input and search_items
 * @param[in] search_items set of vectors to query for neighbors
 * @param[in] n        number of items in search_items
 * @param[out] res_I    pointer to device memory for returning k nearest indices
 * @param[out] res_D    pointer to device memory for returning k nearest distances
 * @param[in] k        number of neighbors to query
 * @param[in] allocator the device memory allocator to use for temporary scratch memory
 * @param[in] userStream the main cuda stream to use
 * @param[in] internalStreams optional when n_params > 0, the index partitions can be
 *        queried in parallel using these streams. Note that n_int_streams also
 *        has to be > 0 for these to be used and their cardinality does not need
 *        to correspond to n_parts.
 * @param[in] n_int_streams size of internalStreams. When this is <= 0, only the
 *        user stream will be used.
 * @param[in] rowMajorIndex are the index arrays in row-major layout?
 * @param[in] rowMajorQuery are the query array in row-major layout?
 * @param[in] translations translation ids for indices when index rows represent
 *        non-contiguous partitions
 * @param[in] metric corresponds to the raft::distance::DistanceType enum (default is L2Expanded)
 * @param[in] metricArg metric argument to use. Corresponds to the p arg for lp norm
 */
template <typename IntType = int>
void brute_force_knn_impl(std::vector<float *> &input, std::vector<int> &sizes,
                          IntType D, float *search_items, IntType n,
                          int64_t *res_I, float *res_D, IntType k,
                          std::shared_ptr<deviceAllocator> allocator,
                          cudaStream_t userStream,
                          cudaStream_t *internalStreams = nullptr,
                          int n_int_streams = 0, bool rowMajorIndex = true,
                          bool rowMajorQuery = true,
                          std::vector<int64_t> *translations = nullptr,
                          raft::distance::DistanceType metric =
                            raft::distance::DistanceType::L2Expanded,
                          float metricArg = 0) {
  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors should be the same size");

  std::vector<int64_t> *id_ranges;
  if (translations == nullptr) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    id_ranges = new std::vector<int64_t>();
    int64_t total_n = 0;
    for (size_t i = 0; i < input.size(); i++) {
      id_ranges->push_back(total_n);
      total_n += sizes[i];
    }
  } else {
    // otherwise, use the given translations
    id_ranges = translations;
  }

  // perform preprocessing
  std::unique_ptr<MetricProcessor<float>> query_metric_processor =
    create_processor<float>(metric, n, D, k, rowMajorQuery, userStream,
                            allocator);
  query_metric_processor->preprocess(search_items);

  std::vector<std::unique_ptr<MetricProcessor<float>>> metric_processors(
    input.size());
  for (size_t i = 0; i < input.size(); i++) {
    metric_processors[i] = create_processor<float>(
      metric, sizes[i], D, k, rowMajorQuery, userStream, allocator);
    metric_processors[i]->preprocess(input[i]);
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  raft::mr::device::buffer<int64_t> trans(allocator, userStream,
                                          id_ranges->size());
  raft::update_device(trans.data(), id_ranges->data(), id_ranges->size(),
                      userStream);

  raft::mr::device::buffer<float> all_D(allocator, userStream, 0);
  raft::mr::device::buffer<int64_t> all_I(allocator, userStream, 0);

  float *out_D = res_D;
  int64_t *out_I = res_I;

  if (input.size() > 1) {
    all_D.resize(input.size() * k * n, userStream);
    all_I.resize(input.size() * k * n, userStream);

    out_D = all_D.data();
    out_I = all_I.data();
  }

  // Sync user stream only if using other streams to parallelize query
  if (n_int_streams > 0) CUDA_CHECK(cudaStreamSynchronize(userStream));

  for (size_t i = 0; i < input.size(); i++) {
    float *out_d_ptr = out_D + (i * k * n);
    int64_t *out_i_ptr = out_I + (i * k * n);

    cudaStream_t stream =
      raft::select_stream(userStream, internalStreams, n_int_streams, i);

    switch (metric) {
      case raft::distance::DistanceType::Haversine:

        ASSERT(D == 2,
               "Haversine distance requires 2 dimensions "
               "(latitude / longitude).");

        haversine_knn(out_i_ptr, out_d_ptr, input[i], search_items, sizes[i], n,
                      k, stream);
        break;
      default:
        faiss::MetricType m = build_faiss_metric(metric);

        faiss::gpu::StandardGpuResources gpu_res;

        gpu_res.noTempMemory();
        gpu_res.setDefaultStream(device, stream);

        faiss::gpu::GpuDistanceParams args;
        args.metric = m;
        args.metricArg = metricArg;
        args.k = k;
        args.dims = D;
        args.vectors = input[i];
        args.vectorsRowMajor = rowMajorIndex;
        args.numVectors = sizes[i];
        args.queries = search_items;
        args.queriesRowMajor = rowMajorQuery;
        args.numQueries = n;
        args.outDistances = out_d_ptr;
        args.outIndices = out_i_ptr;

        /**
         * @todo: Until FAISS supports pluggable allocation strategies,
         * we will not reap the benefits of the pool allocator for
         * avoiding device-wide synchronizations from cudaMalloc/cudaFree
         */
        bfKnn(&gpu_res, args);
    }

    CUDA_CHECK(cudaPeekAtLastError());
  }

  // Sync internal streams if used. We don't need to
  // sync the user stream because we'll already have
  // fully serial execution.
  for (int i = 0; i < n_int_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(internalStreams[i]));
  }

  if (input.size() > 1 || translations != nullptr) {
    // This is necessary for proper index translations. If there are
    // no translations or partitions to combine, it can be skipped.
    knn_merge_parts(out_D, out_I, res_D, res_I, n, input.size(), k, userStream,
                    trans.data());
  }

  // Perform necessary post-processing
  if (metric == raft::distance::DistanceType::L2SqrtExpanded ||
      metric == raft::distance::DistanceType::L2SqrtUnexpanded ||
      metric == raft::distance::DistanceType::LpUnexpanded) {
    /**
	* post-processing
	*/
    float p = 0.5;  // standard l2
    if (metric == raft::distance::DistanceType::LpUnexpanded)
      p = 1.0 / metricArg;
    raft::linalg::unaryOp<float>(
      res_D, res_D, n * k,
      [p] __device__(float input) { return powf(input, p); }, userStream);
  }

  query_metric_processor->revert(search_items);
  query_metric_processor->postprocess(out_D);
  for (size_t i = 0; i < input.size(); i++) {
    metric_processors[i]->revert(input[i]);
  }

  if (translations == nullptr) delete id_ranges;
};

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
