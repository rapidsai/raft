/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "detail/brute_force_knn.hpp"
#include "detail/processing.hpp"

#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

namespace raft {
  namespace knn {

using deviceAllocator = raft::mr::device::allocator;

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
 * @param[in] metric corresponds to the FAISS::metricType enum (default is euclidean)
 * @param[in] metricArg metric argument to use. Corresponds to the p arg for lp norm
 * @param[in] expanded_form whether or not lp variants should be reduced w/ lp-root
 */
template <typename IntType = int>
void brute_force_knn_impl(std::vector<float *> &input, std::vector<int> &sizes,
													IntType D, float *search_items, IntType n, int64_t *res_I,
													float *res_D, IntType k,
													std::shared_ptr<raft::mr::device::allocator> allocator,
													cudaStream_t userStream,
													cudaStream_t *internalStreams = nullptr,
													int n_int_streams = 0, bool rowMajorIndex = true,
													bool rowMajorQuery = true,
													std::vector<int64_t> *translations = nullptr,
													MetricType metric = MetricType::METRIC_L2,
													float metricArg = 2.0, bool expanded_form = false) {

 ASSERT(input.size() == sizes.size(),
         "input and sizes vectors should be the same size");

  faiss::MetricType m = detail::build_faiss_metric(metric);

  std::vector<int64_t> *id_ranges;
  if (translations == nullptr) {
    // If we don't have explicit translations
    // for offsets of the indices, build them
    // from the local partitions
    id_ranges = new std::vector<int64_t>();
    int64_t total_n = 0;
    for (int i = 0; i < input.size(); i++) {
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
  for (int i = 0; i < input.size(); i++) {
    metric_processors[i] = create_processor<float>(
      metric, sizes[i], D, k, rowMajorQuery, userStream, allocator);
    metric_processors[i]->preprocess(input[i]);
  }

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  raft::mr::device::buffer<int64_t> trans(allocator, userStream, id_ranges->size());
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

  for (int i = 0; i < input.size(); i++) {
    faiss::gpu::StandardGpuResources gpu_res;

    cudaStream_t stream =
      raft::select_stream(userStream, internalStreams, n_int_streams, i);

    gpu_res.noTempMemory();
    gpu_res.setCudaMallocWarning(false);
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
    args.outDistances = out_D + (i * k * n);
    args.outIndices = out_I + (i * k * n);

    /**
     * @todo: Until FAISS supports pluggable allocation strategies,
     * we will not reap the benefits of the pool allocator for
     * avoiding device-wide synchronizations from cudaMalloc/cudaFree
     */
    bfKnn(&gpu_res, args);

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
    detail::knn_merge_parts(out_D, out_I, res_D, res_I, n, input.size(), k, userStream,
                    trans.data());
  }

  // Perform necessary post-processing
  if ((m == faiss::MetricType::METRIC_L2 ||
       m == faiss::MetricType::METRIC_Lp) &&
      !expanded_form) {
    /**
	* post-processing
	*/
    float p = 0.5;  // standard l2
    if (m == faiss::MetricType::METRIC_Lp) p = 1.0 / metricArg;
    raft::linalg::unaryOp<float>(
      res_D, res_D, n * k,
      [p] __device__(float input) { return powf(input, p); }, userStream);
  }

  query_metric_processor->revert(search_items);
  query_metric_processor->postprocess(out_D);
  for (int i = 0; i < input.size(); i++) {
    metric_processors[i]->revert(input[i]);
  }

  if (translations == nullptr) delete id_ranges;
}

/**
 * @brief Flat C++ API function to perform a brute force knn on
 * a series of input arrays and combine the results into a single
 * output array for indexes and distances.
 *
 * @param[in] handle the cuml handle to use
 * @param[in] input vector of pointers to the input arrays
 * @param[in] sizes vector of sizes of input arrays
 * @param[in] D the dimensionality of the arrays
 * @param[in] search_items array of items to search of dimensionality D
 * @param[in] n number of rows in search_items
 * @param[out] res_I the resulting index array of size n * k
 * @param[out] res_D the resulting distance array of size n * k
 * @param[in] k the number of nearest neighbors to return
 * @param[in] rowMajorIndex are the index arrays in row-major order?
 * @param[in] rowMajorQuery are the query arrays in row-major order?
 * @param[in] metric distance metric to use. Euclidean (L2) is used by
 * 			   default
 * @param[in] metric_arg the value of `p` for Minkowski (l-p) distances. This
 * 					 is ignored if the metric_type is not Minkowski.
 * @param[in] expanded should lp-based distances be returned in their expanded
 * 					 form (e.g., without raising to the 1/p power).
 */
void brute_force_knn(raft::handle_t &handle, std::vector<float *> &input,
                     std::vector<int> &sizes, int D, float *search_items, int n,
                     int64_t *res_I, float *res_D, int k, bool rowMajorIndex,
                     bool rowMajorQuery, MetricType metric, float metric_arg,
                     bool expanded) {
  ASSERT(input.size() == sizes.size(),
         "input and sizes vectors must be the same size");

  std::vector<cudaStream_t> int_streams = handle.get_internal_streams();

  brute_force_knn_impl(
    input, sizes, D, search_items, n, res_I, res_D, k,
    handle.get_device_allocator(), handle.get_stream(), int_streams.data(),
    handle.get_num_internal_streams(), rowMajorIndex, rowMajorQuery, nullptr,
    metric, metric_arg, expanded);
}

} // namespace knn
} // namespace raft
