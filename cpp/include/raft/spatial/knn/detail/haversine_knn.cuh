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
#include <raft/handle.hpp>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

template <typename value_t>
DI value_t compute_haversine(value_t x1, value_t y1, value_t x2, value_t y2) {
  value_t sin_0 = sin(0.5 * (x1 - y1));
  value_t sin_1 = sin(0.5 * (x2 - y2));
  value_t rdist = sin_0 * sin_0 + cos(x1) * cos(y1) * sin_1 * sin_1;

  return 2 * asin(sqrt(rdist));
}


template <typename value_idx, typename value_t>
struct batcher_t {
  batcher_t(value_idx batch_size, value_idx n_rows,
                const value_t *data)
    : batch_start_(0),
      batch_stop_(0),
      batch_rows_(0),
      total_rows_(n_rows),
      batch_size_(batch_size) {}

  void set_batch(int batch_num) {
    batch_start_ = batch_num * batch_size_;
    batch_stop_ = batch_start_ + batch_size_ - 1;  // zero-based indexing

    if (batch_stop_ >= total_rows_)
      batch_stop_ = total_rows_ - 1;  // zero-based indexing

    batch_rows_ = (batch_stop_ - batch_start_) + 1;
  }

  value_idx batch_rows() const { return batch_rows_; }

  value_idx batch_start() const { return batch_start_; }

  value_idx batch_stop() const { return batch_stop_; }

 private:
  value_idx batch_size_;
  value_idx batch_start_;
  value_idx batch_stop_;
  value_idx batch_rows_;

  value_idx total_rows_;
};


/**
 * @tparam value_idx data type of indices
 * @tparam value_t data type of values and distances
 * @tparam warp_q
 * @tparam thread_q
 * @tparam tpb
 * @param[out] out_inds output indices
 * @param[out] out_dists output distances
 * @param[in] index index array
 * @param[in] query query array
 * @param[in] n_index_rows number of rows in index array
 * @param[in] k number of closest neighbors to return
 */
template <typename value_idx, typename value_t, int warp_q = 1024,
          int thread_q = 8, int tpb = 256>
__global__ void haversine_knn_kernel(value_idx *out_inds, value_t *out_dists,
                                     const value_t *index, const value_t *query,
                                     size_t n_index_rows, int k) {


  // TODO: Each thread needs to load their query and index rows, compute
  // haversine, and write to the correspoding distance batch

  faiss::gpu::BlockSelect<value_t, value_idx, false,
                          faiss::gpu::Comparator<value_t>, warp_q, thread_q,
                          tpb>
    heap(faiss::gpu::Limits<value_t>::getMax(), -1, smemK, smemV, k);

  // Grid is exactly sized to rows available
  int limit = faiss::gpu::utils::roundDown(n_index_rows, faiss::gpu::kWarpSize);

  const value_t *query_ptr = query + (blockIdx.x * 2);
  value_t x1 = query_ptr[0];
  value_t x2 = query_ptr[1];

  int i = threadIdx.x;

  for (; i < limit; i += tpb) {
    const value_t *idx_ptr = index + (i * 2);
    value_t y1 = idx_ptr[0];
    value_t y2 = idx_ptr[1];

    value_t dist = compute_haversine(x1, y1, x2, y2);

    heap.add(dist, i);
  }

  // Handle last remainder fraction of a warp of elements
  if (i < n_index_rows) {
    const value_t *idx_ptr = index + (i * 2);
    value_t y1 = idx_ptr[0];
    value_t y2 = idx_ptr[1];

    value_t dist = compute_haversine(x1, y1, x2, y2);

    heap.addThreadQ(dist, i);
  }

  heap.reduce();

  for (int i = threadIdx.x; i < k; i += tpb) {
    out_dists[blockIdx.x * k + i] = smemK[i];
    out_inds[blockIdx.x * k + i] = smemV[i];
  }
}

template <typename value_idx, typename value_t>
class sparse_knn_t {
 public:
  sparse_knn_t(const value_t *idx_, size_t idx_rows_,
               const value_t *query_, size_t query_rows_,
               value_idx *output_indices_, value_t *output_dists_, int k_,
               std::shared_ptr<raft::mr::device::allocator> allocator_,
               cudaStream_t stream_,
               size_t batch_size_index_ = 2 << 14,  // approx 1M
               size_t batch_size_query_ = 2 << 14)
    : idx(idx_),
      idx_rows(idx_rows_),
      query(query_),
      query_rows(query_rows_),
      output_indices(output_indices_),
      output_dists(output_dists_),
      k(k_),
      allocator(allocator_),
      stream(stream_),
      batch_size_index(batch_size_index_),
      batch_size_query(batch_size_query_) {}

  void run() {
    int n_batches_query = raft::ceildiv((size_t)query_rows, batch_size_query);

    batcher_t<value_idx, value_t> query_batcher(batch_size_query, query_rows);

    // A 3-partition temporary merge space to scale the batching. 2 parts for subsequent
    // batches and 1 space for the results of the merge, which get copied back to the top

    size_t merge_buffer_size = query_batcher.batch_rows() * k * 3;

    rmm::device_uvector<value_idx> merge_buffer_indices(merge_buffer_size, stream);
    rmm::device_uvector<value_t> merge_buffer_dists(merge_buffer_size, stream);

    for (int i = 0; i < n_batches_query; i++) {

      query_batcher.set_batch(i);

      int n_batches_idx = raft::ceildiv((size_t)n_idx_rows, batch_size_index);

      batcher_t<value_idx, value_t> idx_batcher(batch_size_idx, idx_rows);

      value_t *dists_merge_buffer_ptr;
      value_idx *indices_merge_buffer_ptr;

      for (int j = 0; j < n_batches_idx; j++) {

        idx_batcher.set_batch(j);

        /**
         * Compute distances
         */
        size_t dists_size =
          idx_batcher.batch_rows() * query_batcher.batch_rows();
        rmm::device_uvector<value_t> batch_dists(dists_size, stream);

        CUDA_CHECK(cudaMemset(batch_dists.data(), 0,
                              batch_dists.size() * sizeof(value_t)));

        compute_distances(idx_batcher, query_batcher, idx_batch_nnz,
                          n_query_batch_nnz, idx_batch_indptr.data(),
                          idx_batch_indices.data(), idx_batch_data.data(),
                          query_batch_indptr.data(), query_batch_indices.data(),
                          query_batch_data.data(), batch_dists.data());

        // Build batch indices array
        rmm::device_uvector<value_idx> batch_indices(batch_dists.size(),
                                                     stream);

        // populate batch indices array
        value_idx batch_rows = query_batcher.batch_rows(),
          batch_cols = idx_batcher.batch_rows();

        iota_fill(batch_indices.data(), batch_rows, batch_cols, stream);

        /**
         * Perform k-selection on batch & merge with other k-selections
         */
        size_t merge_buffer_offset = batch_rows * k;
        dists_merge_buffer_ptr =
          merge_buffer_dists.data() + merge_buffer_offset;
        indices_merge_buffer_ptr =
          merge_buffer_indices.data() + merge_buffer_offset;

        perform_k_selection(idx_batcher, query_batcher, batch_dists.data(),
                            batch_indices.data(), dists_merge_buffer_ptr,
                            indices_merge_buffer_ptr);

        value_t *dists_merge_buffer_tmp_ptr = dists_merge_buffer_ptr;
        value_idx *indices_merge_buffer_tmp_ptr = indices_merge_buffer_ptr;

        // Merge results of difference batches if necessary
        if (idx_batcher.batch_start() > 0) {
          size_t merge_buffer_tmp_out = batch_rows * k * 2;
          dists_merge_buffer_tmp_ptr =
            merge_buffer_dists.data() + merge_buffer_tmp_out;
          indices_merge_buffer_tmp_ptr =
            merge_buffer_indices.data() + merge_buffer_tmp_out;

          merge_batches(idx_batcher, query_batcher, merge_buffer_dists.data(),
                        merge_buffer_indices.data(), dists_merge_buffer_tmp_ptr,
                        indices_merge_buffer_tmp_ptr);
        }

        // copy merged output back into merge buffer partition for next iteration
        raft::copy_async<value_idx>(merge_buffer_indices.data(),
                                    indices_merge_buffer_tmp_ptr,
                                    batch_rows * k, stream);
        raft::copy_async<value_t>(merge_buffer_dists.data(),
                                  dists_merge_buffer_tmp_ptr, batch_rows * k,
                                  stream);
      }

      // Copy final merged batch to output array
      raft::copy_async<value_idx>(output_indices + (rows_processed * k),
                                  merge_buffer_indices.data(),
                                  query_batcher.batch_rows() * k, stream);
      raft::copy_async<value_t>(output_dists + (rows_processed * k),
                                merge_buffer_dists.data(),
                                query_batcher.batch_rows() * k, stream);

      rows_processed += query_batcher.batch_rows();
    }
  }

 private:
  void merge_batches(csr_batcher_t<value_idx, value_t> &idx_batcher,
                     csr_batcher_t<value_idx, value_t> &query_batcher,
                     value_t *merge_buffer_dists,
                     value_idx *merge_buffer_indices, value_t *out_dists,
                     value_idx *out_indices) {
    // build translation buffer to shift resulting indices by the batch
    std::vector<value_idx> id_ranges = (0, idx_batcher.batch_start());

    rmm::device_uvector<value_idx> trans(id_ranges.size(), stream);
    raft::update_device(trans.data(), id_ranges.data(), id_ranges.size(),
                        stream);

    // combine merge buffers only if there's more than 1 partition to combine
    raft::spatial::knn::detail::knn_merge_parts(
      merge_buffer_dists, merge_buffer_indices, out_dists, out_indices,
      query_batcher.batch_rows(), 2, k, stream, trans.data());
  }

  void perform_k_selection(csr_batcher_t<value_idx, value_t> idx_batcher,
                           csr_batcher_t<value_idx, value_t> query_batcher,
                           value_t *batch_dists, value_idx *batch_indices,
                           value_t *out_dists, value_idx *out_indices) {

    // in the case where the number of idx rows in the batch is < k, we
    // want to adjust k.
    value_idx n_neighbors = ;

    // kernel to slice first (min) k cols and copy into batched merge buffer
    raft::sparse::selection::select_k(batch_dists, batch_indices,
             query_batcher.batch_rows(), idx_batcher.batch_rows(),
             out_dists, out_indices,
             true, min(k, batch_cols), stream);
  }

  void compute_distances(csr_batcher_t<value_idx, value_t> &idx_batcher,
                         csr_batcher_t<value_idx, value_t> &query_batcher,
                         value_t *batch_dists) {

    // @TODO: Call haversine kernel
  }

  value_idx *output_indices;
  value_t *output_dists;

  size_t batch_size_index, batch_size_query;

  int n_idx_rows, n_query_rows, k;

  std::shared_ptr<raft::mr::device::allocator> allocator;

  cudaStream_t stream;
};

/**
   * Search the sparse kNN for the k-nearest neighbors of a set of sparse query vectors
   * using some distance implementation
   * @param[in] idxIndptr csr indptr of the index matrix (size n_idx_rows + 1)
   * @param[in] idxIndices csr column indices array of the index matrix (size n_idx_nnz)
   * @param[in] idxData csr data array of the index matrix (size idxNNZ)
   * @param[in] idxNNA number of non-zeros for sparse index matrix
   * @param[in] n_idx_rows number of data samples in index matrix
   * @param[in] queryIndptr csr indptr of the query matrix (size n_query_rows + 1)
   * @param[in] queryIndices csr indices array of the query matrix (size queryNNZ)
   * @param[in] queryData csr data array of the query matrix (size queryNNZ)
   * @param[in] queryNNZ number of non-zeros for sparse query matrix
   * @param[in] n_query_rows number of data samples in query matrix
   * @param[in] n_query_cols number of features in query matrix
   * @param[out] output_indices dense matrix for output indices (size n_query_rows * k)
   * @param[out] output_dists dense matrix for output distances (size n_query_rows * k)
   * @param[in] k the number of neighbors to query
   * @param[in] cusparseHandle the initialized cusparseHandle instance to use
   * @param[in] allocator device allocator instance to use
   * @param[in] stream CUDA stream to order operations with respect to
   * @param[in] batch_size_index maximum number of rows to use from index matrix per batch
   * @param[in] batch_size_query maximum number of rows to use from query matrix per batch
   * @param[in] metric distance metric/measure to use
   * @param[in] metricArg potential argument for metric (currently unused)
   */
template <typename value_idx = int, typename value_t = float, int TPB_X = 32>
void haversine_knn(const value_idx *idxIndptr, const value_idx *idxIndices,
                     const value_t *idxData, size_t idxNNZ, int n_idx_rows,
                     int n_idx_cols, const value_idx *queryIndptr,
                     const value_idx *queryIndices, const value_t *queryData,
                     size_t queryNNZ, int n_query_rows, int n_query_cols,
                     value_idx *output_indices, value_t *output_dists, int k,
                     cusparseHandle_t cusparseHandle,
                     std::shared_ptr<raft::mr::device::allocator> allocator,
                     cudaStream_t stream,
                     size_t batch_size_index = 2 << 14,  // approx 1M
                     size_t batch_size_query = 2 << 14,
                     raft::distance::DistanceType metric =
                     raft::distance::DistanceType::L2Expanded,
                     float metricArg = 0) {
  sparse_knn_t<value_idx, value_t>(
    idxIndptr, idxIndices, idxData, idxNNZ, n_idx_rows, n_idx_cols, queryIndptr,
    queryIndices, queryData, queryNNZ, n_query_rows, n_query_cols,
    output_indices, output_dists, k, cusparseHandle, allocator, stream,
    batch_size_index, batch_size_query, metric, metricArg)
    .run();
}


/**
 * Conmpute the k-nearest neighbors using the Haversine
 * (great circle arc) distance. Input is assumed to have
 * 2 dimensions (latitude, longitude) in radians.

 * @tparam value_idx
 * @tparam value_t
 * @param[out] out_inds output indices array on device (size n_query_rows * k)
 * @param[out] out_dists output dists array on device (size n_query_rows * k)
 * @param[in] index input index array on device (size n_index_rows * 2)
 * @param[in] query input query array on device (size n_query_rows * 2)
 * @param[in] n_index_rows number of rows in index array
 * @param[in] n_query_rows number of rows in query array
 * @param[in] k number of closest neighbors to return
 * @param[in] stream stream to order kernel launch
 */
template <typename value_idx, typename value_t>
void haversine_knn(value_idx *out_inds, value_t *out_dists,
                   const value_t *index, const value_t *query,
                   size_t n_index_rows, size_t n_query_rows, int k,
                   size_t batch_query_rows, size_t batch_index_rows,
                   cudaStream_t stream) {

  rmm::device_uvector<value_t> distance_buf(tile_rows * tile_cols);

  for(int i = 0; i < num_queries; i += batch_query_rows) {

    int cur_query_size = std::min(tile_rows, num_queries - i);

    for (int j = 0; j < num_indices; j+= batch_index_rows) {

      int cur_col_tile = j / tile_cols;

      // run haversine distance over tile

      // compute block select

    }
  }


  haversine_knn_kernel<<<n_query_rows, 128, 0, stream>>>(
    out_inds, out_dists, index, query, n_index_rows, k);
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
