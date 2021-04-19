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

#include <rmm/device_uvector.hpp>

#include <raft/cudart_utils.h>
#include <raft/sparse/utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/sparse/selection/selection.cuh>

#include <raft/spatial/knn/detail/selection.cuh>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

constexpr int DEFAULT_IDX_BATCH_SIZE = 1 << 14;
constexpr int DEFAULT_QUERY_BATCH_SIZE = 1 << 14;

template <typename value_t>
DI value_t compute_haversine(value_t x1, value_t y1, value_t x2, value_t y2) {
  value_t sin_0 = sin(0.5 * (x1 - y1));
  value_t sin_1 = sin(0.5 * (x2 - y2));
  value_t rdist = sin_0 * sin_0 + cos(x1) * cos(y1) * sin_1 * sin_1;

  return 2 * asin(sqrt(rdist));
}

template <typename value_idx = int>
struct batcher_t {
  batcher_t(value_idx batch_size, value_idx n_rows)
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
template <typename value_t = float>
__global__ void haversine_knn_kernel(value_t *out_dists, const value_t *index,
                                     const value_t *query, int n_index_rows,
                                     int n_query_rows) {
  int out_dist_idx = blockDim.x * blockIdx.x + threadIdx.x;

  int query_row = out_dist_idx / n_index_rows;
  int idx_row = out_dist_idx % n_index_rows;

  const value_t *query_ptr = query + (query_row * 2);
  const value_t *idx_ptr = index + (idx_row * 2);

  if (query_row < n_query_rows && idx_row < n_index_rows) {
    value_t x1 = query_ptr[0];
    value_t x2 = query_ptr[1];
    value_t y1 = idx_ptr[0];
    value_t y2 = idx_ptr[1];

    value_t dist = compute_haversine(x1, y1, x2, y2);
    out_dists[out_dist_idx] = dist;
  }
}

template <typename value_idx = int, typename value_t = float, int tpb = 256>
class haversine_knn_t {
 public:
  haversine_knn_t(const value_t *idx_, size_t idx_rows_, const value_t *query_,
                  size_t query_rows_, value_idx *output_indices_,
                  value_t *output_dists_, int k_, cudaStream_t stream_,
                  int batch_size_index_ = DEFAULT_IDX_BATCH_SIZE,
                  int batch_size_query_ = DEFAULT_QUERY_BATCH_SIZE)
    : idx(idx_),
      n_idx_rows(idx_rows_),
      query(query_),
      n_query_rows(query_rows_),
      output_indices(output_indices_),
      output_dists(output_dists_),
      k(k_),
      stream(stream_),
      batch_size_index(batch_size_index_),
      batch_size_query(batch_size_query_) {}

  void run() {
    int n_batches_query = raft::ceildiv(n_query_rows, batch_size_query);

    batcher_t<int> query_batcher(batch_size_query, n_query_rows);

    size_t rows_processed = 0;

    for (int i = 0; i < n_batches_query; i++) {
      query_batcher.set_batch(i);

      // A 3-partition temporary merge space to scale the batching. 2 parts for subsequent
      // batches and 1 space for the results of the merge, which get copied back to the top
      rmm::device_uvector<value_idx> merge_buffer_indices(0, stream);
      rmm::device_uvector<value_t> merge_buffer_dists(0, stream);

      value_t *dists_merge_buffer_ptr;
      value_idx *indices_merge_buffer_ptr;

      int n_batches_idx = raft::ceildiv(n_idx_rows, batch_size_index);
      batcher_t<int> idx_batcher(batch_size_index, n_idx_rows);

      for (int j = 0; j < n_batches_idx; j++) {
        idx_batcher.set_batch(j);

        printf(
          "n_batches_query=%d, n_batches_index=%d, i=%d, j=%d, "
          "batch_size_query=%d, batch_size_idx=%d, batch_start_query=%d, "
          "batch_start_index=%d\n",
          n_batches_query, n_batches_idx, i, j, batch_size_query,
          batch_size_index, query_batcher.batch_start(),
          idx_batcher.batch_start());

        merge_buffer_indices.resize(query_batcher.batch_rows() * k * 3, stream);
        merge_buffer_dists.resize(query_batcher.batch_rows() * k * 3, stream);

        /**
         * Compute distances
         */
        size_t dists_size =
          idx_batcher.batch_rows() * query_batcher.batch_rows();
        rmm::device_uvector<value_t> batch_dists(dists_size, stream);

        thrust::fill(thrust::cuda::par.on(stream), batch_dists.data(),
                     batch_dists.data() + batch_dists.size(),
                     std::numeric_limits<value_t>::max());

        compute_distances(idx_batcher, query_batcher, batch_dists.data());

        CUDA_CHECK(cudaStreamSynchronize(stream));

        raft::print_device_vector("batch_dists", batch_dists.data(),
                                  batch_dists.size(), std::cout);

        // Build batch indices array
        rmm::device_uvector<value_idx> batch_indices(batch_dists.size(),
                                                     stream);

        // populate batch indices array
        value_idx batch_rows = query_batcher.batch_rows(),
                  batch_cols = idx_batcher.batch_rows();

        raft::sparse::iota_fill(batch_indices.data(), batch_rows, batch_cols,
                                stream);

        /**
         * Perform k-selection on batch & merge with other k-selections
         */
        size_t merge_buffer_offset = batch_rows * k;
        dists_merge_buffer_ptr =
          merge_buffer_dists.data() + merge_buffer_offset;
        indices_merge_buffer_ptr =
          merge_buffer_indices.data() + merge_buffer_offset;

        thrust::fill(thrust::cuda::par.on(stream), dists_merge_buffer_ptr,
                     dists_merge_buffer_ptr + (batch_rows * k),
                     std::numeric_limits<value_t>::max());

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
  void merge_batches(batcher_t<int> &idx_batcher, batcher_t<int> &query_batcher,
                     value_t *merge_buffer_dists,
                     value_idx *merge_buffer_indices, value_t *out_dists,
                     value_idx *out_indices) {
    // build translation buffer to shift resulting indices by the batch
    std::vector<value_idx> id_ranges = {0, idx_batcher.batch_start()};

    rmm::device_uvector<value_idx> trans(id_ranges.size(), stream);
    raft::update_device(trans.data(), id_ranges.data(), id_ranges.size(),
                        stream);

    // combine merge buffers only if there's more than 1 partition to combine
    raft::spatial::knn::detail::knn_merge_parts(
      merge_buffer_dists, merge_buffer_indices, out_dists, out_indices,
      query_batcher.batch_rows(), 2, k, stream, trans.data());
  }

  void perform_k_selection(batcher_t<int> &idx_batcher,
                           batcher_t<int> &query_batcher, value_t *batch_dists,
                           value_idx *batch_indices, value_t *out_dists,
                           value_idx *out_indices) {
    // kernel to slice first (min) k cols and copy into batched merge buffer
    raft::sparse::selection::select_k(
      batch_dists, batch_indices, query_batcher.batch_rows(),
      idx_batcher.batch_rows(), out_dists, out_indices, true,
      min(k, idx_batcher.batch_rows()), stream, k);
  }

  void compute_distances(batcher_t<int> &idx_batcher,
                         batcher_t<int> &query_batcher, value_t *batch_dists) {
    size_t total_threads =
      idx_batcher.batch_rows() * query_batcher.batch_rows();
    size_t grid = raft::ceildiv((size_t)total_threads, (size_t)tpb);

    const value_t *idx_ptr = idx + (idx_batcher.batch_start() * 2);
    const value_t *query_ptr = query + (query_batcher.batch_start() * 2);

    haversine_knn_kernel<<<grid, tpb, 0, stream>>>(
      batch_dists, idx_ptr, query_ptr, idx_batcher.batch_rows(),
      query_batcher.batch_rows());
  }

  value_idx *output_indices;
  value_t *output_dists;

  const value_t *idx, *query;

  int batch_size_index, batch_size_query;
  int n_idx_rows, n_query_rows, k;

  cudaStream_t stream;
};

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
template <typename value_idx = int, typename value_t = float, int tpb = 256>
void haversine_knn(value_idx *out_inds, value_t *out_dists,
                   const value_t *index, const value_t *query, int n_index_rows,
                   int n_query_rows, int k, cudaStream_t stream,
                   size_t batch_size_index = DEFAULT_IDX_BATCH_SIZE,
                   size_t batch_size_query = DEFAULT_QUERY_BATCH_SIZE) {
  haversine_knn_t<value_idx, value_t, tpb>(
    index, n_index_rows, query, n_query_rows, out_inds, out_dists, k, stream,
    batch_size_index, batch_size_query)
    .run();
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
