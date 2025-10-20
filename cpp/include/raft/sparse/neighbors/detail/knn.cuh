/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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
#include <raft/linalg/unary_op.cuh>
#include <raft/matrix/select_k.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/csr.hpp>
#include <raft/sparse/detail/utils.h>
#include <raft/sparse/distance/distance.cuh>
#include <raft/sparse/op/slice.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace raft::sparse::neighbors::detail {

template <typename value_idx, typename value_t>
struct csr_batcher_t {
  csr_batcher_t(value_idx batch_size,
                value_idx n_rows,
                const value_idx* csr_indptr,
                const value_idx* csr_indices,
                const value_t* csr_data)
    : batch_start_(0),
      batch_stop_(0),
      batch_rows_(0),
      total_rows_(n_rows),
      batch_size_(batch_size),
      csr_indptr_(csr_indptr),
      csr_indices_(csr_indices),
      csr_data_(csr_data),
      batch_csr_start_offset_(0),
      batch_csr_stop_offset_(0)
  {
  }

  void set_batch(int batch_num)
  {
    batch_start_ = batch_num * batch_size_;
    batch_stop_  = batch_start_ + batch_size_ - 1;  // zero-based indexing

    if (batch_stop_ >= total_rows_) batch_stop_ = total_rows_ - 1;  // zero-based indexing

    batch_rows_ = (batch_stop_ - batch_start_) + 1;
  }

  value_idx get_batch_csr_indptr_nnz(value_idx* batch_indptr, cudaStream_t stream)
  {
    raft::sparse::op::csr_row_slice_indptr(batch_start_,
                                           batch_stop_,
                                           csr_indptr_,
                                           batch_indptr,
                                           &batch_csr_start_offset_,
                                           &batch_csr_stop_offset_,
                                           stream);

    return batch_csr_stop_offset_ - batch_csr_start_offset_;
  }

  void get_batch_csr_indices_data(value_idx* csr_indices, value_t* csr_data, cudaStream_t stream)
  {
    raft::sparse::op::csr_row_slice_populate(batch_csr_start_offset_,
                                             batch_csr_stop_offset_,
                                             csr_indices_,
                                             csr_data_,
                                             csr_indices,
                                             csr_data,
                                             stream);
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

  const value_idx* csr_indptr_;
  const value_idx* csr_indices_;
  const value_t* csr_data_;

  value_idx batch_csr_start_offset_;
  value_idx batch_csr_stop_offset_;
};

template <typename value_idx, typename value_t>
class sparse_knn_t {
 public:
  sparse_knn_t(const value_idx* idxIndptr_,
               const value_idx* idxIndices_,
               const value_t* idxData_,
               size_t idxNNZ_,
               int n_idx_rows_,
               int n_idx_cols_,
               const value_idx* queryIndptr_,
               const value_idx* queryIndices_,
               const value_t* queryData_,
               size_t queryNNZ_,
               int n_query_rows_,
               int n_query_cols_,
               value_idx* output_indices_,
               value_t* output_dists_,
               int k_,
               raft::resources const& handle_,
               size_t batch_size_index_             = 2 << 14,  // approx 1M
               size_t batch_size_query_             = 2 << 14,
               raft::distance::DistanceType metric_ = raft::distance::DistanceType::L2Expanded,
               float metricArg_                     = 0)
    : idxIndptr(idxIndptr_),
      idxIndices(idxIndices_),
      idxData(idxData_),
      idxNNZ(idxNNZ_),
      n_idx_rows(n_idx_rows_),
      n_idx_cols(n_idx_cols_),
      queryIndptr(queryIndptr_),
      queryIndices(queryIndices_),
      queryData(queryData_),
      queryNNZ(queryNNZ_),
      n_query_rows(n_query_rows_),
      n_query_cols(n_query_cols_),
      output_indices(output_indices_),
      output_dists(output_dists_),
      k(k_),
      handle(handle_),
      batch_size_index(batch_size_index_),
      batch_size_query(batch_size_query_),
      metric(metric_),
      metricArg(metricArg_)
  {
  }

  void run()
  {
    using namespace raft::sparse;

    int n_batches_query = raft::ceildiv((size_t)n_query_rows, batch_size_query);
    csr_batcher_t<value_idx, value_t> query_batcher(
      batch_size_query, n_query_rows, queryIndptr, queryIndices, queryData);

    size_t rows_processed = 0;

    for (int i = 0; i < n_batches_query; i++) {
      /**
       * Compute index batch info
       */
      query_batcher.set_batch(i);

      /**
       * Slice CSR to rows in batch
       */

      rmm::device_uvector<value_idx> query_batch_indptr(query_batcher.batch_rows() + 1,
                                                        resource::get_cuda_stream(handle));

      value_idx n_query_batch_nnz = query_batcher.get_batch_csr_indptr_nnz(
        query_batch_indptr.data(), resource::get_cuda_stream(handle));

      rmm::device_uvector<value_idx> query_batch_indices(n_query_batch_nnz,
                                                         resource::get_cuda_stream(handle));
      rmm::device_uvector<value_t> query_batch_data(n_query_batch_nnz,
                                                    resource::get_cuda_stream(handle));

      query_batcher.get_batch_csr_indices_data(
        query_batch_indices.data(), query_batch_data.data(), resource::get_cuda_stream(handle));

      // A 3-partition temporary merge space to scale the batching. 2 parts for subsequent
      // batches and 1 space for the results of the merge, which get copied back to the top
      rmm::device_uvector<value_idx> merge_buffer_indices(0, resource::get_cuda_stream(handle));
      rmm::device_uvector<value_t> merge_buffer_dists(0, resource::get_cuda_stream(handle));

      value_t* dists_merge_buffer_ptr;
      value_idx* indices_merge_buffer_ptr;

      int n_batches_idx = raft::ceildiv((size_t)n_idx_rows, batch_size_index);
      csr_batcher_t<value_idx, value_t> idx_batcher(
        batch_size_index, n_idx_rows, idxIndptr, idxIndices, idxData);

      for (int j = 0; j < n_batches_idx; j++) {
        idx_batcher.set_batch(j);

        merge_buffer_indices.resize(query_batcher.batch_rows() * k * 3,
                                    resource::get_cuda_stream(handle));
        merge_buffer_dists.resize(query_batcher.batch_rows() * k * 3,
                                  resource::get_cuda_stream(handle));

        /**
         * Slice CSR to rows in batch
         */
        rmm::device_uvector<value_idx> idx_batch_indptr(idx_batcher.batch_rows() + 1,
                                                        resource::get_cuda_stream(handle));
        rmm::device_uvector<value_idx> idx_batch_indices(0, resource::get_cuda_stream(handle));
        rmm::device_uvector<value_t> idx_batch_data(0, resource::get_cuda_stream(handle));

        value_idx idx_batch_nnz = idx_batcher.get_batch_csr_indptr_nnz(
          idx_batch_indptr.data(), resource::get_cuda_stream(handle));

        idx_batch_indices.resize(idx_batch_nnz, resource::get_cuda_stream(handle));
        idx_batch_data.resize(idx_batch_nnz, resource::get_cuda_stream(handle));

        idx_batcher.get_batch_csr_indices_data(
          idx_batch_indices.data(), idx_batch_data.data(), resource::get_cuda_stream(handle));

        /**
         * Compute distances
         */
        uint64_t dense_size =
          (uint64_t)idx_batcher.batch_rows() * (uint64_t)query_batcher.batch_rows();
        rmm::device_uvector<value_t> batch_dists(dense_size, resource::get_cuda_stream(handle));

        RAFT_CUDA_TRY(cudaMemset(batch_dists.data(), 0, batch_dists.size() * sizeof(value_t)));

        compute_distances(idx_batcher,
                          query_batcher,
                          idx_batch_nnz,
                          n_query_batch_nnz,
                          idx_batch_indptr.data(),
                          idx_batch_indices.data(),
                          idx_batch_data.data(),
                          query_batch_indptr.data(),
                          query_batch_indices.data(),
                          query_batch_data.data(),
                          batch_dists.data());

        // Build batch indices array
        rmm::device_uvector<value_idx> batch_indices(batch_dists.size(),
                                                     resource::get_cuda_stream(handle));

        // populate batch indices array
        value_idx batch_rows = query_batcher.batch_rows(), batch_cols = idx_batcher.batch_rows();

        iota_fill(batch_indices.data(), batch_rows, batch_cols, resource::get_cuda_stream(handle));

        /**
         * Perform k-selection on batch & merge with other k-selections
         */
        size_t merge_buffer_offset = batch_rows * k;
        dists_merge_buffer_ptr     = merge_buffer_dists.data() + merge_buffer_offset;
        indices_merge_buffer_ptr   = merge_buffer_indices.data() + merge_buffer_offset;

        perform_k_selection(idx_batcher,
                            query_batcher,
                            batch_dists.data(),
                            batch_indices.data(),
                            dists_merge_buffer_ptr,
                            indices_merge_buffer_ptr);

        value_t* dists_merge_buffer_tmp_ptr     = dists_merge_buffer_ptr;
        value_idx* indices_merge_buffer_tmp_ptr = indices_merge_buffer_ptr;

        // Merge results of difference batches if necessary
        if (idx_batcher.batch_start() > 0) {
          size_t merge_buffer_tmp_out  = batch_rows * k * 2;
          dists_merge_buffer_tmp_ptr   = merge_buffer_dists.data() + merge_buffer_tmp_out;
          indices_merge_buffer_tmp_ptr = merge_buffer_indices.data() + merge_buffer_tmp_out;

          merge_batches(idx_batcher,
                        query_batcher,
                        merge_buffer_dists.data(),
                        merge_buffer_indices.data(),
                        dists_merge_buffer_tmp_ptr,
                        indices_merge_buffer_tmp_ptr);
        }

        // copy merged output back into merge buffer partition for next iteration
        raft::copy_async<value_idx>(merge_buffer_indices.data(),
                                    indices_merge_buffer_tmp_ptr,
                                    batch_rows * k,
                                    resource::get_cuda_stream(handle));
        raft::copy_async<value_t>(merge_buffer_dists.data(),
                                  dists_merge_buffer_tmp_ptr,
                                  batch_rows * k,
                                  resource::get_cuda_stream(handle));
      }

      // Copy final merged batch to output array
      raft::copy_async<value_idx>(output_indices + (rows_processed * k),
                                  merge_buffer_indices.data(),
                                  query_batcher.batch_rows() * k,
                                  resource::get_cuda_stream(handle));
      raft::copy_async<value_t>(output_dists + (rows_processed * k),
                                merge_buffer_dists.data(),
                                query_batcher.batch_rows() * k,
                                resource::get_cuda_stream(handle));

      rows_processed += query_batcher.batch_rows();
    }
  }

 private:
  void merge_batches(csr_batcher_t<value_idx, value_t>& idx_batcher,
                     csr_batcher_t<value_idx, value_t>& query_batcher,
                     value_t* merge_buffer_dists,
                     value_idx* merge_buffer_indices,
                     value_t* out_dists,
                     value_idx* out_indices)
  {
    // build translation buffer to shift resulting indices by the batch
    std::vector<value_idx> id_ranges;
    id_ranges.push_back(0);
    id_ranges.push_back(idx_batcher.batch_start());

    rmm::device_uvector<value_idx> trans(id_ranges.size(), resource::get_cuda_stream(handle));
    raft::update_device(
      trans.data(), id_ranges.data(), id_ranges.size(), resource::get_cuda_stream(handle));

    // combine merge buffers only if there's more than 1 partition to combine
    raft::spatial::knn::knn_merge_parts(merge_buffer_dists,
                                        merge_buffer_indices,
                                        out_dists,
                                        out_indices,
                                        query_batcher.batch_rows(),
                                        2,
                                        k,
                                        resource::get_cuda_stream(handle),
                                        trans.data());
  }

  void perform_k_selection(csr_batcher_t<value_idx, value_t> idx_batcher,
                           csr_batcher_t<value_idx, value_t> query_batcher,
                           value_t* batch_dists,
                           value_idx* batch_indices,
                           value_t* out_dists,
                           value_idx* out_indices)
  {
    // populate batch indices array
    value_idx batch_rows = query_batcher.batch_rows(), batch_cols = idx_batcher.batch_rows();

    // build translation buffer to shift resulting indices by the batch
    std::vector<value_idx> id_ranges;
    id_ranges.push_back(0);
    id_ranges.push_back(idx_batcher.batch_start());

    // in the case where the number of idx rows in the batch is < k, we
    // want to adjust k.
    value_idx n_neighbors = std::min(static_cast<value_idx>(k), batch_cols);

    bool ascending = raft::distance::is_min_close(metric);

    // kernel to slice first (min) k cols and copy into batched merge buffer
    raft::matrix::select_k<value_t, value_idx>(
      handle,
      make_device_matrix_view<const value_t, int64_t>(batch_dists, batch_rows, batch_cols),
      make_device_matrix_view<const value_idx, int64_t>(batch_indices, batch_rows, batch_cols),
      make_device_matrix_view<value_t, int64_t>(out_dists, batch_rows, n_neighbors),
      make_device_matrix_view<value_idx, int64_t>(out_indices, batch_rows, n_neighbors),
      ascending,
      true);
  }

  void compute_distances(csr_batcher_t<value_idx, value_t>& idx_batcher,
                         csr_batcher_t<value_idx, value_t>& query_batcher,
                         size_t idx_batch_nnz,
                         size_t query_batch_nnz,
                         value_idx* idx_batch_indptr,
                         value_idx* idx_batch_indices,
                         value_t* idx_batch_data,
                         value_idx* query_batch_indptr,
                         value_idx* query_batch_indices,
                         value_t* query_batch_data,
                         value_t* batch_dists)
  {
    /**
     * Compute distances
     */
    raft::sparse::distance::detail::distances_config_t<value_idx, value_t> dist_config(handle);
    dist_config.b_nrows = idx_batcher.batch_rows();
    dist_config.b_ncols = n_idx_cols;
    dist_config.b_nnz   = idx_batch_nnz;

    dist_config.b_indptr  = idx_batch_indptr;
    dist_config.b_indices = idx_batch_indices;
    dist_config.b_data    = idx_batch_data;

    dist_config.a_nrows = query_batcher.batch_rows();
    dist_config.a_ncols = n_query_cols;
    dist_config.a_nnz   = query_batch_nnz;

    dist_config.a_indptr  = query_batch_indptr;
    dist_config.a_indices = query_batch_indices;
    dist_config.a_data    = query_batch_data;

    if (raft::sparse::distance::supportedDistance.find(metric) ==
        raft::sparse::distance::supportedDistance.end())
      THROW("DistanceType not supported: %d", metric);

    raft::sparse::distance::pairwiseDistance(batch_dists, dist_config, metric, metricArg);
  }

  const value_idx *idxIndptr, *idxIndices, *queryIndptr, *queryIndices;
  value_idx* output_indices;
  const value_t *idxData, *queryData;
  value_t* output_dists;

  size_t idxNNZ, queryNNZ, batch_size_index, batch_size_query;

  raft::distance::DistanceType metric;

  float metricArg;

  int n_idx_rows, n_idx_cols, n_query_rows, n_query_cols, k;

  raft::resources const& handle;
};

};  // namespace raft::sparse::neighbors::detail
