/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/sparse/op/sort.cuh>

#include <thrust/reduce.h>

namespace raft::sparse::matrix::detail {

template <typename IndexType, typename ValueType>
struct mapper {
  mapper(IndexType* map) : map(map) {}

  __host__ __device__ ValueType operator()(const IndexType& key, const ValueType& val)
  {
    map[key] = val;
    return (ValueType(0));
  }
  IndexType* map;
};

/**
 * @brief Get unique counts
 * @param handle: raft resource handle
 * @param sort_vector: Input COO array that contains the keys.
 * @param secondary_vector: Input with secondary keys of COO, (columns or rows).
 * @param data: Input COO values array.
 * @param itr_vals: Input array used to calculate counts.
 * @param keys_out: Output array with one entry for each key. (same size as counts_out)
 * @param counts_out: Output array with cumulative sum for each key. (same size as keys_out)
 */
template <typename IndexType, typename ValueType, typename IdxT>
void get_uniques_counts(raft::resources const& handle,
                        IndexType* rows,
                        IndexType* columns,
                        ValueType* values,
                        IndexType nnz,
                        raft::device_vector_view<IndexType, int64_t> keys_out,
                        raft::device_vector_view<ValueType, int64_t> counts_out)
{
  // auto rows           = coo_in.structure_view().get_rows();
  // auto columns           = coo_in.structure_view().get_cols();
  // auto values           = coo_in.view().get_elements();
  // auto nnz            = coo_in.structure_view().get_nnz();

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // raft::sparse::op::coo_sort(nnz,
  //                            nnz,
  //                            nnz,
  //                            rows,
  //                            columns,
  //                            values,
  //                            stream);
  // replace this call with raft version when available
  // (https://github.com/rapidsai/raft/issues/2477)
  thrust::reduce_by_key(raft::resource::get_thrust_policy(handle),
                        rows,
                        rows + nnz,
                        values,
                        keys_out.data_handle(),
                        counts_out.data_handle());
}

template <typename ValueType = float, typename IndexType = int>
void fit_tfidf(raft::resources const& handle,
               IndexType* columns,
               ValueType* values,
               IndexType num_cols,
               IndexType nnz,
               raft::device_vector_view<IndexType, int64_t> idFeatCount,
               int& fullFeatCount)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  rmm::device_uvector<char> workspace(0, stream);
  raft::cluster::detail::countLabels(
    handle, columns, idFeatCount.data_handle(), nnz, num_cols, workspace);

  // get total number of words
  auto batchIdLen = raft::make_host_scalar<ValueType>(0);
  auto values_mat = raft::make_device_scalar<ValueType>(handle, 0);
  raft::linalg::mapReduce<ValueType>(
    values_mat.data_handle(), nnz, 0.0f, raft::identity_op(), raft::add_op(), stream, values);
  raft::copy(batchIdLen.data_handle(), values_mat.data_handle(), values_mat.size(), stream);
  fullFeatCount += (int)batchIdLen(0);
}

template <typename ValueType = float, typename IndexType = int>
void fit_bm25(raft::resources const& handle,
              IndexType* rows,
              IndexType* columns,
              ValueType* values,
              IndexType num_rows,
              IndexType num_cols,
              IndexType nnz,
              raft::device_vector_view<IndexType, int64_t> idFeatCount,
              int& fullFeatCount,
              raft::device_vector_view<IndexType, int64_t> rowFeatCnts)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  int uniq_cnt  = raft::sparse::neighbors::get_n_components(rows, nnz, stream);
  auto row_keys = raft::make_device_vector<IndexType>(handle, uniq_cnt);
  auto row_cnts = raft::make_device_vector<ValueType>(handle, uniq_cnt);
  get_uniques_counts<IndexType, ValueType, int64_t>(
    handle, rows, columns, values, nnz, row_keys.view(), row_cnts.view());

  auto dummy_vec = raft::make_device_vector<IndexType>(handle, uniq_cnt);
  raft::linalg::map(handle,
                    dummy_vec.view(),
                    mapper<IndexType, ValueType>(rowFeatCnts.data_handle()),
                    raft::make_const_mdspan(row_keys.view()),
                    raft::make_const_mdspan(row_cnts.view()));

  fit_tfidf(handle, columns, values, num_cols, nnz, idFeatCount, fullFeatCount);
}

template <typename ValueType = float, typename IndexType = int>
void transform_bm25(raft::resources const& handle,
                    IndexType* rows,
                    IndexType* columns,
                    ValueType* values,
                    IndexType num_rows,
                    raft::device_vector_view<IndexType, int64_t> featIdCount,
                    IndexType fullIdLen,
                    raft::device_vector_view<IndexType, int64_t> rowFeatCnts,
                    float k_param,
                    float b_param,
                    raft::device_vector_view<ValueType, int64_t> results)
{
  float avgIdLen = (ValueType)fullIdLen / num_rows;
  raft::linalg::map_offset(handle, results, [=] __device__(IndexType idx) {
    ValueType tf     = (ValueType)raft::log<ValueType>(values[idx]);
    ValueType idf_in = (ValueType)num_rows / featIdCount.data_handle()[columns[idx]];
    ValueType idf    = (ValueType)raft::log<ValueType>(idf_in + 1);
    ValueType bm =
      ((k_param + 1) * tf) /
      (k_param * ((1.0f - b_param) + b_param * (rowFeatCnts.data_handle()[rows[idx]] / avgIdLen)) +
       tf);
    return (ValueType)idf * bm;
  });
}

template <typename ValueType = float, typename IndexType = int>
void transform_tfidf(raft::resources const& handle,
                     IndexType* columns,
                     ValueType* values,
                     IndexType num_rows,
                     raft::device_vector_view<IndexType, int64_t> featIdCount,
                     IndexType fullIdLen,
                     raft::device_vector_view<ValueType, int64_t> results)
{
  raft::linalg::map_offset(handle, results, [=] __device__(IndexType idx) {
    ValueType tf     = (ValueType)raft::log<double>(values[idx]);
    ValueType idf_in = (double)num_rows / featIdCount[columns[idx]];
    ValueType idf    = (ValueType)raft::log<double>(idf_in + 1);
    return (ValueType)tf * idf;
  });
}

}  // namespace raft::sparse::matrix::detail
