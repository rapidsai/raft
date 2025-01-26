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

#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/detail/preprocessing.cuh>

#include <map>
#include <optional>

namespace raft::sparse::matrix {

template <typename ValueType = float, typename IndexType = int>
class SparseEncoder {
 private:
  int* featIdCount;
  float fullIdLen;
  int vocabSize;
  int numRows;

 public:
  SparseEncoder(int vocab_size);
  SparseEncoder(std::map<int, int> featIdValues, int num_rows, int full_id_len, int vocab_size);
  ~SparseEncoder();
  void fit(raft::resources& handle,
           raft::device_coo_matrix<ValueType,
                                   IndexType,
                                   IndexType,
                                   IndexType,
                                   raft::device_uvector_policy,
                                   raft::PRESERVING> coo_in);
  void fit(raft::resources& handle,
           raft::device_csr_matrix<ValueType,
                                   IndexType,
                                   IndexType,
                                   IndexType,
                                   raft::device_uvector_policy,
                                   raft::PRESERVING> csr_in);
  void transform(raft::resources& handle,
                 raft::device_csr_matrix<ValueType,
                                         IndexType,
                                         IndexType,
                                         IndexType,
                                         raft::device_uvector_policy,
                                         raft::PRESERVING> csr_in,
                 float* results,
                 bool bm25_on,
                 float k_param = 1.6f,
                 float b_param = 0.75f);
  void transform(raft::resources& handle,
                 raft::device_coo_matrix<ValueType,
                                         IndexType,
                                         IndexType,
                                         IndexType,
                                         raft::device_uvector_policy,
                                         raft::PRESERVING> coo_in,
                 float* results,
                 bool bm25_on,
                 float k_param = 1.6f,
                 float b_param = 0.75f);

 private:
  void _fit(raft::resources& handle,
            raft::device_vector_view<IndexType, int64_t> rows,
            raft::device_vector_view<IndexType, int64_t> columns,
            raft::device_vector_view<ValueType, int64_t> values,
            int num_rows);
  void _fit_feats(IndexType* cols, IndexType* counts, IndexType nnz, IndexType* results);
  void transform(raft::resources& handle,
                 raft::device_vector_view<IndexType, int64_t> rows,
                 raft::device_vector_view<IndexType, int64_t> columns,
                 raft::device_vector_view<ValueType, int64_t> values,
                 IndexType nnz,
                 ValueType* results,
                 bool bm25_on,
                 float k_param = 1.6f,
                 float b_param = 0.75f);
};

template <typename ValueType, typename IndexType>
SparseEncoder<ValueType, IndexType>::SparseEncoder(int vocab) : vocabSize(vocab)
{
  cudaMallocManaged(&featIdCount, vocab * sizeof(int));
  fullIdLen = 0.0f;
  numRows   = 0;
  for (int i = 0; i < vocabSize; i++) {
    featIdCount[i] = 0;
  }
}

template <typename ValueType, typename IndexType>
SparseEncoder<ValueType, IndexType>::SparseEncoder(std::map<int, int> featIdValues,
                                                   int num_rows,
                                                   int full_id_len,
                                                   int vocab_size)
  : vocabSize(vocab_size), numRows(num_rows), fullIdLen(full_id_len)
{
  cudaMallocManaged(&featIdCount, vocabSize * sizeof(int));
  cudaMemset(featIdCount, 0, vocabSize * sizeof(int));

  for (const auto& item : featIdValues) {
    featIdCount[item.first] = item.second;
  }
}

template <typename ValueType, typename IndexType>
SparseEncoder<ValueType, IndexType>::~SparseEncoder()
{
  cudaFree(featIdCount);
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::_fit_feats(IndexType* cols,
                                                     IndexType* counts,
                                                     IndexType nnz,
                                                     IndexType* results)
{
  int blockSize  = (nnz < 256) ? nnz : 256;
  int num_blocks = (nnz + blockSize - 1) / blockSize;
  raft::sparse::matrix::detail::_scan<<<blockSize, num_blocks>>>(cols, nnz, counts);
  raft::sparse::matrix::detail::_fit_compute_occurs<<<blockSize, num_blocks>>>(
    cols, nnz, counts, results, vocabSize);
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::_fit(raft::resources& handle,
                                               raft::device_vector_view<IndexType, int64_t> rows,
                                               raft::device_vector_view<IndexType, int64_t> columns,
                                               raft::device_vector_view<ValueType, int64_t> values,
                                               int num_rows)
{
  numRows += num_rows;
  IndexType nnz       = values.size();
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto batchIdLen     = raft::make_host_scalar<ValueType>(handle, 0);
  auto values_mat     = raft::make_device_scalar<ValueType>(handle, 0);
  raft::linalg::mapReduce<ValueType>(values_mat.data_handle(),
                                     nnz,
                                     0.0f,
                                     raft::identity_op(),
                                     raft::add_op(),
                                     stream,
                                     values.data_handle());
  raft::copy(batchIdLen.data_handle(), values_mat.data_handle(), values_mat.size(), stream);
  fullIdLen += (ValueType)batchIdLen(0);
  auto d_rows = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto d_cols = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto d_vals = raft::make_device_vector<ValueType, int64_t>(handle, nnz);
  raft::copy(d_rows.data_handle(), rows.data_handle(), nnz, stream);
  raft::copy(d_cols.data_handle(), columns.data_handle(), nnz, stream);
  raft::copy(d_vals.data_handle(), values.data_handle(), nnz, stream);
  raft::sparse::op::coo_sort(
    nnz, nnz, nnz, d_cols.data_handle(), d_rows.data_handle(), d_vals.data_handle(), stream);
  IndexType* counts;
  cudaMallocManaged(&counts, nnz * sizeof(IndexType));
  cudaMemset(counts, 0, nnz * sizeof(IndexType));
  _fit_feats(d_cols.data_handle(), counts, nnz, featIdCount);
  cudaFree(counts);
  cudaDeviceSynchronize();
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::fit(raft::resources& handle,
                                              raft::device_coo_matrix<ValueType,
                                                                      IndexType,
                                                                      IndexType,
                                                                      IndexType,
                                                                      raft::device_uvector_policy,
                                                                      raft::PRESERVING> coo_in)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto n_rows         = coo_in.structure_view().get_n_rows();
  auto rows           = coo_in.structure_view().get_rows();
  auto cols           = coo_in.structure_view().get_cols();
  auto vals           = coo_in.view().get_elements();
  auto nnz            = coo_in.structure_view().get_nnz();

  auto d_rows = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto d_cols = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto d_vals = raft::make_device_vector<ValueType, int64_t>(handle, nnz);

  raft::copy(d_rows.data_handle(), rows.data(), nnz, stream);
  raft::copy(d_cols.data_handle(), cols.data(), nnz, stream);
  raft::copy(d_vals.data_handle(), vals.data(), nnz, stream);

  _fit(handle, d_rows.view(), d_cols.view(), d_vals.view(), n_rows);
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::fit(raft::resources& handle,
                                              raft::device_csr_matrix<ValueType,
                                                                      IndexType,
                                                                      IndexType,
                                                                      IndexType,
                                                                      raft::device_uvector_policy,
                                                                      raft::PRESERVING> csr_in)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto nnz            = csr_in.structure_view().get_nnz();
  auto rows           = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto columns        = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto values         = raft::make_device_vector<ValueType, int64_t>(handle, nnz);

  raft::sparse::matrix::detail::convert_csr_to_coo(
    handle, csr_in, rows.view(), columns.view(), values.view());
  _fit(handle, rows.view(), columns.view(), values.view(), csr_in.structure_view().get_n_rows());
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::transform(
  raft::resources& handle,
  raft::device_coo_matrix<ValueType,
                          IndexType,
                          IndexType,
                          IndexType,
                          raft::device_uvector_policy,
                          raft::PRESERVING> coo_in,
  float* results,
  bool bm25_on,
  float k_param,
  float b_param)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto rows           = coo_in.structure_view().get_rows();
  auto cols           = coo_in.structure_view().get_cols();
  auto vals           = coo_in.view().get_elements();
  auto nnz            = coo_in.structure_view().get_nnz();

  auto d_rows = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto d_cols = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto d_vals = raft::make_device_vector<ValueType, int64_t>(handle, nnz);
  raft::copy(d_rows.data_handle(), rows.data(), nnz, stream);
  raft::copy(d_cols.data_handle(), cols.data(), nnz, stream);
  raft::copy(d_vals.data_handle(), vals.data(), nnz, stream);

  transform(
    handle, d_rows.view(), d_cols.view(), d_vals.view(), nnz, results, bm25_on, k_param, b_param);
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::transform(
  raft::resources& handle,
  raft::device_csr_matrix<ValueType,
                          IndexType,
                          IndexType,
                          IndexType,
                          raft::device_uvector_policy,
                          raft::PRESERVING> csr_in,
  float* results,
  bool bm25_on,
  float k_param,
  float b_param)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto nnz            = csr_in.structure_view().get_nnz();
  auto rows           = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto columns        = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  auto values         = raft::make_device_vector<ValueType, int64_t>(handle, nnz);
  raft::sparse::matrix::detail::convert_csr_to_coo(
    handle, csr_in, rows.view(), columns.view(), values.view());
  transform(
    handle, rows.view(), columns.view(), values.view(), nnz, results, bm25_on, k_param, b_param);
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::transform(
  raft::resources& handle,
  raft::device_vector_view<IndexType, int64_t> rows,
  raft::device_vector_view<IndexType, int64_t> columns,
  raft::device_vector_view<ValueType, int64_t> values,
  IndexType nnz,
  ValueType* results,
  bool bm25_on,
  float k_param,
  float b_param)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  int maxLimit   = nnz;
  int blockSize  = (maxLimit < 128) ? maxLimit : 128;
  int num_blocks = (maxLimit + blockSize - 1) / blockSize;
  float avgIdLen = (ValueType)fullIdLen / numRows;
  int* counts;
  cudaMallocManaged(&counts, maxLimit * sizeof(IndexType));
  cudaMemset(counts, 0, maxLimit * sizeof(IndexType));
  raft::sparse::op::coo_sort(IndexType(rows.size()),
                             IndexType(columns.size()),
                             IndexType(values.size()),
                             rows.data_handle(),
                             columns.data_handle(),
                             values.data_handle(),
                             stream);
  raft::sparse::matrix::detail::_scan<<<num_blocks, blockSize>>>(rows.data_handle(), nnz, counts);
  raft::sparse::matrix::detail::_transform<<<num_blocks, blockSize>>>(rows.data_handle(),
                                                                      columns.data_handle(),
                                                                      values.data_handle(),
                                                                      featIdCount,
                                                                      counts,
                                                                      results,
                                                                      numRows,
                                                                      avgIdLen,
                                                                      k_param,
                                                                      b_param,
                                                                      nnz,
                                                                      vocabSize,
                                                                      bm25_on);
  cudaFree(counts);
  cudaDeviceSynchronize();
}
}  // namespace raft::sparse::matrix
