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

__global__ void _scan(int* rows, int nnz, int* counts)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= nnz) { return; }
  if (index == 0) {
    counts[index] = 1;
    return;
  }
  if (index < nnz) {
    int curr_id = rows[index];
    int old_id  = rows[index - 1];
    if (curr_id != old_id) {
      counts[index] = 1;
    } else {
      counts[index] = 0;
    }
  }
}

__global__ void _fit_compute_occurs(int* cols, int nnz, int* counts, int* feats)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if ((index < nnz) && (counts[index] == 1)) {
    int targetVal = cols[index];
    while (targetVal == cols[index]) {
      feats[targetVal] = feats[targetVal] + 1;
      index++;
      if (index >= nnz) { return; }
    }
  }
}

__global__ void _transform(int* rows,
                           int* columns,
                           float* values,
                           int* feat_id_count,
                           int* counts,
                           float* out_values,
                           int num_rows,
                           float avgRowLen,
                           float k,
                           float b,
                           int nnz,
                           bool bm25 = false)
{
  int start_index = blockIdx.x * blockDim.x + threadIdx.x;
  int index       = start_index;
  if (index < nnz && counts[index] == 1) {
    int row_length = 0;
    int targetVal  = rows[index];
    while (targetVal == rows[index]) {
      row_length = row_length + values[index];
      index++;
      if (index >= nnz) { break; }
    }
    index = start_index;
    float result;
    while (targetVal == rows[index]) {
      int vocab     = columns[index];
      float tf      = (float)values[index] / row_length;
      double idf_in = (double)num_rows / feat_id_count[vocab];
      float idf     = (float)raft::log<double>(idf_in);
      result        = tf * idf;
      if (bm25) {
        float bm = ((k + 1) * tf) / (k * ((1.0f - b) + b * (row_length / avgRowLen)) + tf);
        result   = idf * bm;
      }
      out_values[index] = result;
      index++;
      if (index >= nnz) { break; }
    }
  }
}

template <typename ValueType, typename IndexType>
void convert_csr_to_coo(raft::resources& handle,
                        raft::device_csr_matrix<ValueType,
                                                IndexType,
                                                IndexType,
                                                IndexType,
                                                raft::device_uvector_policy,
                                                raft::PRESERVING> csr_in,
                        raft::device_vector_view<IndexType, int64_t> rows,
                        raft::device_vector_view<IndexType, int64_t> columns,
                        raft::device_vector_view<ValueType, int64_t> values)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto nnz            = csr_in.structure_view().get_nnz();
  auto indptr         = csr_in.structure_view().get_indptr();
  auto indices        = csr_in.structure_view().get_indices();
  auto vals           = csr_in.view().get_elements();

  raft::sparse::convert::csr_to_coo(
    indptr.data(), (int)indptr.size(), rows.data_handle(), (int)nnz, stream);
  raft::copy(columns.data_handle(), indices.data(), (int)nnz, stream);
  raft::copy(values.data_handle(), vals.data(), (int)nnz, stream);
}

class SparseEncoder {
 private:
  int* featIdCount;
  float fullIdLen;
  int vocabSize;
  int numRows;

 public:
  SparseEncoder(int vocab) : vocabSize(vocab)
  {
    cudaMallocManaged(&featIdCount, vocab * sizeof(int));
    fullIdLen = 0.0f;
    numRows   = 0;
    for (int i = 0; i < vocabSize; i++) {
      featIdCount[i] = 0;
    }
  }

  ~SparseEncoder() { cudaFree(featIdCount); }

  template <typename IndexType = int>
  void _fit_feats(IndexType* cols, IndexType* counts, IndexType nnz, IndexType* results)
  {
    int blockSize  = (nnz < 256) ? nnz : 256;
    int num_blocks = (nnz + blockSize - 1) / blockSize;
    _scan<<<blockSize, num_blocks>>>(cols, nnz, counts);
    _fit_compute_occurs<<<blockSize, num_blocks>>>(cols, nnz, counts, results);
  }

  template <typename ValueType = float, typename IndexType = int>
  void _fit(raft::resources& handle,
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

  template <typename ValueType = float, typename IndexType = int>
  void fit(raft::resources& handle,
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

    _fit<ValueType, IndexType>(handle, d_rows.view(), d_cols.view(), d_vals.view(), n_rows);
  }

  template <typename ValueType = float, typename IndexType = int>
  void fit(raft::resources& handle,
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

    convert_csr_to_coo(handle, csr_in, rows.view(), columns.view(), values.view());
    _fit<ValueType, IndexType>(
      handle, rows.view(), columns.view(), values.view(), csr_in.structure_view().get_n_rows());
  }

  template <typename ValueType = float, typename IndexType = int>
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
                 float b_param = 0.75f)
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

  template <typename ValueType = float, typename IndexType = int>
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
                 float b_param = 0.75f)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    auto nnz            = csr_in.structure_view().get_nnz();
    auto rows           = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
    auto columns        = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
    auto values         = raft::make_device_vector<ValueType, int64_t>(handle, nnz);
    convert_csr_to_coo(handle, csr_in, rows.view(), columns.view(), values.view());
    transform<ValueType, IndexType>(
      handle, rows.view(), columns.view(), values.view(), nnz, results, bm25_on, k_param, b_param);
  }

  template <typename ValueType = float, typename IndexType = int>
  void transform(raft::resources& handle,
                 raft::device_vector_view<IndexType, int64_t> rows,
                 raft::device_vector_view<IndexType, int64_t> columns,
                 raft::device_vector_view<ValueType, int64_t> values,
                 IndexType nnz,
                 ValueType* results,
                 bool bm25_on,
                 float k_param = 1.6f,
                 float b_param = 0.75f)
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
    _scan<<<num_blocks, blockSize>>>(rows.data_handle(), nnz, counts);
    _transform<<<num_blocks, blockSize>>>(rows.data_handle(),
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
                                          bm25_on);
    cudaFree(counts);
    cudaDeviceSynchronize();
  }
};
}  // namespace raft::sparse::matrix::detail