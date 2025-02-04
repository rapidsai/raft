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

/**
 * The class facilitates the creation a tfidf and bm25 encoding values for sparse matrices
 *
 * This class creates tfidf and bm25 encoding values for sparse matrices. It allows for
 * batched matrix processing by calling fit for all matrix chunks. Once all matrices have
 * been fitted, the user can use transform to actually produce the encoded values for each
 * subset (chunk) matrix.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] featIdCount
 *   An array that holds the count of how many different rows each feature occurs in.
 * @param[in] fullIdLen
 *   The value that represents the total number of words seen during the fit process.
 * @param[out] numFeats
 *   A value that represents the number of features that exist for the matrices encoded.
 * @param[out] numRows
 *   The number of rows observed during the fit process, accumulates over all fit calls.
 */
template <typename ValueType = float, typename IndexType = int>
class SparseEncoder {
 private:
  int* featIdCount;
  int fullIdLen;
  int numFeats;
  int numRows;

 public:
  SparseEncoder(int num_feats);
  SparseEncoder(
    raft::resources& handle, int* featIdValues, int num_rows, int full_id_len, int num_feats);
  ~SparseEncoder();
  void fit(raft::resources& handle,
           raft::device_coo_matrix<ValueType,
                                   IndexType,
                                   IndexType,
                                   IndexType,
                                   raft::device_uvector_policy,
                                   raft::PRESERVING> coo_in);
  void save(raft::resources& handle, std::string save_path);
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

/**
 * This constructor creates the `SparseEncoder` class with a numFeats equal to the
 * int feats parameter supplied.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] feats
 *   Value that represents the number of features that exist for the matrices encoded.
 */
template <typename ValueType, typename IndexType>
SparseEncoder<ValueType, IndexType>::SparseEncoder(int feats) : numFeats(feats)
{
  cudaMallocManaged(&featIdCount, feats * sizeof(int));
  fullIdLen = 0;
  numRows   = 0;
  for (int i = 0; i < numFeats; i++) {
    featIdCount[i] = 0;
  }
}
/**
 * This constructor creates the `SparseEncoder` class with a numFeats equal to the
 * int feats parameter supplied.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] featIdValues
 *   A map that consists of all the indices and values, to populate the featIdCount array.
 * @param[in] num_rows
 *   Value that represents the number of rows observed during fit cycle.
 * @param[in] full_id_len
 *   Value that represents the number overall number of features observed during the fit
 * cycle.
 * @param[in] num_feats
 *   Value that represents the number of features that exist for the matrices encoded.
 * */
template <typename ValueType, typename IndexType>
SparseEncoder<ValueType, IndexType>::SparseEncoder(
  raft::resources& handle, int* featIdValues, int num_feats, int num_rows, int full_id_len)
  : numFeats(num_feats), numRows(num_rows), fullIdLen(full_id_len)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  cudaMallocManaged(&featIdCount, numFeats * sizeof(int));
  cudaMemset(featIdCount, 0, numFeats * sizeof(int));
  for (int i = 0; i < numFeats; i++) {
    featIdCount[i] = featIdValues[i];
  }
}

/**
 * This destructor deallocates/frees the reserved memory of the class.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 * */
template <typename ValueType, typename IndexType>
SparseEncoder<ValueType, IndexType>::~SparseEncoder()
{
  cudaFree(featIdCount);
}

/**
 * This function exports all values required to recreate the SparseEncoder to a
 * file.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] save_path
 *   The path to save the file container all values required to recreate SparseEncoder.
 * */
template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::save(raft::resources& handle, std::string save_path)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto featIdCount_md = raft::make_device_vector<IndexType, int64_t>(handle, numFeats);
  raft::copy(featIdCount_md.data_handle(), featIdCount, numFeats, stream);
  std::ofstream saveFile(save_path);
  if (saveFile.is_open()) {
    std::ostringstream oss;
    saveFile << numFeats << " ";
    saveFile << numRows << " ";
    saveFile << fullIdLen << " ";
    for (int i = 0; i < numFeats; i++) {
      saveFile << featIdCount[i] << " ";
    }
    saveFile.close();
  }
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
    cols, nnz, counts, results, numFeats);
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
  fullIdLen += (int)batchIdLen(0);
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

/**
 * This function fits the input matrix, recording required statistics to later create
 * encoding values.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] coo_in
 *   Raft container housing a coordinate format sparse matrix representation.

 * */
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

/**
 * This function fits the input matrix, recording required statistics to later create
 * encoding values.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] csr_in
 *   Raft container housing a compressed sparse row matrix representation.

 * */
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
/**
 * This function transforms the coo matrix based on statistics collected during fit
 * cycle.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] coo_in
 *   Raft container housing a compressed sparse row matrix representation.

 * */
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

/**
 * This function transforms the csr matrix based on statistics collected during fit
 * cycle.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] csr_in
 *   Raft container housing a compressed sparse row matrix representation.

 * */
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
                                                                      numFeats,
                                                                      bm25_on);
  cudaFree(counts);
  cudaDeviceSynchronize();
}

/**
 * This function loads a sparse encoder from a previously saved file path.
 *
 * @tparam ValueType
 *   Type of the values in the sparse matrix.
 * @tparam IndexType
 *   Type of the indices associated with the values.
 *
 * @param[in] handle
 *   Container for managing reusable resources.
 * @param[in] save_path
 *   The path with the saved SparseEncoder data.
 * */
template <typename ValueType, typename IndexType>
SparseEncoder<ValueType, IndexType>* loadSparseEncoder(raft::resources& handle,
                                                       std::string save_path)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  std::ifstream loadFile(save_path, std::ios_base::in);
  IndexType num_feats, num_rows, fullIdLen;
  loadFile >> num_feats;
  loadFile >> num_rows;
  loadFile >> fullIdLen;
  IndexType val;
  std::vector<IndexType> vals;
  while (loadFile >> val) {
    vals.push_back(val);
  }
  auto featIdCount_h = raft::make_host_vector<IndexType, int64_t>(handle, num_feats);
  raft::copy(featIdCount_h.data_handle(), vals.data(), vals.size(), stream);
  loadFile.close();
  return new SparseEncoder<ValueType, IndexType>(
    handle, featIdCount_h.data_handle(), num_feats, num_rows, fullIdLen);
}

}  // namespace raft::sparse::matrix
