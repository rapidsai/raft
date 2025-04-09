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

#include <raft/cluster/detail/kmeans_common.cuh>
#include <raft/core/device_csr_matrix.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/label/detail/classlabels.cuh>
#include <raft/sparse/matrix/detail/preprocessing.cuh>

#include <map>
#include <optional>

namespace raft::sparse::matrix {

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

template <typename IndexType, typename ValueType>
struct tfidf {
  tfidf(IndexType numRows, raft::device_vector_view<IndexType> featIdCount)
    : num_rows(numRows), feat_id_count(featIdCount)
  {
  }

  __host__ __device__ ValueType operator()(const IndexType& col, const ValueType& value)
  {
    ValueType tf  = (ValueType)raft::log<double>(value);
    double idf_in = (double)num_rows / feat_id_count[col];
    ValueType idf = (ValueType)raft::log<double>(idf_in + 1);
    return (ValueType)tf * idf;
  }

  raft::device_vector_view<IndexType> feat_id_count;
  IndexType num_rows;
};

template <typename IndexType, typename ValueType>
struct bm25 {
  bm25(IndexType numRows,
       raft::device_vector_view<IndexType> featIdCount,
       raft::device_vector_view<IndexType> rowFeatCnts,
       ValueType avg_feat_len,
       ValueType k_param,
       ValueType b_param)
  {
    num_rows        = numRows;
    feat_id_count   = featIdCount;
    row_feat_cnts   = rowFeatCnts;
    avg_feat_length = avg_feat_len;
    k               = k_param;
    b               = b_param;
  }
  __host__ __device__ ValueType operator()(const IndexType& row,
                                           const IndexType& column,
                                           const ValueType& value)
  {
    ValueType tf  = raft::log<ValueType>(value);
    double idf_in = (double)num_rows / feat_id_count[column];
    ValueType idf = (ValueType)raft::log<double>(idf_in + 1);
    ValueType bm =
      ((k + 1) * tf) / (k * ((1.0f - b) + b * (row_feat_cnts[row] / avg_feat_length)) + tf);

    return (ValueType)idf * bm;
  }
  raft::device_vector_view<IndexType> row_feat_cnts;
  raft::device_vector_view<IndexType> feat_id_count;
  ValueType avg_feat_length;
  IndexType num_rows;
  ValueType k;
  ValueType b;
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
template <typename T1, typename T2, typename IdxT>
void get_uniques_counts(raft::resources const& handle,
                        raft::device_vector_view<T1, IdxT> sort_vector,
                        raft::device_vector_view<T1, IdxT> secondary_vector,
                        raft::device_vector_view<T2, IdxT> data,
                        raft::device_vector_view<T2, IdxT> itr_vals,
                        raft::device_vector_view<T1, IdxT> keys_out,
                        raft::device_vector_view<T2, IdxT> counts_out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::sparse::op::coo_sort(int(sort_vector.size()),
                             int(secondary_vector.size()),
                             int(data.size()),
                             sort_vector.data_handle(),
                             secondary_vector.data_handle(),
                             data.data_handle(),
                             stream);
  // replace this call with raft version when available
  // (https://github.com/rapidsai/raft/issues/2477)
  thrust::reduce_by_key(raft::resource::get_thrust_policy(handle),
                        sort_vector.data_handle(),
                        sort_vector.data_handle() + sort_vector.size(),
                        itr_vals.data_handle(),
                        keys_out.data_handle(),
                        counts_out.data_handle());
}

template <typename ValueType = float, typename IndexType = int>
void _fit(raft::resources const& handle,
          raft::device_vector_view<IndexType, int64_t> rows,
          raft::device_vector_view<IndexType, int64_t> columns,
          raft::device_vector_view<ValueType, int64_t> values,
          int num_rows,
          int num_cols,
          raft::device_vector_view<IndexType, int64_t> idFeatCount,
          int& fullFeatCount,
          raft::device_vector_view<IndexType, int64_t> rowFeatCnts)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int uniq_cnt = raft::sparse::neighbors::get_n_components(rows.data_handle(), rows.size(), stream);
  auto row_keys = raft::make_device_vector<IndexType>(handle, uniq_cnt);
  auto row_cnts = raft::make_device_vector<ValueType>(handle, uniq_cnt);

  get_uniques_counts<IndexType, ValueType, int64_t>(
    handle, rows, columns, values, values, row_keys.view(), row_cnts.view());
  auto dummy_vec = raft::make_device_vector<IndexType>(handle, uniq_cnt);
  raft::linalg::map(handle,
                    dummy_vec.view(),
                    mapper<IndexType, ValueType>(rowFeatCnts.data_handle()),
                    raft::make_const_mdspan(row_keys.view()),
                    raft::make_const_mdspan(row_cnts.view()));

  rmm::device_uvector<char> workspace(0, stream);
  raft::cluster::detail::countLabels(handle,
                                     columns.data_handle(),
                                     idFeatCount.data_handle(),
                                     int(columns.size()),
                                     num_cols,
                                     workspace);

  // get total number of words
  auto batchIdLen = raft::make_host_scalar<ValueType>(0);
  auto values_mat = raft::make_device_scalar<ValueType>(handle, 0);
  raft::linalg::mapReduce<ValueType>(values_mat.data_handle(),
                                     values.size(),
                                     0.0f,
                                     raft::identity_op(),
                                     raft::add_op(),
                                     stream,
                                     values.data_handle());
  raft::copy(batchIdLen.data_handle(), values_mat.data_handle(), values_mat.size(), stream);
  fullFeatCount += (int)batchIdLen(0);
}

template <typename IndexType, typename ValueType>
void transform(raft::resources const& handle,
               raft::device_vector_view<IndexType, int64_t> rows,
               raft::device_vector_view<IndexType, int64_t> columns,
               raft::device_vector_view<ValueType, int64_t> values,
               raft::device_vector_view<IndexType, int64_t> featIdCount,
               IndexType fullIdLen,
               IndexType nnz,
               int num_rows,
               int num_feats,
               raft::device_vector_view<ValueType, int64_t> results,
               bool bm25_on,
               float k_param,
               float b_param,
               raft::device_vector_view<IndexType, int64_t> rowFeatCnts)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  float avgIdLen = (ValueType)fullIdLen / num_rows;
  raft::sparse::op::coo_sort(IndexType(rows.size()),
                             IndexType(columns.size()),
                             IndexType(values.size()),
                             rows.data_handle(),
                             columns.data_handle(),
                             values.data_handle(),
                             stream);
  if (bm25_on) {
    // raft::linalg::map(handle,
    //   results,
    //   tfidf<IndexType, ValueType>(num_rows, featIdCount),
    //   raft::make_const_mdspan(columns),
    //   raft::make_const_mdspan(values)
    // );
    raft::linalg::map(
      handle,
      results,
      bm25<IndexType, ValueType>(num_rows, featIdCount, rowFeatCnts, avgIdLen, k_param, b_param),
      raft::make_const_mdspan(rows),
      raft::make_const_mdspan(columns),
      raft::make_const_mdspan(values));
  } else {
    raft::linalg::map(handle,
                      results,
                      tfidf<IndexType, ValueType>(num_rows, featIdCount),
                      raft::make_const_mdspan(columns),
                      raft::make_const_mdspan(values));
  }
}

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
  IndexType* featIdCount;
  IndexType fullIdLen;
  IndexType numFeats;
  IndexType numRows;

 public:
  SparseEncoder(int num_feats);
  SparseEncoder(
    raft::resources const& handle, int* featIdValues, int num_rows, int full_id_len, int num_feats);
  ~SparseEncoder();
  void fit(raft::resources const& handle,
           raft::device_coo_matrix<ValueType,
                                   IndexType,
                                   IndexType,
                                   IndexType,
                                   raft::device_uvector_policy,
                                   raft::PRESERVING> coo_in);
  void save(raft::resources const& handle, std::string save_path);
  void fit(raft::resources const& handle,
           raft::device_csr_matrix<ValueType,
                                   IndexType,
                                   IndexType,
                                   IndexType,
                                   raft::device_uvector_policy,
                                   raft::PRESERVING> csr_in);
  void transform(raft::resources const& handle,
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
  void transform(raft::resources const& handle,
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
  void _fit(raft::resources const& handle,
            raft::device_vector_view<IndexType, int64_t> rows,
            raft::device_vector_view<IndexType, int64_t> columns,
            raft::device_vector_view<ValueType, int64_t> values,
            int num_rows);
  void _fit_feats(raft::resources const& handle,
                  IndexType* cols,
                  IndexType* counts,
                  IndexType nnz,
                  IndexType* results);
  void transform(raft::resources const& handle,
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
  cudaMallocManaged(&featIdCount, feats * sizeof(IndexType));
  cudaMemset(featIdCount, 0, numFeats * sizeof(IndexType));
  fullIdLen = 0;
  numRows   = 0;
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
  raft::resources const& handle, int* featIdValues, int num_feats, int num_rows, int full_id_len)
  : numFeats(num_feats), numRows(num_rows), fullIdLen(full_id_len)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  cudaMallocManaged(&featIdCount, numFeats * sizeof(IndexType));

  cudaMemset(featIdCount, 0, numFeats * sizeof(IndexType));
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
void SparseEncoder<ValueType, IndexType>::save(raft::resources const& handle, std::string save_path)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto featIdCount_md = raft::make_device_vector<IndexType, int64_t>(handle, numFeats);
  raft::copy(featIdCount_md.data_handle(), featIdCount, numFeats, stream);
  std::ofstream saveFile(save_path);
  raft::serialize_scalar<IndexType>(handle, saveFile, numFeats);
  raft::serialize_scalar<IndexType>(handle, saveFile, numRows);
  raft::serialize_scalar<IndexType>(handle, saveFile, fullIdLen);
  raft::serialize_mdspan<IndexType>(handle, saveFile, featIdCount_md.view());
}

template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::_fit_feats(raft::resources const& handle,
                                                     IndexType* cols,
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
void SparseEncoder<ValueType, IndexType>::_fit(raft::resources const& handle,
                                               raft::device_vector_view<IndexType, int64_t> rows,
                                               raft::device_vector_view<IndexType, int64_t> columns,
                                               raft::device_vector_view<ValueType, int64_t> values,
                                               int num_rows)
{
  // numRows += num_rows;
  // IndexType nnz       = values.size();
  // cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // auto batchIdLen     = raft::make_host_scalar<ValueType>(0);
  // auto values_mat     = raft::make_device_scalar<ValueType>(handle, 0);
  // raft::linalg::mapReduce<ValueType>(values_mat.data_handle(),
  //                                    nnz,
  //                                    0.0f,
  //                                    raft::identity_op(),
  //                                    raft::add_op(),
  //                                    stream,
  //                                    values.data_handle());
  // raft::copy(batchIdLen.data_handle(), values_mat.data_handle(), values_mat.size(), stream);
  // fullIdLen += (int)batchIdLen(0);
  // auto d_rows = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  // auto d_cols = raft::make_device_vector<IndexType, int64_t>(handle, nnz);
  // auto d_vals = raft::make_device_vector<ValueType, int64_t>(handle, nnz);
  // raft::copy(d_rows.data_handle(), rows.data_handle(), nnz, stream);
  // raft::copy(d_cols.data_handle(), columns.data_handle(), nnz, stream);
  // raft::copy(d_vals.data_handle(), values.data_handle(), nnz, stream);
  // raft::sparse::op::coo_sort(
  //   nnz, nnz, nnz, d_cols.data_handle(), d_rows.data_handle(), d_vals.data_handle(), stream);
  // IndexType* counts;
  // cudaMallocManaged(&counts, nnz * sizeof(IndexType));
  // cudaMemset(counts, 0, nnz * sizeof(IndexType));
  // _fit_feats(handle, d_cols.data_handle(), counts, nnz, featIdCount);
  // cudaFree(counts);
  // cudaDeviceSynchronize();
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
void SparseEncoder<ValueType, IndexType>::fit(raft::resources const& handle,
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
void SparseEncoder<ValueType, IndexType>::fit(raft::resources const& handle,
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
 * @param[in] results
 *   array that is the size of the coo_in nnz that will store the encoded
 *      transform values.
 * @param[in] bm25_on
 *   When true calculates bm25 encoding values, otherwise tfidf.
 * @param[in] k_param
 *   bm25 okapi optimization parameter k1
 * @param[in] b_param
 *   bm25 okapi optimization parameter b
 * */
template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::transform(
  raft::resources const& handle,
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
 * @param[in] results
 *   array that is the size of the csr_in nnz that will store the encoded
 *      transform values.
 * @param[in] bm25_on
 *   When true calculates bm25 encoding values, otherwise tfidf.
 * @param[in] k_param
 *   bm25 okapi optimization parameter k1
 * @param[in] b_param
 *   bm25 okapi optimization parameter b
 * */
template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::transform(
  raft::resources const& handle,
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
 * @param[in] rows
 *   array coo representation of rows for non zero values.
 * @param[in] columns
 *   array coo representation of columns for non zero values.
 * @param[in] values
 *   array coo representation of non zero values.
 * @param[in] nnz
 *   The number of non-zero values in the coo array representations.
 * @param[in] results
 *   array that is the size of the csr_in nnz that will store the encoded
 *      transform values.
 * @param[in] bm25_on
 *   When true calculates bm25 encoding values, otherwise tfidf.
 * @param[in] k_param
 *   bm25 okapi optimization parameter k1
 * @param[in] b_param
 *   bm25 okapi optimization parameter b
 * */
template <typename ValueType, typename IndexType>
void SparseEncoder<ValueType, IndexType>::transform(
  raft::resources const& handle,
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
SparseEncoder<ValueType, IndexType>* loadSparseEncoder(raft::resources const& handle,
                                                       std::string save_path)
{
  std::ifstream loadFile(save_path, std::ios_base::in);
  auto num_feats = deserialize_scalar<IndexType>(handle, loadFile);
  auto num_rows  = deserialize_scalar<IndexType>(handle, loadFile);
  auto fullIdLen = deserialize_scalar<IndexType>(handle, loadFile);

  auto featIdCount_md = raft::make_host_vector<IndexType, int64_t>(num_feats);
  deserialize_mdspan<IndexType>(handle, loadFile, featIdCount_md.view());

  return new SparseEncoder<ValueType, IndexType>(
    handle, featIdCount_md.data_handle(), num_feats, num_rows, fullIdLen);
}

}  // namespace raft::sparse::matrix
