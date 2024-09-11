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
#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/matrix/init.cuh>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/sparse/op/sort.cuh>

#include <thrust/extrema.h>
#include <thrust/reduce.h>

#include <iomanip>
#include <iostream>

namespace raft::sparse::matrix::detail {

/**
 * @brief Calculates the BM25 values for a target matrix.
 * @param num_feats: The total number of features in the matrix
 * @param avg_feat_len: The avg length of all features combined.
 * @param k_param: K value required by BM25 algorithm.
 * @param b_param: B value required by BM25 algorithm.
 */
template <typename T1, typename T2>
struct bm25 {
  bm25(T1 num_feats, T2 avg_feat_len, T2 k_param, T2 b_param)
  {
    total_feats     = num_feats;
    avg_feat_length = avg_feat_len;
    k               = k_param;
    b               = b_param;
  }

  float __device__ operator()(const T2& value, const T2& num_feats_id_occ, const T2& feat_length)
  {
    T2 tf  = T2(value / feat_length);
    T2 idf = raft::log<T2>(total_feats / num_feats_id_occ);
    T2 bm  = ((k + 1) * tf) / (k * ((1.0f - b) + b * (feat_length / avg_feat_length)) + tf);

    return idf * bm;
  }
  T2 avg_feat_length;
  T1 total_feats;
  T2 k;
  T2 b;
};

/**
 * @brief Calculates the tfidf values for a target matrix. Term frequency is calculate using
 * logrithmically scaled frequency.
 * @param total_feats_param: The total number of features in the matrix
 */
template <typename T1, typename T2>
struct tfidf {
  tfidf(T1 total_feats_param) { total_feats = total_feats_param; }

  float __device__ operator()(const T2& value, const T2& num_feats_id_occ, const T2& feat_length)
  {
    T2 tf  = T2(value / feat_length);
    T2 idf = raft::log<T2>(total_feats / num_feats_id_occ);
    return tf * idf;
  }
  T1 total_feats;
};

template <typename T>
struct mapper {
  mapper(raft::device_vector_view<T> map) : map(map) {}

  float __device__ operator()(const T& value)
  {
    T new_value = map[value];
    if (new_value) {
      return new_value;
    } else {
      return 0.0;
    }
  }

  raft::device_vector_view<const T> map;
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
void get_uniques_counts(raft::resources& handle,
                        raft::device_vector_view<T1, IdxT> sort_vector,
                        raft::device_vector_view<T1, IdxT> secondary_vector,
                        raft::device_vector_view<T2, IdxT> data,
                        raft::device_vector_view<T2, IdxT> itr_vals,
                        raft::device_vector_view<T1, IdxT> keys_out,
                        raft::device_vector_view<T2, IdxT> counts_out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::sparse::op::coo_sort(sort_vector.size(),
                             secondary_vector.size(),
                             data.size(),
                             sort_vector.data_handle(),
                             secondary_vector.data_handle(),
                             data.data_handle(),
                             stream);

  thrust::reduce_by_key(raft::resource::get_thrust_policy(handle),
                        sort_vector.data_handle(),
                        sort_vector.data_handle() + sort_vector.size(),
                        itr_vals.data_handle(),
                        keys_out.data_handle(),
                        counts_out.data_handle());
}

/**
 * @brief Compute cumulative sum for each unique value in the origin array
 * @param handle: raft resource handle
 * @param origin: Input array that has values to use for computation
 * @param keys: Output array that has keys, should be the size of unique
 * @param counts: Output array that contains the computed counts
 * @param results: Output array that scatters the counts to origin value positions. Same size as
 * origin array.
 */
template <typename T1, typename T2, typename IdxT>
void create_mapped_vector(raft::resources& handle,
                          const raft::device_vector_view<T1, IdxT> origin,
                          const raft::device_vector_view<T1, IdxT> keys,
                          const raft::device_vector_view<T2, IdxT> counts,
                          raft::device_vector_view<T2, IdxT> result)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  thrust::host_vector<T1> host_keys(keys.size());
  raft::copy(host_keys.data(), keys.data_handle(), keys.size(), stream);
  int key_size = *thrust::max_element(host_keys.begin(), host_keys.end());

  raft::linalg::map(handle, result, raft::cast_op<T2>{}, raft::make_const_mdspan(origin));
  // index into the last element and then add 1 to it.
  auto origin_map = raft::make_device_vector<T2, IdxT>(handle, key_size + 1);
  thrust::scatter(raft::resource::get_thrust_policy(handle),
                  counts.data_handle(),
                  counts.data_handle() + counts.size(),
                  keys.data_handle(),
                  origin_map.data_handle());

  raft::linalg::map(handle, result, mapper<T2>(origin_map.view()), raft::make_const_mdspan(result));
}

/**
 * @brief Compute row(id) counts
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param id_counts: Output array that stores counts per row, scattered to same shape as rows.
 * @param n_rows: Number of rows in matrix
 */
template <typename T1, typename T2, typename IdxT>
void get_id_counts(raft::resources& handle,
                   raft::device_vector_view<T1, IdxT> rows,
                   raft::device_vector_view<T1, IdxT> columns,
                   raft::device_vector_view<T2, IdxT> values,
                   raft::device_vector_view<T2, IdxT> id_counts,
                   T1 n_rows)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  // auto preserved_rows = raft::make_device_vector<T1, IdxT>(handle, rows.size());
  // raft::copy(preserved_rows.data_handle(), rows.data_handle(), rows.size(), stream);
  int uniq_rows =
    raft::sparse::neighbors::get_n_components(rows.data_handle(), rows.size(), stream);

  auto row_keys   = raft::make_device_vector<T1, IdxT>(handle, uniq_rows);
  auto row_counts = raft::make_device_vector<T2, IdxT>(handle, uniq_rows);
  auto row_fill   = raft::make_device_vector<T2, IdxT>(handle, n_rows);

  // the amount of columns(features) that each row(id) is found in
  raft::matrix::fill(handle, row_fill.view(), 1.0f);
  get_uniques_counts(
    handle, rows, columns, values, row_fill.view(), row_keys.view(), row_counts.view());
  create_mapped_vector<T1, T2>(handle, rows, row_keys.view(), row_counts.view(), id_counts);
}

/**
 * @brief Gather per feature mean values, returns the cumulative avg feature length.
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param feat_lengths: Output array that stores mean per feature value
 * @param n_cols: Number of columns in matrix
 */
template <typename T1, typename T2, typename IdxT>
float get_feature_data(raft::resources& handle,
                       raft::device_vector_view<T1, IdxT> rows,
                       raft::device_vector_view<T1, IdxT> columns,
                       raft::device_vector_view<T2, IdxT> values,
                       raft::device_vector_view<T2, IdxT> feat_lengths,
                       T1 n_cols)
{
  cudaStream_t stream    = raft::resource::get_cuda_stream(handle);
  auto preserved_columns = raft::make_device_vector<T1, IdxT>(handle, columns.size());
  raft::copy(preserved_columns.data_handle(), columns.data_handle(), columns.size(), stream);
  int uniq_cols =
    raft::sparse::neighbors::get_n_components(columns.data_handle(), columns.size(), stream);
  auto col_keys   = raft::make_device_vector<T1, IdxT>(handle, uniq_cols);
  auto col_counts = raft::make_device_vector<T2, IdxT>(handle, n_cols);

  get_uniques_counts(handle, columns, rows, values, values, col_keys.view(), col_counts.view());

  auto total_feature_lengths = raft::make_device_scalar<T1>(handle, 0);

  raft::linalg::mapReduce(total_feature_lengths.data_handle(),
                          col_counts.size(),
                          0,
                          raft::identity_op(),
                          raft::add_op(),
                          stream,
                          col_counts.data_handle());
  auto total_feature_lengths_host = raft::make_host_scalar<T1>(handle, 0);
  raft::copy(total_feature_lengths_host.data_handle(),
             total_feature_lengths.data_handle(),
             total_feature_lengths.size(),
             stream);
  T2 avg_feat_length = T2(total_feature_lengths_host(0)) / n_cols;
  create_mapped_vector<T1, T2>(
    handle, preserved_columns.view(), col_keys.view(), col_counts.view(), feat_lengths);
  return avg_feat_length;
}

/**
 * @brief Gather per feature mean values and id counts, returns the cumulative avg feature length.
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param feat_lengths: Output array that stores mean per feature value
 * @param id_counts: Output array that stores id(row) counts for nz values
 * @param n_rows: Number of rows in matrix
 * @param n_cols: Number of columns in matrix
 */
template <typename T1, typename T2, typename IdxT>
float sparse_search_preprocess(raft::resources& handle,
                               raft::device_vector_view<T1, IdxT> rows,
                               raft::device_vector_view<T1, IdxT> columns,
                               raft::device_vector_view<T2, IdxT> values,
                               raft::device_vector_view<T2, IdxT> feat_lengths,
                               raft::device_vector_view<T2, IdxT> id_counts,
                               T1 n_rows,
                               T1 n_cols)
{
  auto avg_feature_len = get_feature_data(handle, rows, columns, values, feat_lengths, n_cols);

  get_id_counts(handle, rows, columns, values, id_counts, n_rows);

  return avg_feature_len;
}

/**
 * @brief Use TFIDF algorithm to encode features in COO sparse matrix
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param values_out: Output COO values array
 * @param n_rows: Number of rows in matrix
 * @param n_cols: Number of columns in matrix
 */
template <typename T1, typename T2, typename IdxT>
void base_encode_tfidf(raft::resources& handle,
                       raft::device_vector_view<T1, IdxT> rows,
                       raft::device_vector_view<T1, IdxT> columns,
                       raft::device_vector_view<T2, IdxT> values,
                       raft::device_vector_view<T2, IdxT> values_out,
                       T1 n_rows,
                       T1 n_cols)
{
  auto feat_lengths    = raft::make_device_vector<T2, IdxT>(handle, columns.size());
  auto id_counts       = raft::make_device_vector<T2, IdxT>(handle, rows.size());
  auto col_counts      = raft::make_device_vector<T2, IdxT>(handle, n_cols);
  auto avg_feat_length = sparse_search_preprocess<T1, T2>(
    handle, rows, columns, values, feat_lengths.view(), id_counts.view(), n_rows, n_cols);

  raft::linalg::map(handle,
                    values_out,
                    tfidf<T1, T2>(n_cols),
                    raft::make_const_mdspan(values),
                    raft::make_const_mdspan(id_counts.view()),
                    raft::make_const_mdspan(feat_lengths.view()));
}

/**
 * @brief Use TFIDF algorithm to encode features in COO sparse matrix
 * @param handle: raft resource handle
 * @param coo_in: Input COO matrix
 * @param values_out: Output COO values array
 */
template <typename T1, typename T2, typename IdxT>
void encode_tfidf(raft::resources& handle,
                  raft::device_coo_matrix_view<T2, T1, T1, T1> coo_in,
                  raft::device_vector_view<T2, IdxT> values_out)
{
  auto rows    = raft::make_device_vector_view<T1, IdxT>(coo_in.structure_view().get_rows().data(),
                                                      coo_in.structure_view().get_rows().size());
  auto columns = raft::make_device_vector_view<T1, IdxT>(coo_in.structure_view().get_cols().data(),
                                                         coo_in.structure_view().get_cols().size());
  auto values  = raft::make_device_vector_view<T2, IdxT>(coo_in.get_elements().data(),
                                                        coo_in.get_elements().size());

  base_encode_tfidf<T1, T2, IdxT>(handle,
                                  rows,
                                  columns,
                                  values,
                                  values_out,
                                  coo_in.structure_view().get_n_rows(),
                                  coo_in.structure_view().get_n_cols());
}

/**
 * @brief Use TFIDF algorithm to encode features in CSR sparse matrix
 * @param handle: raft resource handle
 * @param csr_in: Input CSR matrix
 * @param values_out: Output values array
 */
template <typename T1, typename T2, typename IdxT>
void encode_tfidf(raft::resources& handle,
                  raft::device_csr_matrix_view<T2, T1, T1, T1> csr_in,
                  raft::device_vector_view<T2, IdxT> values_out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto indptr = raft::make_device_vector_view<T1, IdxT>(
    csr_in.structure_view().get_indptr().data(), csr_in.structure_view().get_indptr().size());
  auto indices = raft::make_device_vector_view<T1, IdxT>(
    csr_in.structure_view().get_indices().data(), csr_in.structure_view().get_indices().size());
  auto values = raft::make_device_vector_view<T2, IdxT>(csr_in.get_elements().data(),
                                                        csr_in.get_elements().size());

  auto rows = raft::make_device_vector<T1, IdxT>(handle, values.size());
  raft::sparse::convert::csr_to_coo<T1>(indptr.data_handle(),
                                        csr_in.structure_view().get_n_rows(),
                                        rows.data_handle(),
                                        rows.size(),
                                        stream);

  base_encode_tfidf<T1, T2, IdxT>(handle,
                                  rows.view(),
                                  indices,
                                  values,
                                  values_out,
                                  csr_in.structure_view().get_n_rows(),
                                  csr_in.structure_view().get_n_cols());
}

/**
 * @brief Use BM25 algorithm to encode features in COO sparse matrix
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param values_out: Output COO values array
 * @param n_rows: Number of rows in matrix
 * @param n_cols: Number of columns in matrix
 * @param k_param: K value to use for BM25 algorithm
 * @param b_param: B value to use for BM25 algorithm
 */
template <typename T1, typename T2, typename IdxT>
void base_encode_bm25(raft::resources& handle,
                      raft::device_vector_view<T1, IdxT> rows,
                      raft::device_vector_view<T1, IdxT> columns,
                      raft::device_vector_view<T2, IdxT> values,
                      raft::device_vector_view<T2, IdxT> values_out,
                      T1 n_rows,
                      T1 n_cols,
                      T2 k_param = 1.6f,
                      T2 b_param = 0.75f)
{
  auto feat_lengths = raft::make_device_vector<T2, IdxT>(handle, columns.size());
  auto id_counts    = raft::make_device_vector<T2, IdxT>(handle, rows.size());
  auto col_counts   = raft::make_device_vector<T2, IdxT>(handle, n_cols);

  auto avg_feat_length = sparse_search_preprocess<T1, T2>(
    handle, rows, columns, values, feat_lengths.view(), id_counts.view(), n_rows, n_cols);

  raft::linalg::map(handle,
                    values_out,
                    bm25<T1, T2>(n_cols, avg_feat_length, k_param, b_param),
                    raft::make_const_mdspan(values),
                    raft::make_const_mdspan(id_counts.view()),
                    raft::make_const_mdspan(feat_lengths.view()));
}

/**
 * @brief Use BM25 algorithm to encode features in COO sparse matrix
 * @param handle: raft resource handle
 * @param coo_in: Input COO matrix
 * @param values_out: Output values array
 * @param k_param: K value to use for BM25 algorithm
 * @param b_param: B value to use for BM25 algorithm
 */
template <typename T1, typename T2, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_coo_matrix_view<T2, T1, T1, T1> coo_in,
                 raft::device_vector_view<T2, IdxT> values_out,
                 T2 k_param = 1.6f,
                 T2 b_param = 0.75f)
{
  auto rows    = raft::make_device_vector_view<T1, IdxT>(coo_in.structure_view().get_rows().data(),
                                                      coo_in.structure_view().get_rows().size());
  auto columns = raft::make_device_vector_view<T1, IdxT>(coo_in.structure_view().get_cols().data(),
                                                         coo_in.structure_view().get_cols().size());
  auto values  = raft::make_device_vector_view<T2, IdxT>(coo_in.get_elements().data(),
                                                        coo_in.get_elements().size());

  base_encode_bm25<T1, T2, IdxT>(handle,
                                 rows,
                                 columns,
                                 values,
                                 values_out,
                                 coo_in.structure_view().get_n_rows(),
                                 coo_in.structure_view().get_n_cols());
}

/**
 * @brief Use BM25 algorithm to encode features in CSR sparse matrix
 * @param handle: raft resource handle
 * @param csr_in: Input CSR matrix
 * @param values_out: Output values array
 * @param k_param: K value to use for BM25 algorithm
 * @param b_param: B value to use for BM25 algorithm
 */
template <typename T1, typename T2, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_csr_matrix_view<T2, T1, T1, T1> csr_in,
                 raft::device_vector_view<T2, IdxT> values_out,
                 T2 k_param = 1.6f,
                 T2 b_param = 0.75f)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto indptr = raft::make_device_vector_view<T1, IdxT>(
    csr_in.structure_view().get_indptr().data(), csr_in.structure_view().get_indptr().size());
  auto indices = raft::make_device_vector_view<T1, IdxT>(
    csr_in.structure_view().get_indices().data(), csr_in.structure_view().get_indices().size());
  auto values = raft::make_device_vector_view<T2, IdxT>(csr_in.get_elements().data(),
                                                        csr_in.get_elements().size());

  auto rows = raft::make_device_vector<T1, IdxT>(handle, values.size());

  raft::sparse::convert::csr_to_coo<T1>(indptr.data_handle(),
                                        csr_in.structure_view().get_n_rows(),
                                        rows.data_handle(),
                                        rows.size(),
                                        stream);

  base_encode_bm25<T1, T2, IdxT>(handle,
                                 rows.view(),
                                 indices,
                                 values,
                                 values_out,
                                 csr_in.structure_view().get_n_rows(),
                                 csr_in.structure_view().get_n_cols());
}

}  // namespace raft::sparse::matrix::detail