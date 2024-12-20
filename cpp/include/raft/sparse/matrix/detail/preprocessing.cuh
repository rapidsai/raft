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

#include <thrust/reduce.h>

namespace raft::sparse::matrix::detail {

/**
 * @brief Calculates the BM25 values for a target matrix.
 * @param num_feats: The total number of features in the matrix
 * @param avg_feat_len: The avg length of all features combined.
 * @param k_param: K value required by BM25 algorithm.
 * @param b_param: B value required by BM25 algorithm.
 */
template <typename IndexType, typename ValueType>
struct bm25 {
  bm25(IndexType num_feats, ValueType avg_feat_len, ValueType k_param, ValueType b_param)
  {
    total_feats     = num_feats;
    avg_feat_length = avg_feat_len;
    k               = k_param;
    b               = b_param;
  }

  float __device__ operator()(const ValueType& value,
                              const ValueType& num_feats_id_occ,
                              const ValueType& feat_length)
  {
    ValueType tf  = ValueType(value / feat_length);
    ValueType idf = raft::log<ValueType>(total_feats / num_feats_id_occ);
    ValueType bm  = ((k + 1) * tf) / (k * ((1.0f - b) + b * (feat_length / avg_feat_length)) + tf);

    return idf * bm;
  }
  ValueType avg_feat_length;
  IndexType total_feats;
  ValueType k;
  ValueType b;
};

/**
 * @brief Calculates the tfidf values for a target matrix. Term frequency is calculate using
 * logrithmically scaled frequency.
 * @param total_feats_param: The total number of features in the matrix
 */
template <typename IndexType, typename ValueType>
struct tfidf {
  tfidf(IndexType total_feats_param) { total_feats = total_feats_param; }

  float __device__ operator()(const ValueType& value,
                              const ValueType& num_feats_id_occ,
                              const ValueType& feat_length)
  {
    ValueType tf  = ValueType(value / feat_length);
    ValueType idf = raft::log<ValueType>(total_feats / num_feats_id_occ);
    return tf * idf;
  }
  IndexType total_feats;
};

template <typename ValueType>
struct mapper {
  mapper(raft::device_vector_view<ValueType> map) : map(map) {}

  float __device__ operator()(const ValueType& value)
  {
    ValueType new_value = map[value];
    if (new_value) {
      return new_value;
    } else {
      return 0.0f;
    }
  }

  raft::device_vector_view<const ValueType> map;
};

template <typename IndexType, typename ValueType>
struct map_to {
  map_to(raft::device_vector_view<ValueType> map) : map(map) {}

  float __device__ operator()(const IndexType& key, const ValueType& count)
  {
    map[key] = count;
    return 0.0f;
  }

  raft::device_vector_view<ValueType> map;
};

/**
 * @brief Get unique counts
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param sort_vector: Input COO array that contains the keys.
 * @param secondary_vector: Input with secondary keys of COO, (columns or rows).
 * @param data: Input COO values array.
 * @param itr_vals: Input array used to calculate counts.
 * @param keys_out: Output array with one entry for each key. (same size as counts_out)
 * @param counts_out: Output array with cumulative sum for each key. (same size as keys_out)
 */
template <typename IndexType, typename ValueType, typename IdxT>
void get_uniques_counts(raft::resources& handle,
                        raft::device_vector_view<IndexType, IdxT> sort_vector,
                        raft::device_vector_view<IndexType, IdxT> secondary_vector,
                        raft::device_vector_view<ValueType, IdxT> data,
                        raft::device_vector_view<ValueType, IdxT> itr_vals,
                        raft::device_vector_view<IndexType, IdxT> keys_out,
                        raft::device_vector_view<ValueType, IdxT> counts_out)
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
  RAFT_CHECK_CUDA(stream);
  thrust::reduce_by_key(raft::resource::get_thrust_policy(handle),
                        sort_vector.data_handle(),
                        sort_vector.data_handle() + sort_vector.size(),
                        itr_vals.data_handle(),
                        keys_out.data_handle(),
                        counts_out.data_handle());
}

/**
 * @brief Broadcasts values to target indices of vector based on key/value look up
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param origin: Input array that has values to use for computation
 * @param keys: Output array that has keys, should be the size of unique
 * @param counts: Output array that contains the computed counts
 * @param results: Output array that scatters the counts to origin value positions. Same size as
 * origin array.
 */
template <typename IndexType, typename ValueType, typename IdxT>
void create_mapped_vector(raft::resources& handle,
                          const raft::device_vector_view<IndexType, IdxT> origin,
                          const raft::device_vector_view<IndexType, IdxT> keys,
                          const raft::device_vector_view<ValueType, IdxT> counts,
                          raft::device_vector_view<ValueType, IdxT> result,
                          IndexType key_size)
{
  // index into the last element and then add 1 to it.
  auto origin_map = raft::make_device_vector<ValueType, IdxT>(handle, key_size + 1);
  raft::matrix::fill(handle, origin_map.view(), 0.0f);

  auto dummy_vec = raft::make_device_vector<ValueType, IdxT>(handle, keys.size());
  raft::linalg::map(handle,
                    dummy_vec.view(),
                    map_to<IndexType, ValueType>(origin_map.view()),
                    raft::make_const_mdspan(keys),
                    raft::make_const_mdspan(counts));

  raft::linalg::map(handle, result, raft::cast_op<ValueType>{}, raft::make_const_mdspan(origin));
  raft::linalg::map(
    handle, result, mapper<ValueType>(origin_map.view()), raft::make_const_mdspan(result));
}

/**
 * @brief Compute row(id) counts
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param id_counts: Output array that stores counts per row, scattered to same shape as rows.
 * @param n_rows: Number of rows in matrix
 */
template <typename IndexType, typename ValueType, typename IdxT>
void get_id_counts(raft::resources& handle,
                   raft::device_vector_view<IndexType, IdxT> rows,
                   raft::device_vector_view<IndexType, IdxT> columns,
                   raft::device_vector_view<ValueType, IdxT> values,
                   raft::device_vector_view<IndexType, IdxT> id_counts,
                   IndexType n_rows)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  raft::sparse::op::coo_sort(int(rows.size()),
                             int(columns.size()),
                             int(values.size()),
                             rows.data_handle(),
                             columns.data_handle(),
                             values.data_handle(),
                             stream);

  auto rows_counts = raft::make_device_vector<IndexType, IdxT>(handle, n_rows);
  raft::matrix::fill(handle, rows_counts.view(), 0);

  raft::sparse::linalg::coo_degree<IndexType>(raft::make_const_mdspan(rows).data_handle(),
                                              int(rows.size()),
                                              rows_counts.data_handle(),
                                              stream);

  raft::linalg::map(
    handle, id_counts, mapper<IndexType>(rows_counts.view()), raft::make_const_mdspan(rows));
}

/**
 * @brief Gather per feature mean values, returns the cumulative avg feature length.
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param feat_lengths: Output array that stores mean per feature value
 * @param n_cols: Number of columns in matrix
 */
template <typename IndexType, typename ValueType, typename IdxT>
float get_feature_data(raft::resources& handle,
                       raft::device_vector_view<IndexType, IdxT> rows,
                       raft::device_vector_view<IndexType, IdxT> columns,
                       raft::device_vector_view<ValueType, IdxT> values,
                       raft::device_vector_view<ValueType, IdxT> feat_lengths,
                       IndexType n_cols)
{
  cudaStream_t stream    = raft::resource::get_cuda_stream(handle);
  auto preserved_columns = raft::make_device_vector<IndexType, IdxT>(handle, columns.size());

  int uniq_cols =
    raft::sparse::neighbors::get_n_components(columns.data_handle(), columns.size(), stream);
  raft::copy(preserved_columns.data_handle(), columns.data_handle(), columns.size(), stream);

  auto col_keys   = raft::make_device_vector<IndexType, IdxT>(handle, uniq_cols);
  auto col_counts = raft::make_device_vector<ValueType, IdxT>(handle, uniq_cols);

  get_uniques_counts(handle, columns, rows, values, values, col_keys.view(), col_counts.view());

  auto total_feature_lengths = raft::make_device_scalar<IndexType>(handle, 0);
  raft::linalg::mapReduce(total_feature_lengths.data_handle(),
                          col_counts.size(),
                          0,
                          raft::identity_op(),
                          raft::add_op(),
                          stream,
                          col_counts.data_handle());
  auto total_feature_lengths_host = raft::make_host_scalar<IndexType>(handle, 0);
  raft::copy(total_feature_lengths_host.data_handle(),
             total_feature_lengths.data_handle(),
             total_feature_lengths.size(),
             stream);
  ValueType avg_feat_length = ValueType(total_feature_lengths_host(0)) / n_cols;
  create_mapped_vector<IndexType, ValueType>(
    handle, preserved_columns.view(), col_keys.view(), col_counts.view(), feat_lengths, n_cols);
  return avg_feat_length;
}

/**
 * @brief Gather per feature mean values and id counts, returns the cumulative avg feature length.
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param feat_lengths: Output array that stores mean per feature value
 * @param id_counts: Output array that stores id(row) counts for nz values
 * @param n_rows: Number of rows in matrix
 * @param n_cols: Number of columns in matrix
 */
template <typename IndexType, typename ValueType, typename IdxT>
float sparse_search_preprocess(raft::resources& handle,
                               raft::device_vector_view<IndexType, IdxT> rows,
                               raft::device_vector_view<IndexType, IdxT> columns,
                               raft::device_vector_view<ValueType, IdxT> values,
                               raft::device_vector_view<ValueType, IdxT> feat_lengths,
                               raft::device_vector_view<IndexType, IdxT> id_counts,
                               IndexType n_rows,
                               IndexType n_cols)
{
  auto avg_feature_len = get_feature_data(handle, rows, columns, values, feat_lengths, n_cols);

  get_id_counts(handle, rows, columns, values, id_counts, n_rows);

  return avg_feature_len;
}

/**
 * @brief Use TFIDF algorithm to encode features in COO sparse matrix
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param rows: Input COO rows array
 * @param columns: Input COO columns array
 * @param values: Input COO values array
 * @param values_out: Output COO values array
 * @param n_rows: Number of rows in matrix
 * @param n_cols: Number of columns in matrix
 */
template <typename IndexType, typename ValueType, typename IdxT>
void base_encode_tfidf(raft::resources& handle,
                       raft::device_vector_view<IndexType, IdxT> rows,
                       raft::device_vector_view<IndexType, IdxT> columns,
                       raft::device_vector_view<ValueType, IdxT> values,
                       raft::device_vector_view<ValueType, IdxT> values_out,
                       IndexType n_rows,
                       IndexType n_cols)
{
  auto feat_lengths    = raft::make_device_vector<ValueType, IdxT>(handle, values.size());
  auto id_counts       = raft::make_device_vector<IndexType, IdxT>(handle, values.size());
  auto col_counts      = raft::make_device_vector<ValueType, IdxT>(handle, n_cols);
  auto avg_feat_length = sparse_search_preprocess<IndexType, ValueType>(
    handle, rows, columns, values, feat_lengths.view(), id_counts.view(), n_rows, n_cols);

  raft::linalg::map(handle,
                    values_out,
                    tfidf<IndexType, ValueType>(n_cols),
                    raft::make_const_mdspan(values),
                    raft::make_const_mdspan(id_counts.view()),
                    raft::make_const_mdspan(feat_lengths.view()));
}

/**
 * @brief Use TFIDF algorithm to encode features in COO sparse matrix
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param coo_in: Input COO matrix
 * @param values_out: Output COO values array
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_tfidf(raft::resources& handle,
                  raft::device_coo_matrix_view<ValueType, IndexType, IndexType, IndexType> coo_in,
                  raft::device_vector_view<ValueType, IdxT> values_out)
{
  auto rows = raft::make_device_vector_view<IndexType, IdxT>(
    coo_in.structure_view().get_rows().data(), coo_in.structure_view().get_rows().size());
  auto columns = raft::make_device_vector_view<IndexType, IdxT>(
    coo_in.structure_view().get_cols().data(), coo_in.structure_view().get_cols().size());
  auto values = raft::make_device_vector_view<ValueType, IdxT>(coo_in.get_elements().data(),
                                                               coo_in.get_elements().size());

  base_encode_tfidf<IndexType, ValueType, IdxT>(handle,
                                                rows,
                                                columns,
                                                values,
                                                values_out,
                                                coo_in.structure_view().get_n_rows(),
                                                coo_in.structure_view().get_n_cols());
}

/**
 * @brief Use TFIDF algorithm to encode features in CSR sparse matrix
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param csr_in: Input CSR matrix
 * @param values_out: Output values array
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_tfidf(raft::resources& handle,
                  raft::device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType> csr_in,
                  raft::device_vector_view<ValueType, IdxT> values_out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto indptr = raft::make_device_vector_view<IndexType, IdxT>(
    csr_in.structure_view().get_indptr().data(), csr_in.structure_view().get_indptr().size());
  auto indices = raft::make_device_vector_view<IndexType, IdxT>(
    csr_in.structure_view().get_indices().data(), csr_in.structure_view().get_indices().size());
  auto values = raft::make_device_vector_view<ValueType, IdxT>(csr_in.get_elements().data(),
                                                               csr_in.get_elements().size());

  auto rows = raft::make_device_vector<IndexType, IdxT>(handle, values.size());

  raft::sparse::convert::csr_to_coo<IndexType>(indptr.data_handle(),
                                               csr_in.structure_view().get_n_rows(),
                                               rows.data_handle(),
                                               rows.size(),
                                               stream);

  base_encode_tfidf<IndexType, ValueType, IdxT>(handle,
                                                rows.view(),
                                                indices,
                                                values,
                                                values_out,
                                                csr_in.structure_view().get_n_rows(),
                                                csr_in.structure_view().get_n_cols());
}

/**
 * @brief Use BM25 algorithm to encode features in COO sparse matrix
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
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
template <typename IndexType, typename ValueType, typename IdxT>
void base_encode_bm25(raft::resources& handle,
                      raft::device_vector_view<IndexType, IdxT> rows,
                      raft::device_vector_view<IndexType, IdxT> columns,
                      raft::device_vector_view<ValueType, IdxT> values,
                      raft::device_vector_view<ValueType, IdxT> values_out,
                      IndexType n_rows,
                      IndexType n_cols,
                      ValueType k_param = 1.6f,
                      ValueType b_param = 0.75f)
{
  auto feat_lengths = raft::make_device_vector<ValueType, IdxT>(handle, values.size());
  auto id_counts    = raft::make_device_vector<IndexType, IdxT>(handle, values.size());
  auto col_counts   = raft::make_device_vector<ValueType, IdxT>(handle, n_cols);

  auto avg_feat_length = sparse_search_preprocess<IndexType, ValueType>(
    handle, rows, columns, values, feat_lengths.view(), id_counts.view(), n_rows, n_cols);

  raft::linalg::map(handle,
                    values_out,
                    bm25<IndexType, ValueType>(n_cols, avg_feat_length, k_param, b_param),
                    raft::make_const_mdspan(values),
                    raft::make_const_mdspan(id_counts.view()),
                    raft::make_const_mdspan(feat_lengths.view()));
}

/**
 * @brief Use BM25 algorithm to encode features in COO sparse matrix
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param coo_in: Input COO matrix
 * @param values_out: Output values array
 * @param k_param: K value to use for BM25 algorithm
 * @param b_param: B value to use for BM25 algorithm
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_coo_matrix_view<ValueType, IndexType, IndexType, IndexType> coo_in,
                 raft::device_vector_view<ValueType, IdxT> values_out,
                 ValueType k_param = 1.6f,
                 ValueType b_param = 0.75f)
{
  auto rows = raft::make_device_vector_view<IndexType, IdxT>(
    coo_in.structure_view().get_rows().data(), coo_in.structure_view().get_rows().size());
  auto columns = raft::make_device_vector_view<IndexType, IdxT>(
    coo_in.structure_view().get_cols().data(), coo_in.structure_view().get_cols().size());
  auto values = raft::make_device_vector_view<ValueType, IdxT>(coo_in.get_elements().data(),
                                                               coo_in.get_elements().size());

  base_encode_bm25<IndexType, ValueType, IdxT>(handle,
                                               rows,
                                               columns,
                                               values,
                                               values_out,
                                               coo_in.structure_view().get_n_rows(),
                                               coo_in.structure_view().get_n_cols());
}

/**
 * @brief Use BM25 algorithm to encode features in CSR sparse matrix
 * @tparam IndexType: the type of the edge indexes in the matrix
 * @tparam ValueType: the type of the values for edges
 * @tparam IdxT: the type of the index values
 * @param handle: raft resource handle
 * @param csr_in: Input CSR matrix
 * @param values_out: Output values array
 * @param k_param: K value to use for BM25 algorithm
 * @param b_param: B value to use for BM25 algorithm
 */
template <typename IndexType, typename ValueType, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_csr_matrix_view<ValueType, IndexType, IndexType, IndexType> csr_in,
                 raft::device_vector_view<ValueType, IdxT> values_out,
                 ValueType k_param = 1.6f,
                 ValueType b_param = 0.75f)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto indptr = raft::make_device_vector_view<IndexType, IdxT>(
    csr_in.structure_view().get_indptr().data(), csr_in.structure_view().get_indptr().size());
  auto indices = raft::make_device_vector_view<IndexType, IdxT>(
    csr_in.structure_view().get_indices().data(), csr_in.structure_view().get_indices().size());
  auto values = raft::make_device_vector_view<ValueType, IdxT>(csr_in.get_elements().data(),
                                                               csr_in.get_elements().size());

  auto rows = raft::make_device_vector<IndexType, IdxT>(handle, values.size());

  raft::sparse::convert::csr_to_coo<IndexType>(indptr.data_handle(),
                                               csr_in.structure_view().get_n_rows(),
                                               rows.data_handle(),
                                               rows.size(),
                                               stream);

  base_encode_bm25<IndexType, ValueType, IdxT>(handle,
                                               rows.view(),
                                               indices,
                                               values,
                                               values_out,
                                               csr_in.structure_view().get_n_rows(),
                                               csr_in.structure_view().get_n_cols());
}

}  // namespace raft::sparse::matrix::detail