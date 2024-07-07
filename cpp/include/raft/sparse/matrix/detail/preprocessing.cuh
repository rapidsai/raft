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
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_span.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/math.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/sparse_types.hpp>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/sparse/op/sort.cuh>

#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

namespace raft::sparse::matrix::detail {

struct bm25 {
  bm25(int num_docs, float avg_doc_len, float k_param, float b_param)
  {
    total_docs     = num_docs;
    avg_doc_length = avg_doc_len;
    k              = k_param;
    b              = b_param;
  }

  template <typename T1>
  float __device__ operator()(const T1& values, const T1& doc_length, const T1& num_docs_term_occ)
  {
    return raft::log<float>(total_docs / (1 + num_docs_term_occ)) *
           ((values * (k + 1)) / (values + k * (1 - b + b * (doc_length / avg_doc_length))));
  }
  float avg_doc_length;
  int total_docs;
  float k;
  float b;
};

struct tfidf {
  tfidf(int total_docs_param) { total_docs = total_docs_param; }

  template <typename T1, typename T2>
  float __device__ operator()(const T1& values, const T2& num_docs_term_occ)
  {
    return raft::log<float>(1 + values) * raft::log<float>(total_docs / (1 + num_docs_term_occ));
  }
  int total_docs;
};

template <typename T>
struct mapper {
  mapper(raft::device_vector_view<const T> map) : map(map) {}

  __host__ __device__ void operator()(T& value) const
  {
    const T& new_value = map[value];
    if (new_value) {
      value = new_value;
    } else {
      value = 0;
    }
  }

  raft::device_vector_view<const T> map;
};

template <typename T1, typename T2>
void get_uniques_counts(raft::resources& handle,
                        raft::device_vector_view<T1, int64_t> sort_vector,
                        raft::device_vector_view<T1, int64_t> secondary_vector,
                        raft::device_vector_view<T2, int64_t> data,
                        raft::device_vector_view<T2, int64_t> itr_vals,
                        raft::device_vector_view<T1, int64_t> keys_out,
                        raft::device_vector_view<T2, int64_t> counts_out,
                        int n_rows,
                        int n_cols)
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

template <typename T1, typename T2>
void create_mapped_vector(raft::resources& handle,
                          const raft::device_vector_view<T1, int64_t> origin,
                          const raft::device_vector_view<T1, int64_t> keys,
                          const raft::device_vector_view<T2, int64_t> counts,
                          raft::device_vector_view<T2, int64_t> result)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto host_keys      = raft::make_host_vector<T1, int64_t>(handle, keys.size());

  raft::copy(host_keys.data_handle(), keys.data_handle(), keys.size(), stream);
  raft::linalg::map(handle, result, raft::cast_op<T2>{}, raft::make_const_mdspan(origin));
  int new_key_size = host_keys(host_keys.size() - 1) + 1;
  auto origin_map  = raft::make_device_vector<T2, int64_t>(handle, new_key_size);

  thrust::scatter(raft::resource::get_thrust_policy(handle),
                  counts.data_handle(),
                  counts.data_handle() + counts.size(),
                  keys.data_handle(),
                  origin_map.data_handle());
  thrust::for_each(raft::resource::get_thrust_policy(handle),
                   result.data_handle(),
                   result.data_handle() + result.size(),
                   mapper<T2>(raft::make_const_mdspan(origin_map.view())));
}

template <typename T1, typename T2>
void get_id_counts(raft::resources& handle,
                   raft::device_vector_view<T1, int64_t> rows,
                   raft::device_vector_view<T1, int64_t> columns,
                   raft::device_vector_view<T2, int64_t> values,
                   raft::device_vector_view<T2, int64_t> id_counts,
                   T1 n_rows,
                   T1 n_cols)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto num_rows =
    raft::sparse::neighbors::get_n_components(rows.data_handle(), rows.size(), stream);

  auto row_keys   = raft::make_device_vector<T1, int64_t>(handle, num_rows);
  auto row_counts = raft::make_device_vector<T2, int64_t>(handle, num_rows);
  auto row_fill   = raft::make_device_vector<T2, int64_t>(handle, rows.size());

  // the amount of columns(documents) that each row(term) is found in
  thrust::fill(raft::resource::get_thrust_policy(handle),
               row_fill.data_handle(),
               row_fill.data_handle() + row_fill.size(),
               1.0f);
  get_uniques_counts(handle,
                     rows,
                     columns,
                     values,
                     row_fill.view(),
                     row_keys.view(),
                     row_counts.view(),
                     n_rows,
                     n_cols);

  create_mapped_vector<T1, T2>(handle, rows, row_keys.view(), row_counts.view(), id_counts);
}

template <typename T1, typename T2>
std::tuple<T1, T1> get_feature_data(raft::resources& handle,
                                    raft::device_vector_view<T1, int64_t> rows,
                                    raft::device_vector_view<T1, int64_t> columns,
                                    raft::device_vector_view<T2, int64_t> values,
                                    raft::device_vector_view<T2, int64_t> feat_lengths,
                                    T1 n_rows,
                                    T1 n_cols)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto num_cols =
    raft::sparse::neighbors::get_n_components(columns.data_handle(), columns.size(), stream);
  auto col_keys   = raft::make_device_vector<T1, int64_t>(handle, num_cols);
  auto col_counts = raft::make_device_vector<T2, int64_t>(handle, num_cols);

  get_uniques_counts(
    handle, columns, rows, values, values, col_keys.view(), col_counts.view(), n_rows, n_cols);
  int total_feature_lengths = thrust::reduce(raft::resource::get_thrust_policy(handle),
                                             col_counts.data_handle(),
                                             col_counts.data_handle() + col_counts.size());
  float avg_feat_length     = float(total_feature_lengths) / col_keys.size();

  create_mapped_vector<T1, T2>(handle, columns, col_keys.view(), col_counts.view(), feat_lengths);
  return {col_keys.size(), avg_feat_length};
}

template <typename T1, typename T2>
std::tuple<int, int> sparse_search_preprocess(raft::resources& handle,
                                              raft::device_vector_view<T1, int64_t> rows,
                                              raft::device_vector_view<T1, int64_t> columns,
                                              raft::device_vector_view<T2, int64_t> values,
                                              raft::device_vector_view<T2, int64_t> feat_lengths,
                                              raft::device_vector_view<T2, int64_t> id_counts,
                                              T1 n_rows,
                                              T1 n_cols)
{
  auto [n_feature_keys, avg_feature_len] =
    get_feature_data(handle, rows, columns, values, feat_lengths, n_rows, n_cols);

  get_id_counts(handle, rows, columns, values, id_counts, n_rows, n_cols);

  return {n_feature_keys, avg_feature_len};
}

template <typename T1, typename T2, typename IdxT>
void base_encode_tfidf(raft::resources& handle,
                       raft::device_vector_view<T1, IdxT> rows,
                       raft::device_vector_view<T1, IdxT> columns,
                       raft::device_vector_view<T2, IdxT> values,
                       raft::device_vector_view<T2, IdxT> values_out,
                       int n_rows,
                       int n_cols)
{
  auto feat_lengths = raft::make_device_vector<float, IdxT>(handle, columns.size());
  auto id_counts    = raft::make_device_vector<float, IdxT>(handle, rows.size());
  auto [feat_count, avg_feat_length] = sparse_search_preprocess<int, float>(
    handle, rows, columns, values, feat_lengths.view(), id_counts.view(), n_rows, n_cols);

  raft::linalg::map(handle,
                    values_out,
                    tfidf(feat_count),
                    raft::make_const_mdspan(values),
                    raft::make_const_mdspan(id_counts.view()));
}

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

  auto rows = raft::make_device_vector<int, IdxT>(handle, values.size());
  raft::sparse::convert::csr_to_coo<int>(indptr.data_handle(),
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

template <typename T1, typename T2, typename IdxT>
void base_encode_bm25(raft::resources& handle,
                      raft::device_vector_view<T1, IdxT> rows,
                      raft::device_vector_view<T1, IdxT> columns,
                      raft::device_vector_view<T2, IdxT> values,
                      raft::device_vector_view<T2, IdxT> values_out,
                      int n_rows,
                      int n_cols,
                      float k_param = 1.6f,
                      float b_param = 0.75f)
{
  auto doc_lengths = raft::make_device_vector<T2, IdxT>(handle, columns.size());
  auto term_counts = raft::make_device_vector<T2, IdxT>(handle, rows.size());

  auto [doc_count, avg_doc_length] = sparse_search_preprocess<int, float>(
    handle, rows, columns, values, doc_lengths.view(), term_counts.view(), n_rows, n_cols);

  raft::linalg::map(handle,
                    values_out,
                    bm25(doc_count, avg_doc_length, k_param, b_param),
                    raft::make_const_mdspan(values),
                    raft::make_const_mdspan(doc_lengths.view()),
                    raft::make_const_mdspan(term_counts.view()));
}

template <typename T1, typename T2, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_coo_matrix_view<T2, T1, T1, T1> coo_in,
                 raft::device_vector_view<T2, IdxT> values_out,
                 float k_param = 1.6f,
                 float b_param = 0.75f)
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

template <typename T1, typename T2, typename IdxT>
void encode_bm25(raft::resources& handle,
                 raft::device_csr_matrix_view<T2, T1, T1, T1> csr_in,
                 raft::device_vector_view<T2, IdxT> values_out,
                 float k_param = 1.6f,
                 float b_param = 0.75f)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto indptr = raft::make_device_vector_view<T1, IdxT>(
    csr_in.structure_view().get_indptr().data(), csr_in.structure_view().get_indptr().size());
  auto indices = raft::make_device_vector_view<T1, IdxT>(
    csr_in.structure_view().get_indices().data(), csr_in.structure_view().get_indices().size());
  auto values = raft::make_device_vector_view<T2, IdxT>(csr_in.get_elements().data(),
                                                        csr_in.get_elements().size());

  auto rows = raft::make_device_vector<int, IdxT>(handle, values.size());

  raft::sparse::convert::csr_to_coo<int>(indptr.data_handle(),
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