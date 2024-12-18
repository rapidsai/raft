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
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>
#include <raft/sparse/convert/coo.cuh>
#include <raft/sparse/matrix/preprocessing.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/op/sort.cuh>

namespace raft::util {

template <typename T1, typename T2>
struct check_zeroes {
  float __device__ operator()(const T1& value, const T2& idx)
  {
    if (value == 0) {
      return 0.f;
    } else {
      return 1.f;
    }
  }
};

template <typename T1, typename T2>
void preproc_coo(raft::resources& handle,
                 raft::host_vector_view<T1> h_rows,
                 raft::host_vector_view<T1> h_cols,
                 raft::host_vector_view<T2> h_elems,
                 raft::device_vector_view<T2> results,
                 int num_rows,
                 int num_cols,
                 bool tf_idf)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int rows_size       = h_rows.size();
  int cols_size       = h_cols.size();
  int elements_size   = h_elems.size();
  auto device_matrix  = raft::make_device_matrix<T2, int64_t>(handle, num_rows, num_cols);
  raft::matrix::fill<T2>(handle, device_matrix.view(), 0.0f);
  auto host_matrix = raft::make_host_matrix<T2, int64_t>(handle, num_rows, num_cols);
  raft::copy(host_matrix.data_handle(), device_matrix.data_handle(), device_matrix.size(), stream);

  raft::resource::sync_stream(handle, stream);

  for (int i = 0; i < elements_size; i++) {
    int row               = h_rows(i);
    int col               = h_cols(i);
    float element         = h_elems(i);
    host_matrix(row, col) = element;
  }

  raft::copy(device_matrix.data_handle(), host_matrix.data_handle(), host_matrix.size(), stream);
  auto output_cols_lengths = raft::make_device_matrix<T2, int64_t>(handle, 1, num_cols);
  raft::linalg::reduce(output_cols_lengths.data_handle(),
                       device_matrix.data_handle(),
                       num_rows,
                       num_cols,
                       0.0f,
                       false,
                       true,
                       stream);
  auto h_output_cols_lengths = raft::make_host_matrix<T2, int64_t>(handle, 1, num_cols);
  raft::copy(h_output_cols_lengths.data_handle(),
             output_cols_lengths.data_handle(),
             output_cols_lengths.size(),
             stream);

  auto output_cols_length_sum = raft::make_device_scalar<T1>(handle, 0);
  raft::linalg::mapReduce(output_cols_length_sum.data_handle(),
                          num_cols,
                          0,
                          raft::identity_op(),
                          raft::add_op(),
                          stream,
                          output_cols_lengths.data_handle());
  auto h_output_cols_length_sum = raft::make_host_scalar<T1>(handle, 0);
  raft::copy(h_output_cols_length_sum.data_handle(),
             output_cols_length_sum.data_handle(),
             output_cols_length_sum.size(),
             stream);
  T2 avg_col_length = T2(h_output_cols_length_sum(0)) / num_cols;

  auto output_rows_freq = raft::make_device_matrix<T2, int64_t>(handle, 1, num_rows);
  raft::linalg::reduce(output_rows_freq.data_handle(),
                       device_matrix.data_handle(),
                       num_rows,
                       num_cols,
                       0.0f,
                       false,
                       false,
                       stream);

  auto output_rows_cnt = raft::make_device_matrix<T2, int64_t>(handle, 1, num_rows);
  raft::linalg::reduce(output_rows_cnt.data_handle(),
                       device_matrix.data_handle(),
                       num_rows,
                       num_cols,
                       0.0f,
                       false,
                       false,
                       stream,
                       false,
                       check_zeroes<T2, T2>());
  auto h_output_rows_cnt = raft::make_host_matrix<T2, int64_t>(handle, 1, num_rows);
  raft::copy(
    h_output_rows_cnt.data_handle(), output_rows_cnt.data_handle(), output_rows_cnt.size(), stream);

  auto out_device_matrix = raft::make_device_matrix<T2, int64_t>(handle, num_rows, num_cols);
  raft::matrix::fill<T2>(handle, out_device_matrix.view(), 0.0f);
  auto out_host_matrix = raft::make_host_matrix<T2, int64_t>(handle, num_rows, num_cols);
  auto out_host_vector = raft::make_host_vector<T2, int64_t>(handle, results.size());

  float k1  = 1.6f;
  float b   = 0.75f;
  int count = 0;
  float result;
  for (int row = 0; row < num_rows; row++) {
    for (int col = 0; col < num_cols; col++) {
      float val = host_matrix(row, col);
      if (val == 0) {
        out_host_matrix(row, col) = 0.0f;
      } else {
        float tf  = float(val / h_output_cols_lengths(0, col));
        float idf = raft::log<T2>(num_cols / h_output_rows_cnt(0, row));
        if (tf_idf) {
          result = tf * idf;
        } else {
          float bm25 = ((k1 + 1) * tf) /
                       (k1 * ((1 - b) + b * (h_output_cols_lengths(0, col) / avg_col_length)) + tf);
          result = idf * bm25;
        }
        out_host_matrix(row, col) = result;
        out_host_vector(count)    = result;
        count++;
      }
    }
  }
  raft::copy(results.data_handle(), out_host_vector.data_handle(), out_host_vector.size(), stream);
}

template <typename T1, typename T2>
int get_dupe_mask_count(raft::resources& handle,
                        raft::device_vector_view<T1> rows,
                        raft::device_vector_view<T1> columns,
                        raft::device_vector_view<T2> values,
                        const raft::device_vector_view<T1>& mask)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  raft::sparse::op::coo_sort(int(rows.size()),
                             int(columns.size()),
                             int(values.size()),
                             rows.data_handle(),
                             columns.data_handle(),
                             values.data_handle(),
                             stream);

  raft::sparse::op::compute_duplicates_mask<T1>(
    mask.data_handle(), rows.data_handle(), columns.data_handle(), rows.size(), stream);

  int col_nnz_count = thrust::reduce(raft::resource::get_thrust_policy(handle),
                                     mask.data_handle(),
                                     mask.data_handle() + mask.size());
  return col_nnz_count;
}

template <typename T1, typename T2>
void remove_dupes(raft::resources& handle,
                  raft::device_vector_view<T1> rows,
                  raft::device_vector_view<T1> columns,
                  raft::device_vector_view<T2> values,
                  raft::device_vector_view<T1> mask,
                  const raft::device_vector_view<T1>& out_rows,
                  const raft::device_vector_view<T1>& out_cols,
                  const raft::device_vector_view<T2>& out_vals,
                  int num_rows = 128)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  auto col_counts = raft::make_device_vector<T1, int64_t>(handle, columns.size());

  thrust::fill(raft::resource::get_thrust_policy(handle),
               col_counts.data_handle(),
               col_counts.data_handle() + col_counts.size(),
               1.0f);

  auto keys_out   = raft::make_device_vector<T1, int64_t>(handle, num_rows);
  auto counts_out = raft::make_device_vector<T1, int64_t>(handle, num_rows);

  thrust::reduce_by_key(raft::resource::get_thrust_policy(handle),
                        rows.data_handle(),
                        rows.data_handle() + rows.size(),
                        col_counts.data_handle(),
                        keys_out.data_handle(),
                        counts_out.data_handle());

  auto mask_out = raft::make_device_vector<T2, int64_t>(handle, rows.size());

  raft::linalg::map(handle, mask_out.view(), raft::cast_op<T2>{}, raft::make_const_mdspan(mask));

  auto values_c = raft::make_device_vector<T2, int64_t>(handle, values.size());
  raft::linalg::map(handle,
                    values_c.view(),
                    raft::mul_op{},
                    raft::make_const_mdspan(values),
                    raft::make_const_mdspan(mask_out.view()));

  auto keys_nnz_out   = raft::make_device_vector<T1, int64_t>(handle, num_rows);
  auto counts_nnz_out = raft::make_device_vector<T1, int64_t>(handle, num_rows);

  thrust::reduce_by_key(raft::resource::get_thrust_policy(handle),
                        rows.data_handle(),
                        rows.data_handle() + rows.size(),
                        mask.data_handle(),
                        keys_nnz_out.data_handle(),
                        counts_nnz_out.data_handle());

  raft::sparse::op::coo_remove_scalar<T2>(rows.data_handle(),
                                          columns.data_handle(),
                                          values_c.data_handle(),
                                          values_c.size(),
                                          out_rows.data_handle(),
                                          out_cols.data_handle(),
                                          out_vals.data_handle(),
                                          counts_nnz_out.data_handle(),
                                          counts_out.data_handle(),
                                          0,
                                          num_rows,
                                          stream);
}

template <typename T1, typename T2>
void create_dataset(raft::resources& handle,
                    raft::device_vector_view<T1, int64_t> rows,
                    raft::device_vector_view<T1, int64_t> columns,
                    raft::device_vector_view<T2, int64_t> values,
                    int max_term_occurence_doc = 5,
                    int num_rows_unique        = 7,
                    int num_cols_unique        = 7,
                    int seed                   = 12345)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::random::RngState rng(seed);

  auto d_out = raft::make_device_vector<T1, int64_t>(handle, rows.size() * 2);

  int theta_guide = max(num_rows_unique, num_cols_unique);
  auto theta      = raft::make_device_vector<T2, int64_t>(handle, theta_guide * 4);

  raft::random::uniform(handle, rng, theta.view(), 0.0f, 1.0f);

  raft::random::rmat_rectangular_gen(d_out.data_handle(),
                                     rows.data_handle(),
                                     columns.data_handle(),
                                     theta.data_handle(),
                                     num_rows_unique,
                                     num_cols_unique,
                                     int(values.size()),
                                     stream,
                                     rng);

  auto vals = raft::make_device_vector<T1, int64_t>(handle, rows.size());
  raft::random::uniformInt(handle, rng, vals.view(), 1, max_term_occurence_doc);
  raft::linalg::map(handle, values, raft::cast_op<T2>{}, raft::make_const_mdspan(vals.view()));
}

};  // namespace raft::util