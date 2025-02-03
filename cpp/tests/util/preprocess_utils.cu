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
#include <raft/sparse/convert/dense.cuh>
#include <raft/sparse/matrix/preprocessing.cuh>
#include <raft/sparse/neighbors/cross_component_nn.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/op/sort.cuh>

namespace raft::util {

template <typename T>
void print_vals(raft::resources& handle, const raft::device_vector_view<T, size_t>& out)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto h_out          = raft::make_host_vector<T>(out.size());
  raft::copy(h_out.data_handle(), out.data_handle(), out.size(), stream);
  int limit = int(out.size());
  for (int i = 0; i < limit; i++) {
    std::cout << float(h_out(i)) << ", ";
  }
  std::cout << std::endl;
}

template <typename T>
void print_vals(raft::resources& handle, T* out, int len)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto h_out          = raft::make_host_vector<T>(handle, len);
  raft::copy(h_out.data_handle(), out, len, stream);
  int limit = int(len);
  for (int i = 0; i < limit; i++) {
    std::cout << float(h_out(i)) << ", ";
  }
  std::cout << std::endl;
}

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
void preproc(raft::resources& handle,
             raft::device_vector_view<T2> dense_values,
             raft::device_vector_view<T2> results,
             int num_rows,
             int num_cols,
             bool tf_idf)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);

  // create matrix and copy to device
  auto host_dense_vals = raft::make_host_vector<T2, int64_t>(handle, dense_values.size());
  raft::copy(
    host_dense_vals.data_handle(), dense_values.data_handle(), dense_values.size(), stream);

  auto host_matrix =
    raft::make_host_matrix_view<T2, int64_t>(host_dense_vals.data_handle(), num_rows, num_cols);
  auto device_matrix = raft::make_device_matrix<T2, int64_t>(handle, num_rows, num_cols);

  raft::copy(device_matrix.data_handle(), host_matrix.data_handle(), host_matrix.size(), stream);

  // get sum reduce for each row (length of the document)
  auto output_rows_lengths = raft::make_device_matrix<T2, int64_t>(handle, 1, num_rows);
  raft::linalg::reduce(output_rows_lengths.data_handle(),
                       device_matrix.data_handle(),
                       num_cols,
                       num_rows,
                       0.0f,
                       true,
                       true,
                       stream);
  auto h_output_rows_lengths = raft::make_host_matrix<T2, int64_t>(handle, 1, num_rows);
  raft::copy(h_output_rows_lengths.data_handle(),
             output_rows_lengths.data_handle(),
             output_rows_lengths.size(),
             stream);

  // find the avg size of a document
  auto output_rows_length_sum = raft::make_device_scalar<T2>(handle, 0);
  raft::linalg::mapReduce(output_rows_length_sum.data_handle(),
                          num_rows,
                          0.0f,
                          raft::identity_op(),
                          raft::add_op(),
                          stream,
                          output_rows_lengths.data_handle());
  auto h_output_rows_length_sum = raft::make_host_scalar<T2>(handle, 0);
  raft::copy(h_output_rows_length_sum.data_handle(),
             output_rows_length_sum.data_handle(),
             output_rows_length_sum.size(),
             stream);
  T2 avg_row_length = (T2)h_output_rows_length_sum(0) / num_rows;

  // find the number of docs(row) each vocab(col) word is in
  auto output_cols_cnt = raft::make_device_matrix<T2, int64_t>(handle, 1, num_cols);
  raft::linalg::reduce(output_cols_cnt.data_handle(),
                       device_matrix.data_handle(),
                       num_cols,
                       num_rows,
                       0.0f,
                       true,
                       false,
                       stream,
                       false,
                       check_zeroes<T2, T2>());
  auto h_output_cols_cnt = raft::make_host_matrix<T2, int64_t>(handle, 1, num_cols);
  raft::copy(
    h_output_cols_cnt.data_handle(), output_cols_cnt.data_handle(), output_cols_cnt.size(), stream);

  // perform bm25/tfidf calculations
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
      // std::cout << val << ", ";
      if (val == 0) {
        out_host_matrix(row, col) = 0.0f;
      } else {
        float tf      = (float)val / h_output_rows_lengths(0, row);
        double idf_in = (double)num_rows / h_output_cols_cnt(0, col);
        float idf     = (float)raft::log<double>(idf_in);
        if (tf_idf) {
          result = tf * idf;
        } else {
          float bm25 = ((k1 + 1) * tf) /
                       (k1 * ((1 - b) + b * (h_output_rows_lengths(0, row) / avg_row_length)) + tf);
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
void calc_tfidf_bm25(raft::resources& handle,
                     raft::device_csr_matrix_view<T2, T1, T1, T1> csr_in,
                     raft::device_vector_view<T2> results,
                     bool tf_idf = false)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int num_rows        = csr_in.structure_view().get_n_rows();
  int num_cols        = csr_in.structure_view().get_n_cols();
  int rows_size       = csr_in.structure_view().get_indptr().size();
  int cols_size       = csr_in.structure_view().get_indices().size();
  int elements_size   = csr_in.get_elements().size();

  auto indptr = raft::make_device_vector_view<T1, int64_t>(
    csr_in.structure_view().get_indptr().data(), rows_size);
  auto indices = raft::make_device_vector_view<T1, int64_t>(
    csr_in.structure_view().get_indices().data(), cols_size);
  auto values =
    raft::make_device_vector_view<T2, int64_t>(csr_in.get_elements().data(), elements_size);
  auto dense_values = raft::make_device_vector<T2, int64_t>(handle, num_rows * num_cols);

  cusparseHandle_t cu_handle;
  RAFT_CUSPARSE_TRY(cusparseCreate(&cu_handle));

  raft::sparse::convert::csr_to_dense(cu_handle,
                                      num_rows,
                                      num_cols,
                                      elements_size,
                                      indptr.data_handle(),
                                      indices.data_handle(),
                                      values.data_handle(),
                                      num_rows,
                                      dense_values.data_handle(),
                                      stream,
                                      true);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  preproc<T1, T2>(handle, dense_values.view(), results, num_rows, num_cols, tf_idf);
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
