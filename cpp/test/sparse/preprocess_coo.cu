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

#include "../preprocess_utils.cu"
#include "../test_utils.cuh"

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/preprocessing.cuh>
#include <raft/sparse/selection/knn.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename T1, typename T2>
void calc_tfidf_bm25(raft::resources& handle,
                     raft::device_coo_matrix_view<T2, T1, T1, T1> coo_in,
                     raft::device_vector_view<T2> results,
                     bool tf_idf = false)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  int num_rows        = coo_in.structure_view().get_n_rows();
  int num_cols        = coo_in.structure_view().get_n_cols();
  int rows_size       = coo_in.structure_view().get_cols().size();
  int cols_size       = coo_in.structure_view().get_rows().size();
  int elements_size   = coo_in.get_elements().size();

  auto h_rows  = raft::make_host_vector<T1, int64_t>(handle, rows_size);
  auto h_cols  = raft::make_host_vector<T1, int64_t>(handle, cols_size);
  auto h_elems = raft::make_host_vector<T2, int64_t>(handle, elements_size);

  raft::copy(h_rows.data_handle(),
             coo_in.structure_view().get_rows().data(),
             coo_in.structure_view().get_rows().size(),
             stream);
  raft::copy(h_cols.data_handle(),
             coo_in.structure_view().get_cols().data(),
             coo_in.structure_view().get_cols().size(),
             stream);
  raft::copy(
    h_elems.data_handle(), coo_in.get_elements().data(), coo_in.get_elements().size(), stream);
  raft::util::preproc_coo<T1, T2>(
    handle, h_rows.view(), h_cols.view(), h_elems.view(), results, num_rows, num_cols, tf_idf);
}

template <typename Type_f, typename Index_>
struct SparsePreprocessInputs {
  int n_rows;
  int n_cols;
  int nnz_edges;
};

template <typename Type_f, typename Index_>
class SparsePreprocessCoo
  : public ::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>> {
 public:
  SparsePreprocessCoo()
    : params(::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle))
  {
  }

 protected:
  void SetUp() override {}

  void Run(bool bm25_on)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);

    int num_rows = pow(2, params.n_rows);
    int num_cols = pow(2, params.n_cols);

    auto rows    = raft::make_device_vector<Index_, int64_t>(handle, params.nnz_edges);
    auto columns = raft::make_device_vector<Index_, int64_t>(handle, params.nnz_edges);
    auto values  = raft::make_device_vector<Type_f, int64_t>(handle, params.nnz_edges);
    auto mask    = raft::make_device_vector<Index_, int64_t>(handle, params.nnz_edges);

    raft::util::create_dataset<Index_, Type_f>(
      handle, rows.view(), columns.view(), values.view(), 5, params.n_rows, params.n_cols);
    int non_dupe_nnz_count = raft::util::get_dupe_mask_count<Index_, Type_f>(
      handle, rows.view(), columns.view(), values.view(), mask.view());

    auto rows_nnz    = raft::make_device_vector<Index_, int64_t>(handle, non_dupe_nnz_count);
    auto columns_nnz = raft::make_device_vector<Index_, int64_t>(handle, non_dupe_nnz_count);
    auto values_nnz  = raft::make_device_vector<Type_f, int64_t>(handle, non_dupe_nnz_count);
    raft::util::remove_dupes<Index_, Type_f>(handle,
                                             rows.view(),
                                             columns.view(),
                                             values.view(),
                                             mask.view(),
                                             rows_nnz.view(),
                                             columns_nnz.view(),
                                             values_nnz.view(),
                                             num_rows);

    auto coo_struct_view = raft::make_device_coordinate_structure_view(rows_nnz.data_handle(),
                                                                       columns_nnz.data_handle(),
                                                                       num_rows,
                                                                       num_cols,
                                                                       int(values_nnz.size()));
    auto c_matrix =
      raft::make_device_coo_matrix<Type_f, Index_, Index_, Index_>(handle, coo_struct_view);
    raft::update_device<Type_f>(
      c_matrix.view().get_elements().data(), values_nnz.data_handle(), values_nnz.size(), stream);

    auto result     = raft::make_device_vector<Type_f, int64_t>(handle, values_nnz.size());
    auto bm25_vals  = raft::make_device_vector<Type_f, int64_t>(handle, values_nnz.size());
    auto tfidf_vals = raft::make_device_vector<Type_f, int64_t>(handle, values_nnz.size());

    if (bm25_on) {
      sparse::matrix::encode_bm25<Index_, Type_f>(handle, c_matrix.view(), result.view());
      calc_tfidf_bm25<Index_, Type_f>(handle, c_matrix.view(), bm25_vals.view());
      ASSERT_TRUE(raft::devArrMatch<Type_f>(bm25_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    } else {
      sparse::matrix::encode_tfidf<Index_, Type_f>(handle, c_matrix.view(), result.view());
      calc_tfidf_bm25<Index_, Type_f>(handle, c_matrix.view(), tfidf_vals.view(), true);
      ASSERT_TRUE(raft::devArrMatch<Type_f>(tfidf_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    }

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SparsePreprocessInputs<Type_f, Index_> params;
};

using SparsePreprocessTfidfCoo = SparsePreprocessCoo<float, int>;
TEST_P(SparsePreprocessTfidfCoo, Result) { Run(false); }

using SparsePreprocessBm25Coo = SparsePreprocessCoo<float, int>;
TEST_P(SparsePreprocessBm25Coo, Result) { Run(true); }

using SparsePreprocessTfidfCooBig = SparsePreprocessCoo<float, int>;
TEST_P(SparsePreprocessTfidfCooBig, Result) { Run(false); }

using SparsePreprocessBm25CooBig = SparsePreprocessCoo<float, int>;
TEST_P(SparsePreprocessBm25CooBig, Result) { Run(true); }

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs = {
  {
    10,   // n_rows_factor
    10,   // n_cols_factor
    1000  // nnz_edges
  },
};

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs_big = {
  {
    15,      // n_rows_factor
    15,      // n_cols_factor
    1000000  // nnz_edges
  },
};

INSTANTIATE_TEST_CASE_P(SparsePreprocessCoo,
                        SparsePreprocessTfidfCoo,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCoo,
                        SparsePreprocessBm25Coo,
                        ::testing::ValuesIn(sparse_preprocess_inputs));

INSTANTIATE_TEST_CASE_P(SparsePreprocessCoo,
                        SparsePreprocessTfidfCooBig,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCoo,
                        SparsePreprocessBm25CooBig,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));

}  // namespace sparse
}  // namespace raft