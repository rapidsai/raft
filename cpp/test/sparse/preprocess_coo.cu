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
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

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

    rmm::device_uvector<Index_> rows_uvec(rows.size(), stream);
    rmm::device_uvector<Index_> cols_uvec(rows.size(), stream);
    rmm::device_uvector<Type_f> vals_uvec(rows.size(), stream);

    raft::util::create_dataset<Index_, Type_f>(
      handle, rows.view(), columns.view(), values.view(), 5, params.n_rows, params.n_cols);

    raft::sparse::op::coo_sort(int(rows.size()),
                               int(columns.size()),
                               int(values.size()),
                               rows.data_handle(),
                               columns.data_handle(),
                               values.data_handle(),
                               stream);

    raft::copy(rows_uvec.data(), rows.data_handle(), rows.size(), stream);
    raft::copy(cols_uvec.data(), columns.data_handle(), columns.size(), stream);
    raft::copy(vals_uvec.data(), values.data_handle(), values.size(), stream);

    raft::sparse::COO<Type_f, Index_> coo(stream);
    raft::sparse::op::max_duplicates(handle,
                                     coo,
                                     rows_uvec.data(),
                                     cols_uvec.data(),
                                     vals_uvec.data(),
                                     params.nnz_edges,
                                     num_rows,
                                     num_cols);

    auto rows_csr = raft::make_device_vector<Index_, int64_t>(handle, num_rows + 1);

    raft::sparse::convert::sorted_coo_to_csr(
      coo.rows(), coo.nnz, rows_csr.data_handle(), num_rows + 1, stream);

    auto csr_struct_view = raft::make_device_compressed_structure_view(
      rows_csr.data_handle(), coo.cols(), num_rows, num_cols, coo.nnz);

    auto csr_matrix =
      raft::make_device_csr_matrix<Type_f, Index_, Index_, Index_>(handle, csr_struct_view);
    raft::update_device<Type_f>(
      csr_matrix.view().get_elements().data(), coo.vals(), coo.nnz, stream);

    auto coo_struct_view = raft::make_device_coordinate_structure_view(
      coo.rows(), coo.cols(), num_rows, num_cols, int(coo.nnz));
    auto c_matrix =
      raft::make_device_coo_matrix<Type_f, Index_, Index_, Index_>(handle, coo_struct_view);
    raft::update_device<Type_f>(c_matrix.view().get_elements().data(), coo.vals(), coo.nnz, stream);

    auto result = raft::make_device_vector<Type_f, int64_t>(handle, coo.nnz);

    if (bm25_on) {
      auto bm25_vals = raft::make_device_vector<Type_f, int64_t>(handle, coo.nnz);
      sparse::matrix::encode_bm25<Index_, Type_f>(handle, c_matrix.view(), result.view());
      raft::util::calc_tfidf_bm25<Index_, Type_f>(handle, csr_matrix.view(), bm25_vals.view());
      ASSERT_TRUE(raft::devArrMatch<Type_f>(bm25_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    } else {
      auto tfidf_vals = raft::make_device_vector<Type_f, int64_t>(handle, coo.nnz);
      sparse::matrix::encode_tfidf<Index_, Type_f>(handle, c_matrix.view(), result.view());
      raft::util::calc_tfidf_bm25<Index_, Type_f>(
        handle, csr_matrix.view(), tfidf_vals.view(), true);
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