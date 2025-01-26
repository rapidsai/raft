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

#include "../test_utils.cuh"
#include "../util/preprocess_utils.cu"

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/preprocessing.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename T2, typename T1>
void get_clean_coo(raft::resources& handle,
                   raft::device_vector_view<T1> rows,
                   raft::device_vector_view<T1> columns,
                   raft::device_vector_view<T2> values,
                   int nnz,
                   int num_rows,
                   int num_cols,
                   raft::sparse::COO<T2, T1>& coo)
{
  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  raft::sparse::op::coo_sort(int(rows.size()),
                             int(columns.size()),
                             int(values.size()),
                             rows.data_handle(),
                             columns.data_handle(),
                             values.data_handle(),
                             stream);

  // raft::sparse::COO<Type_f, Index_> coo(stream);
  raft::sparse::op::max_duplicates(handle,
                                   coo,
                                   rows.data_handle(),
                                   columns.data_handle(),
                                   values.data_handle(),
                                   nnz,
                                   num_rows,
                                   num_cols);
}
template <typename T2, typename T1>
raft::device_coo_matrix<T2, T1, T1, T1, raft::device_uvector_policy, raft::PRESERVING>
create_coo_matrix(raft::resources& handle,
                  raft::device_vector_view<T1> rows,
                  raft::device_vector_view<T1> columns,
                  raft::device_vector_view<T2> values,
                  int num_rows,
                  int num_cols)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto coo_struct_view = raft::make_device_coordinate_structure_view(
    rows.data_handle(), columns.data_handle(), num_rows, num_cols, int(rows.size()));
  auto c_matrix = raft::make_device_coo_matrix<T2, T1, T1, T1>(handle, coo_struct_view);
  raft::update_device<T2>(
    c_matrix.view().get_elements().data(), values.data_handle(), int(values.size()), stream);
  return c_matrix;
}

template <typename T2, typename T1>
raft::device_coo_matrix<T2, T1, T1, T1, raft::device_uvector_policy, raft::PRESERVING>
create_coo_matrix(raft::resources& handle, raft::sparse::COO<T2, T1>& coo)
{
  cudaStream_t stream  = raft::resource::get_cuda_stream(handle);
  auto coo_struct_view = raft::make_device_coordinate_structure_view(
    coo.rows(), coo.cols(), coo.n_rows, coo.n_cols, int(coo.nnz));
  auto c_matrix = raft::make_device_coo_matrix<T2, T1, T1, T1>(handle, coo_struct_view);
  raft::update_device<T2>(c_matrix.view().get_elements().data(), coo.vals(), coo.nnz, stream);
  return c_matrix;
}

template <typename Type_f, typename Index_>
struct SparsePreprocessInputs {
  int n_rows;
  int n_cols;
  int nnz_edges;
};

template <typename Type_f, typename Index_>
class SparsePreprocessCSR
  : public ::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>> {
 public:
  SparsePreprocessCSR()
    : params(::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle))
  {
  }

 protected:
  void SetUp() override {}

  void Run(bool bm25_on, bool coo_on)
  {
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    int num_rows        = pow(2, params.n_rows);
    int num_cols        = pow(2, params.n_cols);
    int nnz             = params.nnz_edges;
    auto a_rows         = raft::make_device_vector<Index_, int64_t>(handle, nnz);
    auto a_columns      = raft::make_device_vector<Index_, int64_t>(handle, nnz);
    auto a_values       = raft::make_device_vector<Type_f, int64_t>(handle, nnz);

    raft::util::create_dataset<Index_, Type_f>(
      handle, a_rows.view(), a_columns.view(), a_values.view(), 5, params.n_rows, params.n_cols);

    auto b_rows    = raft::make_device_vector<Index_, int64_t>(handle, nnz);
    auto b_columns = raft::make_device_vector<Index_, int64_t>(handle, nnz);
    auto b_values  = raft::make_device_vector<Type_f, int64_t>(handle, nnz);
    raft::util::create_dataset<Index_, Type_f>(handle,
                                               b_rows.view(),
                                               b_columns.view(),
                                               b_values.view(),
                                               5,
                                               params.n_rows,
                                               params.n_cols,
                                               67584);

    raft::sparse::COO<Type_f, Index_> coo_a(stream);
    get_clean_coo<Type_f, Index_>(
      handle, a_rows.view(), a_columns.view(), a_values.view(), nnz, num_rows, num_cols, coo_a);

    raft::sparse::COO<Type_f, Index_> coo_b(stream);
    get_clean_coo<Type_f, Index_>(
      handle, b_rows.view(), b_columns.view(), b_values.view(), nnz, num_rows, num_cols, coo_b);

    auto b_rows_h = raft::make_host_vector<Index_, int64_t>(handle, coo_b.nnz);
    raft::copy(b_rows_h.data_handle(), coo_b.rows(), coo_b.nnz, stream);
    for (int i = 0; i < coo_b.nnz; i++) {
      int new_val = b_rows_h(i) + coo_a.n_rows;
      b_rows_h(i) = new_val;
    }
    auto b_rows_stack = raft::make_device_vector<Index_, int64_t>(handle, coo_b.nnz);
    raft::copy(b_rows_stack.data_handle(), b_rows_h.data_handle(), coo_b.nnz, stream);

    int ab_nnz      = coo_b.nnz + coo_a.nnz;
    auto ab_rows    = raft::make_device_vector<Index_, int64_t>(handle, ab_nnz);
    auto ab_columns = raft::make_device_vector<Index_, int64_t>(handle, ab_nnz);
    auto ab_values  = raft::make_device_vector<Type_f, int64_t>(handle, ab_nnz);

    raft::copy(ab_rows.data_handle(), coo_a.rows(), coo_a.nnz, stream);
    raft::copy(ab_rows.data_handle() + coo_a.nnz, b_rows_stack.data_handle(), coo_b.nnz, stream);
    raft::copy(ab_columns.data_handle(), coo_a.cols(), coo_a.nnz, stream);
    raft::copy(ab_columns.data_handle() + coo_a.nnz, coo_b.cols(), coo_b.nnz, stream);
    raft::copy(ab_values.data_handle(), coo_a.vals(), coo_a.nnz, stream);
    raft::copy(ab_values.data_handle() + coo_a.nnz, coo_b.vals(), coo_b.nnz, stream);

    int merged_num_rows = coo_a.n_rows + coo_b.n_rows;

    auto rows_csr = raft::make_device_vector<Index_, int64_t>(handle, merged_num_rows + 1);
    raft::sparse::convert::sorted_coo_to_csr(
      ab_rows.data_handle(), ab_rows.size(), rows_csr.data_handle(), merged_num_rows + 1, stream);
    auto csr_struct_view = raft::make_device_compressed_structure_view(rows_csr.data_handle(),
                                                                       ab_columns.data_handle(),
                                                                       merged_num_rows,
                                                                       num_cols,
                                                                       int(ab_values.size()));
    auto csr_matrix =
      raft::make_device_csr_matrix<Type_f, Index_, Index_, Index_>(handle, csr_struct_view);
    raft::update_device<Type_f>(csr_matrix.view().get_elements().data(),
                                ab_values.data_handle(),
                                int(ab_values.size()),
                                stream);

    auto result = raft::make_device_vector<Type_f, int64_t>(handle, int(ab_values.size()));
    raft::sparse::matrix::SparseEncoder<float, int>* sparseEncoder =
      new raft::sparse::matrix::SparseEncoder<float, int>(num_cols);

    if (coo_on) {
      auto a_matrix = create_coo_matrix<Type_f, Index_>(handle, coo_a);
      auto b_matrix = create_coo_matrix<Type_f, Index_>(handle, coo_b);
      auto c_matrix = create_coo_matrix<Type_f, Index_>(
        handle, ab_rows.view(), ab_columns.view(), ab_values.view(), num_rows * 2, num_cols);
      sparseEncoder->fit(handle, a_matrix);
      sparseEncoder->fit(handle, b_matrix);
      sparseEncoder->transform(handle, c_matrix, result.data_handle(), bm25_on);
    } else {
      sparseEncoder->fit(handle, csr_matrix);
      sparseEncoder->transform(handle, csr_matrix, result.data_handle(), bm25_on);
    }
    delete sparseEncoder;

    if (bm25_on) {
      auto bm25_vals = raft::make_device_vector<Type_f, int64_t>(handle, int(ab_values.size()));
      raft::util::calc_tfidf_bm25<Index_, Type_f>(handle, csr_matrix.view(), bm25_vals.view());
      ASSERT_TRUE(raft::devArrMatch<Type_f>(bm25_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    } else {
      auto tfidf_vals = raft::make_device_vector<Type_f, int64_t>(handle, int(ab_values.size()));
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

using SparsePreprocessFF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessFF, Result) { Run(false, false); }

using SparsePreprocessTT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessTT, Result) { Run(true, true); }

using SparsePreprocessFT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessFT, Result) { Run(false, true); }

using SparsePreprocessTF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessTF, Result) { Run(true, false); }

using SparsePreprocessBigFF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigFF, Result) { Run(false, false); }

using SparsePreprocessBigTT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigTT, Result) { Run(true, true); }

using SparsePreprocessBigFT = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigFT, Result) { Run(false, true); }

using SparsePreprocessBigTF = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBigTF, Result) { Run(true, false); }

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs = {
  {
    7,   // n_rows_factor
    5,   // n_cols_factor
    100  // num nnz values
  },
};

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs_big = {
  {
    14,     // n_rows_factor
    14,     // n_cols_factor
    500000  // nnz_edges
  },
};

INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessFF,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessTT,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessFT,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessTF,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigTT,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigFF,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigTF,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBigFT,
                        ::testing::ValuesIn(sparse_preprocess_inputs_big));

}  // namespace sparse
}  // namespace raft