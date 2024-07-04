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

#include <raft/core/device_coo_matrix.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/matrix/preprocessing.cuh>
#include <raft/sparse/selection/knn.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Type_f, typename Index_>
struct SparsePreprocessInputs {
  int n_rows;
  int n_cols;
  std::vector<Index_> rows_h;
  std::vector<Index_> columns_h;
  // std::vector<Type_f> tfidf_vals_h;
  // std::vector<Index_> out_idxs_bm25_h;
  // std::vector<Type_f> out_dists_bm25_h;
  // std::vector<Index_> out_idxs_tfidf_h;
  // std::vector<Type_f> out_dists_tfidf_h;
  std::vector<Type_f> values_h;
};

template <typename Type_f, typename Index_>
class SparsePreprocessCoo
  : public ::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>> {
 public:
  SparsePreprocessCoo()
    : params(::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      n_rows(params.n_rows),
      n_cols(params.n_cols),
      rows(params.rows_h.size(), stream),
      columns(params.columns_h.size(), stream),
      values(params.values_h.size(), stream),
      result(params.values_h.size(), stream)
  {
  }

 protected:
  void SetUp() override {}

  void Run(bool bm25_on)
  {
    int k                                 = 2;
    std::vector<Type_f> bm25_vals_h       = {0.850086,
                                             1.15525,
                                             0.682645,
                                             0.860915,
                                             0.99021,
                                             0.860915,
                                             0.850086,
                                             0.850086,
                                             0.850086,
                                             1.25152};  // bm25
    std::vector<Type_f> tfidf_vals_h      = {0.480453,
                                             0.7615,
                                             0.7615,
                                             0.960906,
                                             1.11558,
                                             0.960906,
                                             0.480453,
                                             0.480453,
                                             0.480453,
                                             0.7615};  // tfidf
    std::vector<Index_> out_idxs_bm25_h   = {0, 3, 1, 2, 1, 2, 3, 0, 4,  8, 5, 10,
                                             6, 5, 7, 8, 8, 7, 9, 1, 10, 5, 0, 0};
    std::vector<Type_f> out_dists_bm25_h  = {0, 0.305159,  0, 0,         0, 0,         0, 0.305159,
                                             0, 0.167441,  0, 0.0108292, 0, 0.129295,  0, 0.0108292,
                                             0, 0.0108292, 0, 0.850086,  0, 0.0108292, 0, 0};
    std::vector<Index_> out_idxs_tfidf_h  = {0, 3, 1, 2, 1, 2, 3, 0, 4,  7, 5, 6,
                                             6, 5, 7, 4, 8, 4, 9, 1, 10, 1, 0, 0};
    std::vector<Type_f> out_dists_tfidf_h = {0, 0.281047, 0, 0,        0, 0,        0, 0.281047,
                                             0, 0.199406, 0, 0.154671, 0, 0.154671, 0, 0.199406,
                                             0, 0.281047, 0, 0.480453, 0, 0.480453, 0, 0};

    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    auto rows           = raft::make_device_vector<Index_, int64_t>(handle, params.rows_h.size());
    auto columns       = raft::make_device_vector<Index_, int64_t>(handle, params.columns_h.size());
    auto values        = raft::make_device_vector<Type_f, int64_t>(handle, params.values_h.size());
    auto result        = raft::make_device_vector<Type_f, int64_t>(handle, params.values_h.size());
    auto bm25_vals     = raft::make_device_vector<Type_f, int64_t>(handle, bm25_vals_h.size());
    auto tfidf_vals    = raft::make_device_vector<Type_f, int64_t>(handle, tfidf_vals_h.size());
    auto out_idxs_bm25 = raft::make_device_vector<Index_, int64_t>(handle, out_idxs_bm25_h.size());
    auto out_idxs_tfidf =
      raft::make_device_vector<Index_, int64_t>(handle, out_idxs_tfidf_h.size());
    auto out_dists_bm25 =
      raft::make_device_vector<Type_f, int64_t>(handle, out_dists_bm25_h.size());
    auto out_dists_tfidf =
      raft::make_device_vector<Type_f, int64_t>(handle, out_dists_tfidf_h.size());

    raft::copy(rows.data_handle(), params.rows_h.data(), params.rows_h.size(), stream);
    raft::copy(columns.data_handle(), params.columns_h.data(), params.columns_h.size(), stream);
    raft::copy(values.data_handle(), params.values_h.data(), params.values_h.size(), stream);
    raft::copy(bm25_vals.data_handle(), bm25_vals_h.data(), bm25_vals_h.size(), stream);
    raft::copy(tfidf_vals.data_handle(), tfidf_vals_h.data(), tfidf_vals_h.size(), stream);
    raft::copy(out_idxs_bm25.data_handle(), out_idxs_bm25_h.data(), out_idxs_bm25_h.size(), stream);
    raft::copy(
      out_idxs_tfidf.data_handle(), out_idxs_tfidf_h.data(), out_idxs_tfidf_h.size(), stream);
    raft::copy(
      out_dists_bm25.data_handle(), out_dists_bm25_h.data(), out_dists_bm25_h.size(), stream);
    raft::copy(
      out_dists_tfidf.data_handle(), out_dists_tfidf_h.data(), out_dists_tfidf_h.size(), stream);

    auto coo_struct_view = raft::make_device_coordinate_structure_view(
      rows.data_handle(), columns.data_handle(), params.n_rows, params.n_cols, int(values.size()));
    auto c_matrix =
      raft::make_device_coo_matrix<Type_f, Index_, Index_, Index_>(handle, coo_struct_view);
    raft::update_device<Type_f>(
      c_matrix.view().get_elements().data(), values.data_handle(), values.size(), stream);

    if (bm25_on) {
      sparse::matrix::encode_bm25<Index_, Type_f>(handle, c_matrix.view(), result.view());
      ASSERT_TRUE(raft::devArrMatch<Type_f>(bm25_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    } else {
      sparse::matrix::encode_tfidf<Index_, Type_f>(handle, c_matrix.view(), result.view());
      ASSERT_TRUE(raft::devArrMatch<Type_f>(tfidf_vals.data_handle(),
                                            result.data_handle(),
                                            result.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    }

    raft::update_device<Type_f>(
      c_matrix.view().get_elements().data(), result.data_handle(), result.size(), stream);

    auto out_indices =
      raft::make_device_vector<Index_, int64_t>(handle, c_matrix.structure_view().get_n_rows() * k);
    auto out_dists =
      raft::make_device_vector<Type_f, int64_t>(handle, c_matrix.structure_view().get_n_rows() * k);

    raft::sparse::neighbors::brute_force_knn<Index_, Type_f>(c_matrix,
                                                             c_matrix,
                                                             out_indices.data_handle(),
                                                             out_dists.data_handle(),
                                                             k,
                                                             handle,
                                                             c_matrix.structure_view().get_n_rows(),
                                                             c_matrix.structure_view().get_n_rows(),
                                                             raft::distance::DistanceType::L1);

    if (bm25_on) {
      ASSERT_TRUE(raft::devArrMatch<Index_>(out_idxs_bm25.data_handle(),
                                            out_indices.data_handle(),
                                            out_indices.size(),
                                            raft::Compare<Index_>(),
                                            stream));
      ASSERT_TRUE(raft::devArrMatch<Type_f>(out_dists_bm25.data_handle(),
                                            out_dists.data_handle(),
                                            out_dists.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    } else {
      ASSERT_TRUE(raft::devArrMatch<Index_>(out_idxs_tfidf.data_handle(),
                                            out_indices.data_handle(),
                                            out_indices.size(),
                                            raft::Compare<Index_>(),
                                            stream));
      ASSERT_TRUE(raft::devArrMatch<Type_f>(out_dists_tfidf.data_handle(),
                                            out_dists.data_handle(),
                                            out_dists.size(),
                                            raft::CompareApprox<Type_f>(2e-5),
                                            stream));
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SparsePreprocessInputs<Type_f, Index_> params;
  rmm::device_uvector<Index_> rows, columns;
  rmm::device_uvector<Type_f> values, result;
  int n_rows, n_cols;
  // rmm::device_uvector<Index_> rows, columns, out_idxs_bm25, out_idxs_tfidf;
  // rmm::device_uvector<Type_f> values, result, bm25_vals, tfidf_vals, out_dists_bm25,
  // out_dists_tfidf;
  bool bm25;
};

using SparsePreprocessTfidfCoo = SparsePreprocessCoo<float, int>;
TEST_P(SparsePreprocessTfidfCoo, Result) { Run(false); }

using SparsePreprocessBm25Coo = SparsePreprocessCoo<float, int>;
TEST_P(SparsePreprocessBm25Coo, Result) { Run(true); }

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs = {
  {12,                                // n_rows
   5,                                 // n_cols
   {0, 3, 4, 5, 6, 7, 8, 9, 10, 11},  // rows
   {0, 0, 1, 2, 2, 1, 1, 3, 2, 1},
   // out_dists_tfidf
   {1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 4.0, 2.0, 1.0, 3.0}},  // vals
};

INSTANTIATE_TEST_CASE_P(SparsePreprocessCoo,
                        SparsePreprocessTfidfCoo,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCoo,
                        SparsePreprocessBm25Coo,
                        ::testing::ValuesIn(sparse_preprocess_inputs));

}  // namespace sparse
}  // namespace raft