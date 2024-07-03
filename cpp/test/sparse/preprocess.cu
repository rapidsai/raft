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
// #include <thrust/reduce.h>
// #include <thrust/fill.h>
// #include <thrust/functional.h>

namespace raft {
namespace sparse {

template <typename Type_f, typename Index_>
struct SparsePreprocessInputs {
  int n_rows;
  int n_cols;
  std::vector<Index_> rows_h;
  std::vector<Index_> columns_h;
  std::vector<Type_f> values_h;
};

template <typename Type_f, typename Index_>
class SparseTest : public ::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>> {
 public:
  SparseTest()
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
    int k               = 3;
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    auto rows           = raft::make_device_vector<int, int64_t>(handle, params.rows_h.size());
    auto columns        = raft::make_device_vector<int, int64_t>(handle, params.columns_h.size());
    auto values         = raft::make_device_vector<float, int64_t>(handle, params.values_h.size());
    auto result         = raft::make_device_vector<float, int64_t>(handle, params.values_h.size());

    raft::copy(rows.data_handle(), params.rows_h.data(), params.rows_h.size(), stream);
    raft::copy(columns.data_handle(), params.columns_h.data(), params.columns_h.size(), stream);

    raft::copy(values.data_handle(), params.values_h.data(), params.values_h.size(), stream);

    auto coo_struct_view = raft::make_device_coordinate_structure_view(
      rows.data_handle(), columns.data_handle(), params.n_rows, params.n_cols, int(values.size()));
    auto coo_matrix = raft::make_device_coo_matrix<float, int, int, int>(handle, coo_struct_view);
    raft::update_device<float>(
      coo_matrix.view().get_elements().data(), values.data_handle(), values.size(), stream);

    if (bm25_on) {
      sparse::matrix::encode_bm25<int, float>(handle, coo_matrix.view(), result.view());
    } else {
      sparse::matrix::encode_tfidf<int, float>(handle, coo_matrix.view(), result.view());
    }

    auto out_rows_coo =
      raft::make_device_vector<int, int64_t>(handle, coo_matrix.structure_view().get_n_rows() * k);
    auto out_dists_coo = raft::make_device_vector<float, int64_t>(
      handle, coo_matrix.structure_view().get_n_rows() * k);

    raft::sparse::neighbors::brute_force_knn<int, float>(coo_matrix,
                                                         coo_matrix,
                                                         out_rows_coo.data_handle(),
                                                         out_dists_coo.data_handle(),
                                                         k,
                                                         handle,
                                                         coo_matrix.structure_view().get_n_rows(),
                                                         coo_matrix.structure_view().get_n_rows(),
                                                         raft::distance::DistanceType::L1);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    ASSERT_TRUE(values.size() == result.size());
    //   raft::devArrMatch<Type_f>(, nnz, raft::Compare<Type_f>()));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SparsePreprocessInputs<Type_f, Index_> params;
  int n_rows, n_cols;
  rmm::device_uvector<Index_> rows, columns;
  rmm::device_uvector<Type_f> values, result;
  bool bm25;
};

using SparseTestFF = SparseTest<float, int>;
TEST_P(SparseTestFF, Result) { Run(false); }

using SparseTestFT = SparseTest<float, int>;
TEST_P(SparseTestFT, Result) { Run(true); }

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs = {
  {9,                                                    // n_rows
   4,                                                    // n_cols
   {0, 3, 4, 5, 6, 7, 8, 9, 10, 11},                     // rows
   {0, 0, 1, 2, 2, 1, 1, 3, 2, 1},                       // cols
   {1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 4.0, 2.0, 1.0, 3.0}},  // vals
};

INSTANTIATE_TEST_CASE_P(SparseTest, SparseTestFF, ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparseTest, SparseTestFT, ::testing::ValuesIn(sparse_preprocess_inputs));

}  // namespace sparse
}  // namespace raft