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
  std::vector<Type_f> values_h;
};

template <typename Type_f, typename Index_>
class SparsePreprocessCSR
  : public ::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>> {
 public:
  SparsePreprocessCSR()
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
    int k               = 2;
    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    auto indptr         = raft::make_device_vector<int, int64_t>(handle, params.rows_h.size());
    auto indices        = raft::make_device_vector<int, int64_t>(handle, params.columns_h.size());
    auto values         = raft::make_device_vector<float, int64_t>(handle, params.values_h.size());
    auto result         = raft::make_device_vector<float, int64_t>(handle, params.values_h.size());

    raft::copy(indptr.data_handle(), params.rows_h.data(), params.rows_h.size(), stream);
    raft::copy(indices.data_handle(), params.columns_h.data(), params.columns_h.size(), stream);
    raft::copy(values.data_handle(), params.values_h.data(), params.values_h.size(), stream);

    auto csr_struct_view = raft::make_device_compressed_structure_view(indptr.data_handle(),
                                                                       indices.data_handle(),
                                                                       params.n_rows,
                                                                       params.n_cols,
                                                                       int(values.size()));
    auto c_matrix = raft::make_device_csr_matrix<float, int, int, int>(handle, csr_struct_view);

    raft::update_device<float>(
      c_matrix.view().get_elements().data(), values.data_handle(), values.size(), stream);

    if (bm25_on) {
      sparse::matrix::encode_bm25<int, float>(handle, c_matrix.view(), result.view());
    } else {
      sparse::matrix::encode_tfidf<int, float>(handle, c_matrix.view(), result.view());
    }

    raft::update_device<float>(
      c_matrix.view().get_elements().data(), result.data_handle(), result.size(), stream);

    auto out_indices =
      raft::make_device_vector<int, int64_t>(handle, c_matrix.structure_view().get_n_rows() * k);
    auto out_dists =
      raft::make_device_vector<float, int64_t>(handle, c_matrix.structure_view().get_n_rows() * k);

    raft::sparse::neighbors::brute_force_knn<int, float>(c_matrix,
                                                         c_matrix,
                                                         out_indices.data_handle(),
                                                         out_dists.data_handle(),
                                                         k,
                                                         handle,
                                                         c_matrix.structure_view().get_n_rows(),
                                                         c_matrix.structure_view().get_n_rows(),
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

using SparsePreprocessTfidfCsr = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessTfidfCsr, Result) { Run(false); }

using SparsePreprocessBm25Csr = SparsePreprocessCSR<float, int>;
TEST_P(SparsePreprocessBm25Csr, Result) { Run(true); }

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs = {
  {12,                                                   // n_rows
   5,                                                    // n_cols
   {0, 1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9},                 // rows
   {0, 0, 1, 2, 2, 1, 1, 3, 2, 1},                       // cols
   {1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 4.0, 2.0, 1.0, 3.0}},  // vals
  //  {0.850086, 1.15525, 0.682645, 0.860915, 0.99021, 0.860915, 0.850086, 0.850086,
  //  0.850086, 1.25152}, // bm25 {0.480453, 0.7615, 0.7615, 0.960906, 1.11558, 0.960906, 0.480453,
  //  0.480453, 0.480453, 0.7615},   // tfidf {0, 3, 1, 2, 1, 2, 3, 0, 4, 7, 5, 6, 6, 5, 7, 4, 8, 4,
  //  9, 1, 10, 1, 0, 0}, //out_idx {0, 0.281047, 0, 0, 0, 0, 0, 0.281047, 0, 0.199406, 0, 0.154671,
  //  0, 0.154671, 0, 0.199406, 0, 0.281047, 0, 0.480453, 0, 0.480453, 0, 0}}, //out_dists
};

INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessTfidfCsr,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
INSTANTIATE_TEST_CASE_P(SparsePreprocessCSR,
                        SparsePreprocessBm25Csr,
                        ::testing::ValuesIn(sparse_preprocess_inputs));

}  // namespace sparse
}  // namespace raft