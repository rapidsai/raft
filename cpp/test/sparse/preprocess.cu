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
  std::vector<Index_> rows_h;
  std::vector<Index_> columns_h;
  std::vector<Type_f> values_h;
};

template <typename Type_f, typename Index_>
class SparsePreprocessTest
  : public ::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>> {
 public:
  SparsePreprocessTest()
    : params(::testing::TestWithParam<SparsePreprocessInputs<Type_f, Index_>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rows(params.rows_h.size(), stream),
      columns(params.columns_h.size(), stream),
      values(params.values_h.size(), stream),
      result(params.values_h.size(), stream)
  {
  }

 protected:
  void SetUp() override {}

  void Run()
  {
    auto rows    = raft::make_device_vector<int, int64_t>(handle, params.rows_h.size());
    auto columns = raft::make_device_vector<int, int64_t>(handle, params.columns_h.size());
    auto values  = raft::make_device_vector<float, int64_t>(handle, params.values_h.size());
    auto result  = raft::make_device_vector<float, int64_t>(handle, params.values_h.size());

    sparse::matrix::encode_bm25<int, float>(
      handle, rows.view(), columns.view(), values.view(), result.view());

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    ASSERT_TRUE(values.size() == values.size());
    //   raft::devArrMatch<Type_f>(, nnz, raft::Compare<Type_f>()));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SparsePreprocessInputs<Type_f, Index_> params;
  rmm::device_uvector<Index_> rows, columns;
  rmm::device_uvector<Type_f> values, result;
};
using SparsePreprocessTestF = SparsePreprocessTest<float, int>;
TEST_P(SparsePreprocessTestF, Result) { Run(); }

const std::vector<SparsePreprocessInputs<float, int>> sparse_preprocess_inputs = {
  {{0, 3, 4, 5, 6, 7, 8, 9, 10, 11},
   {0, 0, 1, 2, 2, 1, 1, 3, 2, 1},
   {1.0, 2.0, 2.0, 1.0, 1.0, 3.0, 4.0, 2.0, 1.0, 3.0}},
};

INSTANTIATE_TEST_CASE_P(SparsePreprocessTest,
                        SparsePreprocessTestF,
                        ::testing::ValuesIn(sparse_preprocess_inputs));
}  // namespace sparse
}  // namespace raft