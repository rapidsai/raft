/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <cusparse_v2.h>

#include <gtest/gtest.h>

#include <raft/cudart_utils.h>
#include <raft/sparse/cusparse_wrappers.h>

#include <raft/sparse/linalg/transpose.h>

#include "../test_utils.h"

namespace raft {
namespace sparse {

using namespace raft;
using namespace raft::sparse;

template <typename value_idx, typename value_t>
struct CSRTransposeInputs {
  value_idx nrows;
  value_idx ncols;
  value_idx nnz;

  std::vector<value_idx> indptr_h;
  std::vector<value_idx> indices_h;
  std::vector<value_t> data_h;

  std::vector<value_idx> out_indptr_ref_h;
  std::vector<value_idx> out_indices_ref_h;
  std::vector<value_t> out_data_ref_h;
};

template <typename value_idx, typename value_t>
::std::ostream &operator<<(::std::ostream &os,
                           const CSRTransposeInputs<value_idx, value_t> &dims) {
  return os;
}

template <typename value_idx, typename value_t>
class CSRTransposeTest
  : public ::testing::TestWithParam<CSRTransposeInputs<value_idx, value_t>> {
 protected:
  void make_data() {
    std::vector<value_idx> indptr_h = params.indptr_h;
    std::vector<value_idx> indices_h = params.indices_h;
    std::vector<value_t> data_h = params.data_h;

    raft::allocate(indptr, indptr_h.size(), stream);
    raft::allocate(indices, indices_h.size(), stream);
    raft::allocate(data, data_h.size(), stream);

    update_device(indptr, indptr_h.data(), indptr_h.size(), stream);
    update_device(indices, indices_h.data(), indices_h.size(), stream);
    update_device(data, data_h.data(), data_h.size(), stream);

    std::vector<value_idx> out_indptr_ref_h = params.out_indptr_ref_h;
    std::vector<value_idx> out_indices_ref_h = params.out_indices_ref_h;
    std::vector<value_t> out_data_ref_h = params.out_data_ref_h;

    raft::allocate(out_indptr_ref, out_indptr_ref_h.size(), stream);
    raft::allocate(out_indices_ref, out_indices_ref_h.size(), stream);
    raft::allocate(out_data_ref, out_data_ref_h.size(), stream);

    update_device(out_indptr_ref, out_indptr_ref_h.data(),
                  out_indptr_ref_h.size(), stream);
    update_device(out_indices_ref, out_indices_ref_h.data(),
                  out_indices_ref_h.size(), stream);
    update_device(out_data_ref, out_data_ref_h.data(), out_data_ref_h.size(),
                  stream);

    raft::allocate(out_indptr, out_indptr_ref_h.size(), stream);
    raft::allocate(out_indices, out_indices_ref_h.size(), stream);
    raft::allocate(out_data, out_data_ref_h.size(), stream);
  }

  void SetUp() override {
    params = ::testing::TestWithParam<
      CSRTransposeInputs<value_idx, value_t>>::GetParam();
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUSPARSE_CHECK(cusparseCreate(&handle));

    make_data();

    raft::sparse::linalg::csr_transpose(
      handle, indptr, indices, data, out_indptr, out_indices, out_data,
      params.nrows, params.ncols, params.nnz, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUSPARSE_CHECK(cusparseDestroy(handle));
  }

  void TearDown() override {
    raft::deallocate_all(stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void compare() {
    ASSERT_TRUE(devArrMatch(out_indptr, out_indptr_ref,
                            params.out_indptr_ref_h.size(),
                            Compare<value_t>()));
    ASSERT_TRUE(devArrMatch(out_indices, out_indices_ref,
                            params.out_indices_ref_h.size(),
                            Compare<value_t>()));
    ASSERT_TRUE(devArrMatch(out_data, out_data_ref,
                            params.out_data_ref_h.size(), Compare<value_t>()));
  }

 protected:
  cudaStream_t stream;
  cusparseHandle_t handle;

  // input data
  value_idx *indptr, *indices;
  value_t *data;

  // output data
  value_idx *out_indptr, *out_indices;
  value_t *out_data;

  // expected output data
  value_idx *out_indptr_ref, *out_indices_ref;
  value_t *out_data_ref;

  CSRTransposeInputs<value_idx, value_t> params;
};

const std::vector<CSRTransposeInputs<int, float>> inputs_i32_f = {
  {
    4,
    2,
    8,
    {0, 2, 4, 6, 8},
    {0, 1, 0, 1, 0, 1, 0, 1},  // indices
    {1.0f, 3.0f, 1.0f, 5.0f, 50.0f, 28.0f, 16.0f, 2.0f},
    {0, 4, 8},
    {0, 1, 2, 3, 0, 1, 2, 3},  // indices
    {1.0f, 1.0f, 50.0f, 16.0f, 3.0f, 5.0f, 28.0f, 2.0f},
  },
};
typedef CSRTransposeTest<int, float> CSRTransposeTestF;
TEST_P(CSRTransposeTestF, Result) { compare(); }
INSTANTIATE_TEST_CASE_P(CSRTransposeTest, CSRTransposeTestF,
                        ::testing::ValuesIn(inputs_i32_f));

};  // end namespace sparse
};  // end namespace raft
