/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/transpose.h>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"

namespace raft {
namespace linalg {

template <typename T>
struct TranposeInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const TranposeInputs<T> &dims) {
  return os;
}

template <typename T>
class TransposeTest : public ::testing::TestWithParam<TranposeInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<TranposeInputs<T>>::GetParam();

    stream = handle.get_stream();

    int len = params.len;

    raft::allocate(data, len);
    ASSERT(params.len == 9, "This test works only with len=9!");
    T data_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    raft::update_device(data, data_h, len, stream);

    raft::allocate(data_trans_ref, len);
    T data_ref_h[] = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
    raft::update_device(data_trans_ref, data_ref_h, len, stream);

    raft::allocate(data_trans, len);

    transpose(handle, data, data_trans, params.n_row, params.n_col, stream);
    transpose(data, params.n_row, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(data_trans));
    CUDA_CHECK(cudaFree(data_trans_ref));
  }

 protected:
  TranposeInputs<T> params;
  T *data, *data_trans, *data_trans_ref;
  raft::handle_t handle;
  cudaStream_t stream;
};

const std::vector<TranposeInputs<float>> inputsf2 = {
  {0.1f, 3 * 3, 3, 3, 1234ULL}};

const std::vector<TranposeInputs<double>> inputsd2 = {
  {0.1, 3 * 3, 3, 3, 1234ULL}};

typedef TransposeTest<float> TransposeTestValF;
TEST_P(TransposeTestValF, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(data_trans_ref, data_trans, params.len,
                      raft::CompareApproxAbs<float>(params.tolerance)));

  ASSERT_TRUE(
    raft::devArrMatch(data_trans_ref, data, params.len,
                      raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef TransposeTest<double> TransposeTestValD;
TEST_P(TransposeTestValD, Result) {
  ASSERT_TRUE(
    raft::devArrMatch(data_trans_ref, data_trans, params.len,
                      raft::CompareApproxAbs<double>(params.tolerance)));

  ASSERT_TRUE(
    raft::devArrMatch(data_trans_ref, data, params.len,
                      raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValF,
                         ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValD,
                         ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
