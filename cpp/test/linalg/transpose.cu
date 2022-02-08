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

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/linalg/transpose.hpp>
#include <raft/random/rng.hpp>

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
::std::ostream& operator<<(::std::ostream& os, const TranposeInputs<T>& dims)
{
  return os;
}

template <typename T>
class TransposeTest : public ::testing::TestWithParam<TranposeInputs<T>> {
 public:
  TransposeTest()
    : params(::testing::TestWithParam<TranposeInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      data(params.len, stream),
      data_trans_ref(params.len, stream),
      data_trans(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    int len = params.len;
    ASSERT(params.len == 9, "This test works only with len=9!");
    T data_h[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0};
    raft::update_device(data.data(), data_h, len, stream);
    T data_ref_h[] = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0};
    raft::update_device(data_trans_ref.data(), data_ref_h, len, stream);

    transpose(handle, data.data(), data_trans.data(), params.n_row, params.n_col, stream);
    transpose(data.data(), params.n_row, stream);
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  TranposeInputs<T> params;
  rmm::device_uvector<T> data, data_trans, data_trans_ref;
};

const std::vector<TranposeInputs<float>> inputsf2 = {{0.1f, 3 * 3, 3, 3, 1234ULL}};

const std::vector<TranposeInputs<double>> inputsd2 = {{0.1, 3 * 3, 3, 3, 1234ULL}};

typedef TransposeTest<float> TransposeTestValF;
TEST_P(TransposeTestValF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.len,
                                raft::CompareApproxAbs<float>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data.data(),
                                params.len,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef TransposeTest<double> TransposeTestValD;
TEST_P(TransposeTestValD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data_trans.data(),
                                params.len,
                                raft::CompareApproxAbs<double>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(data_trans_ref.data(),
                                data.data(),
                                params.len,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(TransposeTests, TransposeTestValD, ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
