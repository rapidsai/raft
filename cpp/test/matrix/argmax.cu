/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/matrix/argmax.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace matrix {

template <typename T, typename IdxT>
struct ArgMaxInputs {
  std::vector<T> input_matrix;
  std::vector<IdxT> output_matrix;
  int n_cols;
  int n_rows;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const ArgMaxInputs<T>& dims)
{
  return os;
}

template <typename T, typename IdxT>
class ArgMaxTest : public ::testing::TestWithParam<ArgMaxInputs<T, IdxT>> {
 public:
  ArgMaxTest()
    : params(::testing::TestWithParam<ArgMaxInputs<T, IdxT>>::GetParam()),
      input(std::move(raft::make_device_matrix<T>(handle, params.n_rows, params.n_cols))),
      output(std::move(raft::make_device_vector<IdxT>(handle, params.n_rows))),
      expected(std::move(raft::make_device_vector<IdxT>(handle, params.n_rows)))
  {
    raft::copy(input.data_handle(), params.input_matrix.data(), params.n_rows * params.n_cols);
    raft::copy(expected.data_handle(), params.output_matrix.data(), params.n_rows * params.n_cols);

    raft::matrix::argmax(handle, input, output);
  }

 protected:
  raft::handle_t handle;
  ArgMaxInputs<T> params;

  raft::device_matrix<T> input;
  raft::device_vector<IdxT> output;
  raft::device_vector<IdxT> expected;
};

const std::vector<ArgMaxInputs<float, int>> inputsf = {
  {0.1f, 0.2f, 0.3f, 0.4f}, {0.4f, 0.3f, 0.2f, 0.1f}, {0.2f, 0.3f, 0.5f, 0.0f}, {3, 0, 2}, 3, 4};

const std::vector<ArgMaxInputs<double, int>> inputsd = {
  {0.1, 0.2, 0.3, 0.4}, {0.4, 0.3, 0.2, 0.1}, {0.2, 0.3, 0.5, 0.0}, {3, 0, 2}, 3, 4};

typedef ArgMaxTest<float, int> ArgMaxTestF;
TEST_P(ArgMaxTestF, Result)
{
  ASSERT_TRUE(devArrMatch(output.data_handle(),
                          expected.data_handle(),
                          params.n_rows,
                          Compare<int>(),
                          handle.get_stream()));
}

typedef ArgMaxTest<double, int> ArgMaxTestD;
TEST_P(ArgMaxTestD, Result)
{
  ASSERT_TRUE(devArrMatch(output.data_handle(),
                          expected.data_handle(),
                          params.n_rows,
                          Compare<int>(),
                          handle.get_stream()));
}

INSTANTIATE_TEST_SUITE_P(ArgMaxTest, ArgMaxTestTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(ArgMaxTest, ArgMaxTestTestD, ::testing::ValuesIn(inputsd));

}  // namespace matrix
}  // namespace raft