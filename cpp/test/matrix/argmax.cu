/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
#include <cstdint>
#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/argmax.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace matrix {

template <typename T, typename IdxT>
struct ArgMaxInputs {
  std::vector<T> input_matrix;
  std::vector<IdxT> output_matrix;
  std::size_t n_rows;
  std::size_t n_cols;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const ArgMaxInputs<T, IdxT>& dims)
{
  return os;
}

template <typename T, typename IdxT>
class ArgMaxTest : public ::testing::TestWithParam<ArgMaxInputs<T, IdxT>> {
 public:
  ArgMaxTest()
    : params(::testing::TestWithParam<ArgMaxInputs<T, IdxT>>::GetParam()),
      input(raft::make_device_matrix<T, std::uint32_t, row_major>(
        handle, params.n_rows, params.n_cols)),
      output(raft::make_device_vector<IdxT, std::uint32_t>(handle, params.n_rows)),
      expected(raft::make_device_vector<IdxT, std::uint32_t>(handle, params.n_rows))
  {
    raft::update_device(input.data_handle(),
                        params.input_matrix.data(),
                        params.input_matrix.size(),
                        resource::get_cuda_stream(handle));
    raft::update_device(expected.data_handle(),
                        params.output_matrix.data(),
                        params.output_matrix.size(),
                        resource::get_cuda_stream(handle));

    auto input_const_view = raft::make_device_matrix_view<const T, std::uint32_t, row_major>(
      input.data_handle(), input.extent(0), input.extent(1));

    raft::matrix::argmax(handle, input_const_view, output.view());

    resource::sync_stream(handle);
  }

 protected:
  raft::resources handle;
  ArgMaxInputs<T, IdxT> params;

  raft::device_matrix<T, std::uint32_t, row_major> input;
  raft::device_vector<IdxT, std::uint32_t> output;
  raft::device_vector<IdxT, std::uint32_t> expected;
};

const std::vector<ArgMaxInputs<float, int>> inputsf = {
  {{0.1f, 0.2f, 0.3f, 0.4f, 0.4f, 0.3f, 0.2f, 0.1f, 0.2f, 0.3f, 0.5f, 0.0f}, {3, 0, 2}, 3, 4}};

const std::vector<ArgMaxInputs<double, int>> inputsd = {
  {{0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1, 0.2, 0.3, 0.5, 0.0}, {3, 0, 2}, 3, 4}};

typedef ArgMaxTest<float, int> ArgMaxTestF;
TEST_P(ArgMaxTestF, Result)
{
  ASSERT_TRUE(devArrMatch(expected.data_handle(),
                          output.data_handle(),
                          params.n_rows,
                          Compare<int>(),
                          resource::get_cuda_stream(handle)));
}

typedef ArgMaxTest<double, int> ArgMaxTestD;
TEST_P(ArgMaxTestD, Result)
{
  ASSERT_TRUE(devArrMatch(expected.data_handle(),
                          output.data_handle(),
                          params.n_rows,
                          Compare<int>(),
                          resource::get_cuda_stream(handle)));
}

INSTANTIATE_TEST_SUITE_P(ArgMaxTest, ArgMaxTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(ArgMaxTest, ArgMaxTestD, ::testing::ValuesIn(inputsd));

}  // namespace matrix
}  // namespace raft