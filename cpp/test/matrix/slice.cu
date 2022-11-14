/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <raft/matrix/slice.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_scalar.hpp>

namespace raft {
namespace matrix {

template <typename T>
struct SliceInputs {
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SliceInputs<T>& I)
{
  os << "{ " << I.rows << ", " << I.cols << ", " << I.seed << '}' << std::endl;
  return os;
}

// Col-major slice reference test
template <typename Type>
void naiveSlice(const Type* in, Type* out, int rows, int cols, int x1, int y1, int x2, int y2)
{
  int out_rows = x2 - x1;
  // int out_cols = y2 - y1;
  for (int j = y1; j < y2; ++j) {
    for (int i = x1; i < x2; ++i) {
      out[(i - x1) + (j - y1) * out_rows] = in[i + j * rows];
    }
  }
}

template <typename T>
class SliceTest : public ::testing::TestWithParam<SliceInputs<T>> {
 public:
  SliceTest()
    : params(::testing::TestWithParam<SliceInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      data(params.rows * params.cols, stream)
  {
  }

  void SetUp() override
  {
    std::random_device rd;
    std::default_random_engine dre(rd());
    raft::random::RngState r(params.seed);
    int rows = params.rows, cols = params.cols, len = rows * cols;
    uniform(handle, r, data.data(), len, T(-10.0), T(10.0));

    std::uniform_int_distribution<int> rowGenerator(0, (rows / 2) - 1);
    auto row1 = rowGenerator(dre);
    auto row2 = rowGenerator(dre) + rows / 2;

    std::uniform_int_distribution<int> colGenerator(0, (cols / 2) - 1);
    auto col1 = colGenerator(dre);
    auto col2 = colGenerator(dre) + cols / 2;

    rmm::device_uvector<T> d_act_result((row2 - row1) * (col2 - col1), stream);
    act_result.resize((row2 - row1) * (col2 - col1));
    exp_result.resize((row2 - row1) * (col2 - col1));

    std::vector<T> h_data(rows * cols);
    raft::update_host(h_data.data(), data.data(), rows * cols, stream);
    naiveSlice(h_data.data(), exp_result.data(), rows, cols, row1, col1, row2, col2);
    auto input =
      raft::make_device_matrix_view<const T, int, raft::col_major>(data.data(), rows, cols);
    auto output = raft::make_device_matrix_view<T, int, raft::col_major>(
      d_act_result.data(), row2 - row1, col2 - col1);
    slice(handle, input, output, slice_coordinates(row1, col1, row2, col2));

    raft::update_host(act_result.data(), d_act_result.data(), d_act_result.size(), stream);
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  SliceInputs<T> params;
  rmm::device_uvector<T> data;
  std::vector<T> exp_result, act_result;
};

///// Row- and column-wise tests
const std::vector<SliceInputs<float>> inputsf = {{32, 1024, 1234ULL},
                                                 {64, 1024, 1234ULL},
                                                 {128, 1024, 1234ULL},
                                                 {256, 1024, 1234ULL},
                                                 {512, 512, 1234ULL},
                                                 {1024, 32, 1234ULL},
                                                 {1024, 64, 1234ULL},
                                                 {1024, 128, 1234ULL},
                                                 {1024, 256, 1234ULL}};

const std::vector<SliceInputs<double>> inputsd = {
  {32, 1024, 1234ULL},
  {64, 1024, 1234ULL},
  {128, 1024, 1234ULL},
  {256, 1024, 1234ULL},
  {512, 512, 1234ULL},
  {1024, 32, 1234ULL},
  {1024, 64, 1234ULL},
  {1024, 128, 1234ULL},
  {1024, 256, 1234ULL},
};

typedef SliceTest<float> SliceTestF;
TEST_P(SliceTestF, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<float>()));
}

typedef SliceTest<double> SliceTestD;
TEST_P(SliceTestD, Result)
{
  ASSERT_TRUE(hostVecMatch(exp_result, act_result, raft::Compare<double>()));
}

INSTANTIATE_TEST_CASE_P(SliceTests, SliceTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SliceTests, SliceTestD, ::testing::ValuesIn(inputsd));

}  // end namespace matrix
}  // end namespace raft
