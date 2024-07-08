/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
#include <raft/linalg/eltwise.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace stats {

template <typename T>
struct SumInputs {
  T tolerance;
  int rows, cols;
  bool rowMajor;
  T value = T(1);
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SumInputs<T>& dims)
{
  return os << "{ " << dims.tolerance << ", " << dims.rows << ", " << dims.cols << ", "
            << dims.rowMajor << "}" << std::endl;
}

template <typename T>
class SumTest : public ::testing::TestWithParam<SumInputs<T>> {
 public:
  SumTest()
    : params(::testing::TestWithParam<SumInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rows(params.rows),
      cols(params.cols),
      data(rows * cols, stream),
      sum_act(cols, stream)
  {
  }

 protected:
  void runTest()
  {
    int len = rows * cols;

    std::vector<T> data_h(len);
    for (int i = 0; i < len; i++) {
      data_h[i] = T(params.value);
    }

    raft::update_device(data.data(), data_h.data(), len, stream);

    if (params.rowMajor) {
      using layout = raft::row_major;
      sum(handle,
          raft::make_device_matrix_view<const T, int, layout>(data.data(), rows, cols),
          raft::make_device_vector_view(sum_act.data(), cols));
    } else {
      using layout = raft::col_major;
      sum(handle,
          raft::make_device_matrix_view<const T, int, layout>(data.data(), rows, cols),
          raft::make_device_vector_view(sum_act.data(), cols));
    }
    resource::sync_stream(handle, stream);

    double expected = double(params.rows) * params.value;

    ASSERT_TRUE(raft::devArrMatch(
      T(expected), sum_act.data(), params.cols, raft::CompareApprox<T>(params.tolerance)));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SumInputs<T> params;
  int rows, cols;
  rmm::device_uvector<T> data, sum_act;
};

const std::vector<SumInputs<float>> inputsf = {
  {0.0001f, 4, 5, true, 1},          {0.0001f, 1024, 32, true, 1},
  {0.0001f, 1024, 256, true, 1},     {0.0001f, 100000000, 1, true, 0.001},
  {0.0001f, 1 << 27, 2, true, 0.1},  {0.0001f, 1, 30, true, 0.001},
  {0.0001f, 1, 1, true, 0.001},      {0.0001f, 17, 5, true, 0.001},
  {0.0001f, 7, 23, true, 0.001},     {0.0001f, 3, 97, true, 0.001},
  {0.0001f, 4, 5, false, 1},         {0.0001f, 1024, 32, false, 1},
  {0.0001f, 1024, 256, false, 1},    {0.0001f, 100000000, 1, false, 0.001},
  {0.0001f, 1 << 27, 2, false, 0.1}, {0.0001f, 1, 30, false, 0.001},
  {0.0001f, 1, 1, false, 0.001},     {0.0001f, 17, 5, false, 0.001},
  {0.0001f, 7, 23, false, 0.001},    {0.0001f, 3, 97, false, 0.001}};

const std::vector<SumInputs<double>> inputsd = {
  {0.000001, 1024, 32, true, 1},    {0.000001, 1024, 256, true, 1},
  {0.000001, 1024, 256, true, 1},   {0.000001, 100000000, 1, true, 0.001},
  {1e-10, 1 << 27, 2, true, 0.1},   {0.000001, 1, 30, true, 0.0001},
  {0.000001, 1, 1, true, 0.0001},   {0.000001, 17, 5, true, 0.0001},
  {0.000001, 7, 23, true, 0.0001},  {0.000001, 3, 97, true, 0.0001},
  {0.000001, 1024, 32, false, 1},   {0.000001, 1024, 256, false, 1},
  {0.000001, 1024, 256, false, 1},  {0.000001, 100000000, 1, false, 0.001},
  {1e-10, 1 << 27, 2, false, 0.1},  {0.000001, 1, 30, false, 0.0001},
  {0.000001, 1, 1, false, 0.0001},  {0.000001, 17, 5, false, 0.0001},
  {0.000001, 7, 23, false, 0.0001}, {0.000001, 3, 97, false, 0.0001}};

typedef SumTest<float> SumTestF;
typedef SumTest<double> SumTestD;

TEST_P(SumTestF, Result) { runTest(); }
TEST_P(SumTestD, Result) { runTest(); }

INSTANTIATE_TEST_CASE_P(SumTests, SumTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(SumTests, SumTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
