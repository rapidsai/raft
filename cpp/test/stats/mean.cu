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
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>

namespace raft {
namespace stats {

template <typename T>
struct MeanInputs {
  T tolerance, mean;
  int rows, cols;
  bool sample, rowMajor;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MeanInputs<T>& dims)
{
  return os;
}

template <typename T>
class MeanTest : public ::testing::TestWithParam<MeanInputs<T>> {
 public:
  MeanTest()
    : params(::testing::TestWithParam<MeanInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rows(params.rows),
      cols(params.cols),
      data(rows * cols, stream),
      mean_act(cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int len = rows * cols;
    normal(handle, r, data.data(), len, params.mean, (T)1.0);
    meanSGtest(data.data(), stream);
  }

  void meanSGtest(T* data, cudaStream_t stream)
  {
    int rows = params.rows, cols = params.cols;
    if (params.rowMajor) {
      using layout = raft::row_major;
      mean(handle,
           raft::make_device_matrix_view<const T, int, layout>(data, rows, cols),
           raft::make_device_vector_view<T, int>(mean_act.data(), cols),
           params.sample);
    } else {
      using layout = raft::col_major;
      mean(handle,
           raft::make_device_matrix_view<const T, int, layout>(data, rows, cols),
           raft::make_device_vector_view<T, int>(mean_act.data(), cols),
           params.sample);
    }
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MeanInputs<T> params;
  int rows, cols;
  rmm::device_uvector<T> data, mean_act;
};

// Note: For 1024 samples, 256 experiments, a mean of 1.0 with stddev=1.0, the
// measured mean (of a normal distribution) will fall outside of an epsilon of
// 0.15 only 4/10000 times. (epsilon of 0.1 will fail 30/100 times)
const std::vector<MeanInputs<float>> inputsf = {
  {0.15f, 1.f, 1024, 32, true, false, 1234ULL},    {0.15f, 1.f, 1024, 64, true, false, 1234ULL},
  {0.15f, 1.f, 1024, 128, true, false, 1234ULL},   {0.15f, 1.f, 1024, 256, true, false, 1234ULL},
  {0.15f, -1.f, 1024, 32, false, false, 1234ULL},  {0.15f, -1.f, 1024, 64, false, false, 1234ULL},
  {0.15f, -1.f, 1024, 128, false, false, 1234ULL}, {0.15f, -1.f, 1024, 256, false, false, 1234ULL},
  {0.15f, 1.f, 1024, 32, true, true, 1234ULL},     {0.15f, 1.f, 1024, 64, true, true, 1234ULL},
  {0.15f, 1.f, 1024, 128, true, true, 1234ULL},    {0.15f, 1.f, 1024, 256, true, true, 1234ULL},
  {0.15f, -1.f, 1024, 32, false, true, 1234ULL},   {0.15f, -1.f, 1024, 64, false, true, 1234ULL},
  {0.15f, -1.f, 1024, 128, false, true, 1234ULL},  {0.15f, -1.f, 1024, 256, false, true, 1234ULL},
  {0.15f, -1.f, 1030, 1, false, false, 1234ULL},   {0.15f, -1.f, 1030, 60, true, false, 1234ULL},
  {2.0f, -1.f, 31, 120, false, false, 1234ULL},    {2.0f, -1.f, 1, 130, true, false, 1234ULL},
  {0.15f, -1.f, 1030, 1, false, true, 1234ULL},    {0.15f, -1.f, 1030, 60, true, true, 1234ULL},
  {2.0f, -1.f, 31, 120, false, true, 1234ULL},     {2.0f, -1.f, 1, 130, false, true, 1234ULL},
  {2.0f, -1.f, 1, 1, false, false, 1234ULL},       {2.0f, -1.f, 1, 1, false, true, 1234ULL},
  {2.0f, -1.f, 7, 23, false, false, 1234ULL},      {2.0f, -1.f, 7, 23, false, true, 1234ULL},
  {2.0f, -1.f, 17, 5, false, false, 1234ULL},      {2.0f, -1.f, 17, 5, false, true, 1234ULL}};

const std::vector<MeanInputs<double>> inputsd = {
  {0.15, 1.0, 1024, 32, true, false, 1234ULL},    {0.15, 1.0, 1024, 64, true, false, 1234ULL},
  {0.15, 1.0, 1024, 128, true, false, 1234ULL},   {0.15, 1.0, 1024, 256, true, false, 1234ULL},
  {0.15, -1.0, 1024, 32, false, false, 1234ULL},  {0.15, -1.0, 1024, 64, false, false, 1234ULL},
  {0.15, -1.0, 1024, 128, false, false, 1234ULL}, {0.15, -1.0, 1024, 256, false, false, 1234ULL},
  {0.15, 1.0, 1024, 32, true, true, 1234ULL},     {0.15, 1.0, 1024, 64, true, true, 1234ULL},
  {0.15, 1.0, 1024, 128, true, true, 1234ULL},    {0.15, 1.0, 1024, 256, true, true, 1234ULL},
  {0.15, -1.0, 1024, 32, false, true, 1234ULL},   {0.15, -1.0, 1024, 64, false, true, 1234ULL},
  {0.15, -1.0, 1024, 128, false, true, 1234ULL},  {0.15, -1.0, 1024, 256, false, true, 1234ULL},
  {0.15, -1.0, 1030, 1, false, false, 1234ULL},   {0.15, -1.0, 1030, 60, true, false, 1234ULL},
  {2.0, -1.0, 31, 120, false, false, 1234ULL},    {2.0, -1.0, 1, 130, true, false, 1234ULL},
  {0.15, -1.0, 1030, 1, false, true, 1234ULL},    {0.15, -1.0, 1030, 60, true, true, 1234ULL},
  {2.0, -1.0, 31, 120, false, true, 1234ULL},     {2.0, -1.0, 1, 130, false, true, 1234ULL},
  {2.0, -1.0, 1, 1, false, false, 1234ULL},       {2.0, -1.0, 1, 1, false, true, 1234ULL},
  {2.0, -1.0, 7, 23, false, false, 1234ULL},      {2.0, -1.0, 7, 23, false, true, 1234ULL},
  {2.0, -1.0, 17, 5, false, false, 1234ULL},      {2.0, -1.0, 17, 5, false, true, 1234ULL}};

typedef MeanTest<float> MeanTestF;
TEST_P(MeanTestF, Result)
{
  ASSERT_TRUE(
    devArrMatch(params.mean, mean_act.data(), params.cols, CompareApprox<float>(params.tolerance)));
}

typedef MeanTest<double> MeanTestD;
TEST_P(MeanTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    params.mean, mean_act.data(), params.cols, CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MeanTests, MeanTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
