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
#include <raft/matrix/math.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>

namespace raft {
namespace stats {

template <typename T>
struct StdDevInputs {
  T tolerance, mean, stddev;
  int rows, cols;
  bool sample, rowMajor;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const StdDevInputs<T>& dims)
{
  return os;
}

template <typename T>
class StdDevTest : public ::testing::TestWithParam<StdDevInputs<T>> {
 public:
  StdDevTest()
    : params(::testing::TestWithParam<StdDevInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rows(params.rows),
      cols(params.cols),
      data(rows * cols, stream),
      mean_act(cols, stream),
      stddev_act(cols, stream),
      vars_act(cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    random::RngState r(params.seed);
    int len = rows * cols;

    data.resize(len, stream);
    mean_act.resize(cols, stream);
    stddev_act.resize(cols, stream);
    vars_act.resize(cols, stream);
    normal(handle, r, data.data(), len, params.mean, params.stddev);
    stdVarSGtest(data.data(), stream);
    resource::sync_stream(handle, stream);
  }

  void stdVarSGtest(T* data, cudaStream_t stream)
  {
    int rows = params.rows, cols = params.cols;

    if (params.rowMajor) {
      using layout_t = raft::row_major;
      mean(handle,
           raft::make_device_matrix_view<const T, int, layout_t>(data, rows, cols),
           raft::make_device_vector_view<T, int>(mean_act.data(), cols),
           params.sample);

      stddev(handle,
             raft::make_device_matrix_view<const T, int, layout_t>(data, rows, cols),
             raft::make_device_vector_view<const T, int>(mean_act.data(), cols),
             raft::make_device_vector_view<T, int>(stddev_act.data(), cols),
             params.sample);

      vars(handle,
           raft::make_device_matrix_view<const T, int, layout_t>(data, rows, cols),
           raft::make_device_vector_view<const T, int>(mean_act.data(), cols),
           raft::make_device_vector_view<T, int>(vars_act.data(), cols),
           params.sample);
    } else {
      using layout_t = raft::col_major;
      mean(handle,
           raft::make_device_matrix_view<const T, int, layout_t>(data, rows, cols),
           raft::make_device_vector_view<T>(mean_act.data(), cols),
           params.sample);

      stddev(handle,
             raft::make_device_matrix_view<const T, int, layout_t>(data, rows, cols),
             raft::make_device_vector_view<const T, int>(mean_act.data(), cols),
             raft::make_device_vector_view<T, int>(stddev_act.data(), cols),
             params.sample);

      vars(handle,
           raft::make_device_matrix_view<const T, int, layout_t>(data, rows, cols),
           raft::make_device_vector_view<const T, int>(mean_act.data(), cols),
           raft::make_device_vector_view<T, int>(vars_act.data(), cols),
           params.sample);
    }
    raft::matrix::seqRoot(vars_act.data(), T(1), cols, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  StdDevInputs<T> params;
  int rows, cols;
  rmm::device_uvector<T> data, mean_act, stddev_act, vars_act;
};

const std::vector<StdDevInputs<float>> inputsf = {
  {0.1f, 1.f, 2.f, 1024, 32, true, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 64, true, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 128, true, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 256, true, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 32, false, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 64, false, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 128, false, false, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 256, false, false, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 32, true, true, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 64, true, true, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 128, true, true, 1234ULL},
  {0.1f, 1.f, 2.f, 1024, 256, true, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 32, false, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 64, false, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 128, false, true, 1234ULL},
  {0.1f, -1.f, 2.f, 1024, 256, false, true, 1234ULL}};

const std::vector<StdDevInputs<double>> inputsd = {
  {0.1, 1.0, 2.0, 1024, 32, true, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 64, true, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 128, true, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 256, true, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 32, false, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 64, false, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 128, false, false, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 256, false, false, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 32, true, true, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 64, true, true, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 128, true, true, 1234ULL},
  {0.1, 1.0, 2.0, 1024, 256, true, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 32, false, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 64, false, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 128, false, true, 1234ULL},
  {0.1, -1.0, 2.0, 1024, 256, false, true, 1234ULL}};

typedef StdDevTest<float> StdDevTestF;
TEST_P(StdDevTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    params.stddev, stddev_act.data(), params.cols, CompareApprox<float>(params.tolerance), stream));

  ASSERT_TRUE(devArrMatch(stddev_act.data(),
                          vars_act.data(),
                          params.cols,
                          CompareApprox<float>(params.tolerance),
                          stream));
}

typedef StdDevTest<double> StdDevTestD;
TEST_P(StdDevTestD, Result)
{
  ASSERT_TRUE(devArrMatch(params.stddev,
                          stddev_act.data(),
                          params.cols,
                          CompareApprox<double>(params.tolerance),
                          stream));

  ASSERT_TRUE(devArrMatch(stddev_act.data(),
                          vars_act.data(),
                          params.cols,
                          CompareApprox<double>(params.tolerance),
                          stream));
}

INSTANTIATE_TEST_SUITE_P(StdDevTests, StdDevTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(StdDevTests, StdDevTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
