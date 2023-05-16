/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/math.cuh>
#include <raft/random/rng.cuh>
#include <raft/stats/meanvar.cuh>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>

namespace raft {
namespace stats {

template <typename T>
struct MeanVarInputs {
  T mean, stddev;
  int rows, cols;
  bool sample, rowMajor;
  unsigned long long int seed;
  static const int N_SIGMAS = 6;

  T mean_tol() const { return T(N_SIGMAS) * stddev / sqrt(T(rows)); }

  T var_tol() const
  {
    return T(N_SIGMAS) * stddev * stddev * sqrt(T(2.0) / T(std::max(1, rows - 1)));
  }
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MeanVarInputs<T>& ps)
{
  return os << "rows: " << ps.rows << "; cols: " << ps.cols << "; "
            << (ps.rowMajor ? "row-major" : "col-major") << " (tolerance: mean = " << ps.mean_tol()
            << ", var = " << ps.var_tol() << ")";
}

template <typename T>
class MeanVarTest : public ::testing::TestWithParam<MeanVarInputs<T>> {
 public:
  MeanVarTest()
    : params(::testing::TestWithParam<MeanVarInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.rows * params.cols, stream),
      mean_act(params.cols, stream),
      vars_act(params.cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    random::RngState r(params.seed);
    normal(handle, r, data.data(), params.cols * params.rows, params.mean, params.stddev);

    if (params.rowMajor) {
      using layout = raft::row_major;
      meanvar(
        handle,
        raft::make_device_matrix_view<const T, int, layout>(data.data(), params.rows, params.cols),
        raft::make_device_vector_view<T, int>(mean_act.data(), params.cols),
        raft::make_device_vector_view<T, int>(vars_act.data(), params.cols),
        params.sample);
    } else {
      using layout = raft::col_major;
      meanvar(
        handle,
        raft::make_device_matrix_view<const T, int, layout>(data.data(), params.rows, params.cols),
        raft::make_device_vector_view<T, int>(mean_act.data(), params.cols),
        raft::make_device_vector_view<T, int>(vars_act.data(), params.cols),
        params.sample);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MeanVarInputs<T> params;
  rmm::device_uvector<T> data, mean_act, vars_act;
};

const std::vector<MeanVarInputs<float>> inputsf = {
  {1.f, 2.f, 1024, 32, true, false, 1234ULL},    {1.f, 2.f, 1024, 64, true, false, 1234ULL},
  {1.f, 2.f, 1024, 128, true, false, 1234ULL},   {1.f, 2.f, 1024, 256, true, false, 1234ULL},
  {-1.f, 2.f, 1024, 32, false, false, 1234ULL},  {-1.f, 2.f, 1024, 64, false, false, 1234ULL},
  {-1.f, 2.f, 1024, 128, false, false, 1234ULL}, {-1.f, 2.f, 1024, 256, false, false, 1234ULL},
  {-1.f, 2.f, 1024, 256, false, false, 1234ULL}, {-1.f, 2.f, 1024, 257, false, false, 1234ULL},
  {1.f, 2.f, 1024, 32, true, true, 1234ULL},     {1.f, 2.f, 1024, 64, true, true, 1234ULL},
  {1.f, 2.f, 1024, 128, true, true, 1234ULL},    {1.f, 2.f, 1024, 256, true, true, 1234ULL},
  {-1.f, 2.f, 1024, 32, false, true, 1234ULL},   {-1.f, 2.f, 1024, 64, false, true, 1234ULL},
  {-1.f, 2.f, 1024, 128, false, true, 1234ULL},  {-1.f, 2.f, 1024, 256, false, true, 1234ULL},
  {-1.f, 2.f, 1024, 257, false, true, 1234ULL},  {-1.f, 2.f, 700, 13, false, true, 1234ULL},
  {10.f, 2.f, 500000, 811, false, true, 1234ULL}};

const std::vector<MeanVarInputs<double>> inputsd = {{1.0, 2.0, 1024, 32, true, false, 1234ULL},
                                                    {1.0, 2.0, 1024, 64, true, false, 1234ULL},
                                                    {1.0, 2.0, 1024, 128, true, false, 1234ULL},
                                                    {1.0, 2.0, 1024, 256, true, false, 1234ULL},
                                                    {-1.0, 2.0, 1024, 32, false, false, 1234ULL},
                                                    {-1.0, 2.0, 1024, 64, false, false, 1234ULL},
                                                    {-1.0, 2.0, 1024, 128, false, false, 1234ULL},
                                                    {-1.0, 2.0, 1024, 256, false, false, 1234ULL},
                                                    {1.0, 2.0, 1024, 32, true, true, 1234ULL},
                                                    {1.0, 2.0, 1024, 64, true, true, 1234ULL},
                                                    {1.0, 2.0, 1024, 128, true, true, 1234ULL},
                                                    {1.0, 2.0, 1024, 256, true, true, 1234ULL},
                                                    {-1.0, 2.0, 1024, 32, false, true, 1234ULL},
                                                    {-1.0, 2.0, 1024, 64, false, true, 1234ULL},
                                                    {-1.0, 2.0, 1024, 128, false, true, 1234ULL},
                                                    {-1.0, 2.0, 1024, 256, false, true, 1234ULL}};

typedef MeanVarTest<float> MeanVarTestF;
TEST_P(MeanVarTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    params.mean, mean_act.data(), params.cols, CompareApprox<float>(params.mean_tol()), stream));

  ASSERT_TRUE(devArrMatch(params.stddev * params.stddev,
                          vars_act.data(),
                          params.cols,
                          CompareApproxNoScaling<float>(params.var_tol()),
                          stream));
}

typedef MeanVarTest<double> MeanVarTestD;
TEST_P(MeanVarTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    params.mean, mean_act.data(), params.cols, CompareApprox<double>(params.mean_tol()), stream));

  ASSERT_TRUE(devArrMatch(params.stddev * params.stddev,
                          vars_act.data(),
                          params.cols,
                          CompareApproxNoScaling<double>(params.var_tol()),
                          stream));
}

INSTANTIATE_TEST_SUITE_P(MeanVarTests, MeanVarTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MeanVarTests, MeanVarTestD, ::testing::ValuesIn(inputsd));

}  // end namespace stats
}  // end namespace raft
