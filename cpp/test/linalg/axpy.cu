/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <raft/linalg/axpy.cuh>

#include "../test_utils.h"
#include "unary_op.cuh"
#include <gtest/gtest.h>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace linalg {

// Reference axpy implementation.
template <typename T>
__global__ void naiveAxpy(const int n, const T alpha, const T* x, T* y)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < n) { y[idx] += alpha * x[idx]; }
}

template <typename T>
class AxpyTest : public ::testing::TestWithParam<UnaryOpInputs<T>> {
 protected:
  UnaryOpInputs<T> params;
  rmm::device_uvector<T> refy;
  rmm::device_uvector<T> y;

 public:
  AxpyTest()
    : testing::TestWithParam<UnaryOpInputs<T>>(),
      refy(0, rmm::cuda_stream_default),
      y(0, rmm::cuda_stream_default)
  {
    rmm::cuda_stream_default.synchronize();
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<UnaryOpInputs<T>>::GetParam();

    raft::handle_t handle;
    cudaStream_t stream = handle.get_stream();

    raft::random::RngState r(params.seed);

    rmm::device_uvector<T> x(params.len, stream);
    y.resize(params.len, stream);

    uniform(handle, r, x.data(), params.len, T(-1.0), T(1.0));
    uniform(handle, r, y.data(), params.len, T(-1.0), T(1.0));
    refy = rmm::device_uvector<T>(y, stream);

    refy        = rmm::device_uvector<T>(y, stream);
    int threads = 64;
    int blocks  = raft::ceildiv<int>(params.len, threads);
    naiveAxpy<T><<<blocks, threads, 0, stream>>>(params.len, params.scalar, x.data(), refy.data());

    axpy(handle, params.len, &params.scalar, x.data(), 1.0, y.data(), 1.0, stream);

    handle.sync_stream();
  }

  void TearDown() override {}
};

const std::vector<UnaryOpInputs<float>> inputsf  = {{0.000001f, 1024 * 1024, 2.f, 1234ULL}};
const std::vector<UnaryOpInputs<double>> inputsd = {{0.000001f, 1024 * 1024, 2.f, 1234ULL}};

typedef AxpyTest<float> AxpyTestF;
TEST_P(AxpyTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refy.data(), y.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}

typedef AxpyTest<double> AxpyTestD;
TEST_P(AxpyTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refy.data(), y.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(AxpyTests, AxpyTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(AxpyTests, AxpyTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
