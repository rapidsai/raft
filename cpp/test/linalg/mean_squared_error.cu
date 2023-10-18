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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/mean_squared_error.cuh>

#include "../test_utils.cuh"
#include <gtest/gtest.h>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <rmm/device_scalar.hpp>

namespace raft {
namespace linalg {

// reference MSE calculation
template <typename T>
RAFT_KERNEL naiveMeanSquaredError(const int n, const T* a, const T* b, T weight, T* out)
{
  T err = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    T diff = a[i] - b[i];
    err += weight * diff * diff / n;
  }
  atomicAdd(out, err);
}

template <typename T, typename IndexType = std::uint32_t>
struct MeanSquaredErrorInputs {
  T tolerance;
  IndexType len;
  T weight;
  unsigned long long int seed;
};

template <typename T>
class MeanSquaredErrorTest : public ::testing::TestWithParam<MeanSquaredErrorInputs<T>> {
 protected:
  MeanSquaredErrorInputs<T> params;

  raft::resources handle;
  rmm::device_scalar<T> output;
  rmm::device_scalar<T> refoutput;

 public:
  MeanSquaredErrorTest()
    : testing::TestWithParam<MeanSquaredErrorInputs<T>>(),
      output(0, resource::get_cuda_stream(handle)),
      refoutput(0, resource::get_cuda_stream(handle))
  {
    resource::sync_stream(handle);
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<MeanSquaredErrorInputs<T>>::GetParam();

    cudaStream_t stream = resource::get_cuda_stream(handle);

    raft::random::RngState r(params.seed);

    rmm::device_uvector<T> a(params.len, stream);
    rmm::device_uvector<T> b(params.len, stream);
    uniform(handle, r, a.data(), params.len, T(-1.0), T(1.0));
    uniform(handle, r, b.data(), params.len, T(-1.0), T(1.0));
    resource::sync_stream(handle);

    mean_squared_error<T, std::uint32_t, T>(handle,
                                            make_device_vector_view<const T>(a.data(), params.len),
                                            make_device_vector_view<const T>(b.data(), params.len),
                                            make_device_scalar_view<T>(output.data()),
                                            params.weight);

    naiveMeanSquaredError<<<256, 256, 0, stream>>>(
      params.len, a.data(), b.data(), params.weight, refoutput.data());
    resource::sync_stream(handle);
  }

  void TearDown() override {}
};

const std::vector<MeanSquaredErrorInputs<float>> inputsf = {
  {0.0001f, 1024 * 1024, 1.0, 1234ULL},
  {0.0001f, 4 * 1024 * 1024, 8.0, 1234ULL},
  {0.0001f, 16 * 1024 * 1024, 24.0, 1234ULL},
  {0.0001f, 98689, 1.0, 1234ULL},
};

const std::vector<MeanSquaredErrorInputs<double>> inputsd = {
  {0.0001f, 1024 * 1024, 1.0, 1234ULL},
  {0.0001f, 4 * 1024 * 1024, 8.0, 1234ULL},
  {0.0001f, 16 * 1024 * 1024, 24.0, 1234ULL},
  {0.0001f, 98689, 1.0, 1234ULL},
};

typedef MeanSquaredErrorTest<float> MeanSquaredErrorTestF;
TEST_P(MeanSquaredErrorTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refoutput.data(), output.data(), 1, raft::CompareApprox<float>(params.tolerance)));
}

typedef MeanSquaredErrorTest<double> MeanSquaredErrorTestD;
TEST_P(MeanSquaredErrorTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    refoutput.data(), output.data(), 1, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(MeanSquaredErrorTests,
                         MeanSquaredErrorTestF,
                         ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_SUITE_P(MeanSquaredErrorTests,
                         MeanSquaredErrorTestD,
                         ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
