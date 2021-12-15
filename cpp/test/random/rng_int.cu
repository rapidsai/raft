/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>
#include <raft/cudart_utils.h>
#include <cub/cub.cuh>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.hpp>
#include "../test_utils.h"

namespace raft {
namespace random {

using namespace raft::random;

enum RandomType { RNG_Uniform };

template <typename T, int TPB>
__global__ void meanKernel(float* out, const T* data, int len)
{
  typedef cub::BlockReduce<float, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  int tid   = threadIdx.x + blockIdx.x * blockDim.x;
  float val = tid < len ? data[tid] : T(0);
  float x   = BlockReduce(temp_storage).Sum(val);
  __syncthreads();
  float xx = BlockReduce(temp_storage).Sum(val * val);
  __syncthreads();
  if (threadIdx.x == 0) {
    raft::myAtomicAdd(out, x);
    raft::myAtomicAdd(out + 1, xx);
  }
}

template <typename T>
struct RngInputs {
  float tolerance;
  int len;
  // start, end: for uniform
  // mean, sigma: for normal/lognormal
  // mean, beta: for gumbel
  // mean, scale: for logistic and laplace
  // lambda: for exponential
  // sigma: for rayleigh
  T start, end;
  RandomType type;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RngInputs<T>& dims)
{
  return os;
}

template <typename T>
class RngTest : public ::testing::TestWithParam<RngInputs<T>> {
 public:
  RngTest()
    : params(::testing::TestWithParam<RngInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      data(0, stream),
      stats(2, stream)
  {
    data.resize(params.len, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(stats.data(), 0, 2 * sizeof(float), stream));
  }

 protected:
  void SetUp() override
  {
    Rng r(params.seed, params.gtype);

    switch (params.type) {
      case RNG_Uniform:
        r.uniformInt(data.data(), params.len, params.start, params.end, stream);
        break;
    };
    static const int threads = 128;
    meanKernel<T, threads><<<raft::ceildiv(params.len, threads), threads, 0, stream>>>(
      stats.data(), data.data(), params.len);
    update_host<float>(h_stats, stats.data(), 2, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    h_stats[0] /= params.len;
    h_stats[1] = (h_stats[1] / params.len) - (h_stats[0] * h_stats[0]);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void getExpectedMeanVar(float meanvar[2])
  {
    switch (params.type) {
      case RNG_Uniform:
        meanvar[0] = (params.start + params.end) * 0.5f;
        meanvar[1] = params.end - params.start;
        meanvar[1] = meanvar[1] * meanvar[1] / 12.f;
        break;
    };
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  RngInputs<T> params;
  rmm::device_uvector<T> data;
  rmm::device_uvector<float> stats;
  float h_stats[2];  // mean, var
};

typedef RngTest<uint32_t> RngTestU32;
const std::vector<RngInputs<uint32_t>> inputs_u32 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};
TEST_P(RngTestU32, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestU32, ::testing::ValuesIn(inputs_u32));

typedef RngTest<uint64_t> RngTestU64;
const std::vector<RngInputs<uint64_t>> inputs_u64 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};
TEST_P(RngTestU64, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestU64, ::testing::ValuesIn(inputs_u64));

typedef RngTest<int32_t> RngTestS32;
const std::vector<RngInputs<int32_t>> inputs_s32 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};
TEST_P(RngTestS32, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestS32, ::testing::ValuesIn(inputs_s32));

typedef RngTest<int64_t> RngTestS64;
const std::vector<RngInputs<int64_t>> inputs_s64 = {
  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPhilox, 1234ULL},

  {0.1f, 32 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL},
  {0.1f, 8 * 1024, 0, 20, RNG_Uniform, GenPC, 1234ULL}};
TEST_P(RngTestS64, Result)
{
  float meanvar[2];
  getExpectedMeanVar(meanvar);
  ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
  ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngTestS64, ::testing::ValuesIn(inputs_s64));

}  // namespace random
}  // namespace raft
