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

#include "../test_utils.h"
#include <gtest/gtest.h>
#include <raft/core/cudart_utils.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/random/rng.cuh>
#include <set>
#include <vector>

namespace raft {
namespace random {

using namespace raft::random;

// Terminology:
// SWoR - Sample Without Replacement

template <typename T>
struct SWoRInputs {
  int len, sampledLen;
  int largeWeightIndex;
  T largeWeight;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SWoRInputs<T>& dims)
{
  return os;
}

template <typename T>
class SWoRTest : public ::testing::TestWithParam<SWoRInputs<T>> {
 public:
  SWoRTest()
    : params(::testing::TestWithParam<SWoRInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      in(params.len, stream),
      wts(params.len, stream),
      out(params.sampledLen, stream),
      outIdx(params.sampledLen, stream)
  {
  }

 protected:
  void SetUp() override
  {
    RngState r(params.seed, params.gtype);
    h_outIdx.resize(params.sampledLen);
    uniform(handle, r, in.data(), params.len, T(-1.0), T(1.0));
    uniform(handle, r, wts.data(), params.len, T(1.0), T(2.0));
    if (params.largeWeightIndex >= 0) {
      update_device(wts.data() + params.largeWeightIndex, &params.largeWeight, 1, stream);
    }
    sampleWithoutReplacement(
      handle, r, out.data(), outIdx.data(), in.data(), wts.data(), params.sampledLen, params.len);
    update_host(&(h_outIdx[0]), outIdx.data(), params.sampledLen, stream);
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  SWoRInputs<T> params;
  rmm::device_uvector<T> in, out, wts;
  rmm::device_uvector<int> outIdx;
  std::vector<int> h_outIdx;
};

typedef SWoRTest<float> SWoRTestF;
const std::vector<SWoRInputs<float>> inputsf = {{1024, 512, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 1024, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 1024 - 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 1024 - 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 512, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 1024, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 1024 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 1, 1024 - 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 512, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 1024, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 1024 + 1, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024 + 2, 1024 + 2, -1, 0.f, GenPhilox, 1234ULL},
                                                {1024, 512, 10, 100000.f, GenPhilox, 1234ULL},

                                                {1024, 512, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 1024, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 512 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 1024 - 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 512 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 1024 - 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 512, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 1024, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 512 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 1024 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 512 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 1, 1024 - 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 512, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 1024, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 512 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 1024 + 1, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 512 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024 + 2, 1024 + 2, -1, 0.f, GenPC, 1234ULL},
                                                {1024, 512, 10, 100000.f, GenPC, 1234ULL}};

TEST_P(SWoRTestF, Result)
{
  std::set<int> occurence;
  for (int i = 0; i < params.sampledLen; ++i) {
    auto val = h_outIdx[i];
    // indices must be in the given range
    ASSERT_TRUE(0 <= val && val < params.len)
      << "out-of-range index @i=" << i << " val=" << val << " sampledLen=" << params.sampledLen;
    // indices should not repeat
    ASSERT_TRUE(occurence.find(val) == occurence.end())
      << "repeated index @i=" << i << " idx=" << val;
    occurence.insert(val);
  }
  // if there's a skewed distribution, the top index should correspond to the
  // particular item with a large weight
  if (params.largeWeightIndex >= 0) { ASSERT_EQ(h_outIdx[0], params.largeWeightIndex); }
}
INSTANTIATE_TEST_SUITE_P(SWoRTests, SWoRTestF, ::testing::ValuesIn(inputsf));

typedef SWoRTest<double> SWoRTestD;
const std::vector<SWoRInputs<double>> inputsd = {{1024, 512, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 1024, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 1024 - 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 1024 - 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 512, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 1024, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 1024 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 1, 1024 - 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 512, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 1024, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 1024 + 1, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024 + 2, 1024 + 2, -1, 0.0, GenPhilox, 1234ULL},
                                                 {1024, 512, 10, 100000.0, GenPhilox, 1234ULL},

                                                 {1024, 512, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 1024, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 512 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 1024 - 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 512 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 1024 - 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 512, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 1024, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 512 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 1024 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 512 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 1, 1024 - 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 512, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 1024, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 512 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 1024 + 1, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 512 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024 + 2, 1024 + 2, -1, 0.0, GenPC, 1234ULL},
                                                 {1024, 512, 10, 100000.0, GenPC, 1234ULL}};

TEST_P(SWoRTestD, Result)
{
  std::set<int> occurence;
  for (int i = 0; i < params.sampledLen; ++i) {
    auto val = h_outIdx[i];
    // indices must be in the given range
    ASSERT_TRUE(0 <= val && val < params.len)
      << "out-of-range index @i=" << i << " val=" << val << " sampledLen=" << params.sampledLen;
    // indices should not repeat
    ASSERT_TRUE(occurence.find(val) == occurence.end())
      << "repeated index @i=" << i << " idx=" << val;
    occurence.insert(val);
  }
  // if there's a skewed distribution, the top index should correspond to the
  // particular item with a large weight
  if (params.largeWeightIndex >= 0) { ASSERT_EQ(h_outIdx[0], params.largeWeightIndex); }
}
INSTANTIATE_TEST_SUITE_P(SWoRTests, SWoRTestD, ::testing::ValuesIn(inputsd));

}  // namespace random
}  // namespace raft
