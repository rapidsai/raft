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
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <vector>

namespace raft {
namespace random {

/* In this test we generate pseudo-random values following a probability distribution defined by the
 * given weights. If the probability of i is is p=w_i/sum(w), the expected value for the normalized
 * histogram is E=p, the standard deviation sigma=sqrt(p*(1-p)/n).
 * We use as the test tolerance eps=4*sigma(p,n) where p=min(w_i/sum(w)) and n=sampledLen.
 */

template <typename WeightT, typename IdxT>
struct RngDiscreteInputs {
  float tolerance;
  IdxT sampledLen;
  std::vector<WeightT> weights;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename WeightT, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const RngDiscreteInputs<WeightT, IdxT>& d)
{
  using raft::operator<<;
  return os << "{" << d.sampledLen << ", " << d.weights << "}";
}

// Computes the intensity histogram from a sequence of labels
template <typename LabelT, typename IdxT>
void compute_normalized_histogram(
  const LabelT* labels, float* histogram, IdxT sampledLen, IdxT len, const cudaStream_t& stream)
{
  IdxT num_levels  = len + 1;
  IdxT lower_level = 0;
  IdxT upper_level = len;

  rmm::device_uvector<IdxT> count(len, stream);

  size_t temp_storage_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(nullptr,
                                                    temp_storage_bytes,
                                                    labels,
                                                    count.data(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    sampledLen,
                                                    stream));

  rmm::device_uvector<char> workspace(temp_storage_bytes, stream);

  RAFT_CUDA_TRY(cub::DeviceHistogram::HistogramEven(workspace.data(),
                                                    temp_storage_bytes,
                                                    labels,
                                                    count.data(),
                                                    num_levels,
                                                    lower_level,
                                                    upper_level,
                                                    sampledLen,
                                                    stream));

  float scale = static_cast<float>(sampledLen);
  raft::linalg::unaryOp(
    histogram,
    count.data(),
    len,
    [scale] __device__(const IdxT& cnt) { return static_cast<float>(cnt) / scale; },
    stream);
}

template <typename OutT, typename WeightT, typename IdxT>
class RngDiscreteTest : public ::testing::TestWithParam<RngDiscreteInputs<WeightT, IdxT>> {
 public:
  RngDiscreteTest()
    : params(::testing::TestWithParam<RngDiscreteInputs<WeightT, IdxT>>::GetParam()),
      stream(handle.get_stream()),
      out(params.sampledLen, stream),
      weights(params.weights.size(), stream),
      histogram(params.weights.size(), stream),
      exp_histogram(params.weights.size())
  {
  }

 protected:
  void SetUp() override
  {
    IdxT len = params.weights.size();

    raft::copy(weights.data(), params.weights.data(), len, stream);

    RngState r(params.seed, params.gtype);
    raft::device_vector_view<OutT, IdxT> out_view(out.data(), out.size());
    auto weights_view =
      raft::make_device_vector_view<const WeightT, IdxT>(weights.data(), weights.size());

    discrete(handle, r, out_view, weights_view);

    // Compute the actual and expected normalized histogram of the values
    float total_weight = 0.0f;
    for (IdxT i = 0; i < len; i++) {
      total_weight += params.weights[i];
    }
    for (IdxT i = 0; i < len; i++) {
      exp_histogram[i] = params.weights[i] / total_weight;
    }
    compute_normalized_histogram(out.data(), histogram.data(), params.sampledLen, len, stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  RngDiscreteInputs<WeightT, IdxT> params;
  rmm::device_uvector<OutT> out;
  rmm::device_uvector<WeightT> weights;
  rmm::device_uvector<float> histogram;
  std::vector<float> exp_histogram;
};

const std::vector<RngDiscreteInputs<float, int>> inputs_u32 = {
  {0.016f, 10000, {1.f, 2.f, 3.f, 4.f}, GenPhilox, 1234ULL},
  {0.01f, 10000, {0.5f, 0.3f, 0.3f, 0.f, 0.f, 0.f, 1.5f, 2.0f}, GenPhilox, 1234ULL},

  {0.016f, 10000, {1.f, 2.f, 3.f, 4.f}, GenPC, 1234ULL},
};

using RngDiscreteTestU32F = RngDiscreteTest<uint32_t, float, int>;
TEST_P(RngDiscreteTestU32F, Result)
{
  ASSERT_TRUE(devArrMatchHost(exp_histogram.data(),
                              histogram.data(),
                              exp_histogram.size(),
                              CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(RngTests, RngDiscreteTestU32F, ::testing::ValuesIn(inputs_u32));

}  // namespace random
}  // namespace raft
