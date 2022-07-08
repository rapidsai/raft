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

#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <sys/timeb.h>
#include <vector>

#include "../test_utils.h"

#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>

namespace raft {
namespace random {

// Courtesy: cuGraph unit-tests
// static constexpr float kTolerance = 0.01f;
// static constexpr size_t kMinEdges = 100000;

struct RmatInputs {
  size_t r_scale;
  size_t c_scale;
  size_t n_edges;
  bool clip_and_flip;
  bool theta_array;
  uint64_t seed;
};

__global__ void normalize_kernel(float* theta, size_t len) {
  size_t idx = threadIdx.x;
  if (idx < len) {
    auto a = theta[4 * idx];
    auto b = theta[4 * idx + 1];
    auto c = theta[4 * idx + 2];
    auto d = theta[4 * idx + 3];
    auto sum = a + b + c + d;
    theta[4 * idx] = a / sum;
    theta[4 * idx + 1] = b / sum;
    theta[4 * idx + 2] = c / sum;
    theta[4 * idx + 3] = d / sum;
  }
}

class RmatGenTest : public ::testing::TestWithParam<RmatInputs> {
 public:
  RmatGenTest()
    : handle{},
      stream{handle.get_stream()},
      params{::testing::TestWithParam<RmatInputs>::GetParam()},
      out{params.n_edges * 2, stream},
      out_src{params.n_edges, stream},
      out_dst{params.n_edges, stream},
      theta{0, stream},
      h_theta{},
      state{params.seed, GeneratorType::GenPC}
  {
    auto theta_len = params.theta_array ? max(params.r_scale, params.c_scale) : 1;
    theta.resize(4 * theta_len, stream);
    uniform<float>(state, theta.data(), 4 * theta_len, 0.0f, 1.0f, stream);
    // one threadblock with 256 threads is more than enough as the 'scale' parameters
    // won't be that large!
    normalize_kernel<<<1, 256, 0, stream>>>(theta.data(), theta_len);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    h_theta.resize(theta.size());
    raft::update_host(h_theta, theta.data(), theta.size(), stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  void SetUp() override
  {
    if (params.theta_array) {
      rmat_rectangular_gen(out.data(), out_src.data(), out_dst.data(), theta.data(), params.r_scale,
			   params.c_scale, params.n_edges, params.clip_and_flip, stream, state);
    } else {
      rmat_rectangular_gen(out.data(), out_src.data(), out_dst.data(), h_theta[0], h_theta[1],
			   h_theta[2], params.r_scale, params.c_scale, params.n_edges,
			   params.clip_and_flip, stream, state);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void validate()
  {
    //@todo!!!
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  RmatInputs params;
  rmm::device_uvector<size_t> out, out_src, out_dst;
  rmm::device_uvector<float> theta;
  std::vector<float> h_theta;
  RngState state;
};

const std::vector<RmatInputs> inputs = {
  // square adjacency
  {16, 16, 100000, false, false, 123456ULL},
  {16, 16, 100000, false, true, 123456ULL},
  {16, 16, 100000, true, false, 123456ULL},
  {16, 16, 100000, true, true, 123456ULL},
  {16, 16, 200000, false, false, 123456ULL},
  {16, 16, 200000, false, true, 123456ULL},
  {16, 16, 200000, true, false, 123456ULL},
  {16, 16, 200000, true, true, 123456ULL},
  {18, 18, 100000, false, false, 123456ULL},
  {18, 18, 100000, false, true, 123456ULL},
  {18, 18, 100000, true, false, 123456ULL},
  {18, 18, 100000, true, true, 123456ULL},
  {18, 18, 200000, false, false, 123456ULL},
  {18, 18, 200000, false, true, 123456ULL},
  {18, 18, 200000, true, false, 123456ULL},
  {18, 18, 200000, true, true, 123456ULL},
  {16, 16, 100000, false, false, 456789ULL},
  {16, 16, 100000, false, true, 456789ULL},
  {16, 16, 100000, true, false, 456789ULL},
  {16, 16, 100000, true, true, 456789ULL},
  {16, 16, 200000, false, false, 456789ULL},
  {16, 16, 200000, false, true, 456789ULL},
  {16, 16, 200000, true, false, 456789ULL},
  {16, 16, 200000, true, true, 456789ULL},
  {18, 18, 100000, false, false, 456789ULL},
  {18, 18, 100000, false, true, 456789ULL},
  {18, 18, 100000, true, false, 456789ULL},
  {18, 18, 100000, true, true, 456789ULL},
  {18, 18, 200000, false, false, 456789ULL},
  {18, 18, 200000, false, true, 456789ULL},
  {18, 18, 200000, true, false, 456789ULL},
  {18, 18, 200000, true, true, 456789ULL},

  // rectangular adjacency
  {16, 18, 200000, false, false, 123456ULL},
  {16, 18, 200000, false, true, 123456ULL},
  {18, 16, 200000, false, false, 123456ULL},
  {18, 16, 200000, false, true, 123456ULL},
  {16, 18, 200000, false, false, 456789ULL},
  {16, 18, 200000, false, true, 456789ULL},
  {18, 16, 200000, false, false, 456789ULL},
  {18, 16, 200000, false, true, 456789ULL}};

TEST_P(RmatGenTest, Result)
{
  validate();
}
INSTANTIATE_TEST_SUITE_P(RmatGenTests, RmatGenTest, ::testing::ValuesIn(inputs));

}  // namespace random
}  // namespace raft
