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

#include <cub/cub.cuh>
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <sys/timeb.h>
#include <vector>

#include "../test_utils.cuh"

#include <raft/core/resources.hpp>
#include <raft/random/rmat_rectangular_generator.cuh>
#include <raft/random/rng.cuh>

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace random {

// Courtesy: cuGraph unit-tests

struct RmatInputs {
  size_t r_scale;
  size_t c_scale;
  size_t n_edges;
  bool theta_array;
  uint64_t seed;
  float eps;
};

template <typename OutT, typename InT>
RAFT_KERNEL normalize_kernel(
  OutT* theta, const InT* in_vals, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = threadIdx.x;
  if (idx < max_scale) {
    auto a   = OutT(in_vals[4 * idx]);
    auto b   = OutT(in_vals[4 * idx + 1]);
    auto c   = OutT(in_vals[4 * idx + 2]);
    auto d   = OutT(in_vals[4 * idx + 3]);
    auto sum = a + b + c + d;
    a /= sum;
    b /= sum;
    c /= sum;
    d /= sum;
    theta[4 * idx]     = a;
    theta[4 * idx + 1] = b;
    theta[4 * idx + 2] = c;
    theta[4 * idx + 3] = d;
  }
}

// handle rectangular cases correctly
template <typename OutT>
RAFT_KERNEL handle_rect_kernel(OutT* theta, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = threadIdx.x;
  if (idx < max_scale) {
    auto a = theta[4 * idx];
    auto b = theta[4 * idx + 1];
    auto c = theta[4 * idx + 2];
    auto d = theta[4 * idx + 3];
    if (idx >= r_scale) {
      a += c;
      c = OutT(0);
      b += d;
      d = OutT(0);
    }
    if (idx >= c_scale) {
      a += b;
      b = OutT(0);
      c += d;
      d = OutT(0);
    }
    theta[4 * idx]     = a;
    theta[4 * idx + 1] = b;
    theta[4 * idx + 2] = c;
    theta[4 * idx + 3] = d;
  }
}

// for a single probability distribution across depths, just replicate the theta's!
// this will keep the test code simpler
template <typename OutT>
RAFT_KERNEL theta_kernel(OutT* theta, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = threadIdx.x;
  if (idx != 0 && idx < max_scale) {
    auto a = theta[0];
    auto b = theta[1];
    auto c = theta[2];
    auto d = theta[3];
    if (idx >= r_scale) {
      a += c;
      c = OutT(0);
      b += d;
      d = OutT(0);
    }
    if (idx >= c_scale) {
      a += b;
      b = OutT(0);
      c += d;
      d = OutT(0);
    }
    theta[4 * idx]     = a;
    theta[4 * idx + 1] = b;
    theta[4 * idx + 2] = c;
    theta[4 * idx + 3] = d;
  }
}

template <typename OutT, typename InT>
void normalize(OutT* theta,
               const InT* in_vals,
               size_t max_scale,
               size_t r_scale,
               size_t c_scale,
               bool handle_rect,
               bool theta_array,
               cudaStream_t stream)
{
  // one threadblock with 256 threads is more than enough as the 'scale' parameters
  // won't be that large!
  normalize_kernel<OutT, InT><<<1, 256, 0, stream>>>(theta, in_vals, max_scale, r_scale, c_scale);
  RAFT_CUDA_TRY(cudaGetLastError());
  if (handle_rect) {
    handle_rect_kernel<<<1, 256, 0, stream>>>(theta, max_scale, r_scale, c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
  if (!theta_array) {
    theta_kernel<<<1, 256, 0, stream>>>(theta, max_scale, r_scale, c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
  }
}

RAFT_KERNEL compute_hist(
  int* hist, const size_t* out, size_t len, size_t max_scale, size_t r_scale, size_t c_scale)
{
  size_t idx = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  if (idx + 1 < len) {
    auto src = out[idx], dst = out[idx + 1];
    for (size_t j = 0; j < max_scale; ++j) {
      bool src_bit = j < r_scale ? src & (1 << (r_scale - j - 1)) : 0;
      bool dst_bit = j < c_scale ? dst & (1 << (c_scale - j - 1)) : 0;
      auto idx     = j * 4 + src_bit * 2 + dst_bit;
      atomicAdd(hist + idx, 1);
    }
  }
}

class RmatGenTest : public ::testing::TestWithParam<RmatInputs> {
 public:
  RmatGenTest()
    : handle{},
      stream{resource::get_cuda_stream(handle)},
      params{::testing::TestWithParam<RmatInputs>::GetParam()},
      out{params.n_edges * 2, stream},
      out_src{params.n_edges, stream},
      out_dst{params.n_edges, stream},
      theta{0, stream},
      h_theta{},
      state{params.seed, GeneratorType::GenPC},
      max_scale{std::max(params.r_scale, params.c_scale)}
  {
    theta.resize(4 * max_scale, stream);
    uniform<float>(handle, state, theta.data(), theta.size(), 0.0f, 1.0f);
    normalize<float, float>(theta.data(),
                            theta.data(),
                            max_scale,
                            params.r_scale,
                            params.c_scale,
                            params.r_scale != params.c_scale,
                            params.theta_array,
                            stream);
    h_theta.resize(theta.size());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    raft::update_host(h_theta.data(), theta.data(), theta.size(), stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  void SetUp() override
  {
    if (params.theta_array) {
      rmat_rectangular_gen(out.data(),
                           out_src.data(),
                           out_dst.data(),
                           theta.data(),
                           params.r_scale,
                           params.c_scale,
                           params.n_edges,
                           stream,
                           state);
    } else {
      rmat_rectangular_gen(out.data(),
                           out_src.data(),
                           out_dst.data(),
                           h_theta[0],
                           h_theta[1],
                           h_theta[2],
                           params.r_scale,
                           params.c_scale,
                           params.n_edges,
                           stream,
                           state);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void validate()
  {
    rmm::device_uvector<int> hist{theta.size(), stream};
    RAFT_CUDA_TRY(cudaMemsetAsync(hist.data(), 0, hist.size() * sizeof(int), stream));
    compute_hist<<<raft::ceildiv<size_t>(out.size() / 2, 256), 256, 0, stream>>>(
      hist.data(), out.data(), out.size(), max_scale, params.r_scale, params.c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
    rmm::device_uvector<float> computed_theta{theta.size(), stream};
    normalize<float, int>(computed_theta.data(),
                          hist.data(),
                          max_scale,
                          params.r_scale,
                          params.c_scale,
                          false,
                          true,
                          stream);
    RAFT_CUDA_TRY(cudaGetLastError());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_TRUE(devArrMatchHost(
      h_theta.data(), computed_theta.data(), theta.size(), CompareApprox<float>(params.eps)));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RmatInputs params;
  rmm::device_uvector<size_t> out, out_src, out_dst;
  rmm::device_uvector<float> theta;
  std::vector<float> h_theta;
  RngState state;
  size_t max_scale;
};

class RmatGenMdspanTest : public ::testing::TestWithParam<RmatInputs> {
 public:
  RmatGenMdspanTest()
    : handle{},
      stream{resource::get_cuda_stream(handle)},
      params{::testing::TestWithParam<RmatInputs>::GetParam()},
      out{params.n_edges * 2, stream},
      out_src{params.n_edges, stream},
      out_dst{params.n_edges, stream},
      theta{0, stream},
      h_theta{},
      state{params.seed, GeneratorType::GenPC},
      max_scale{std::max(params.r_scale, params.c_scale)}
  {
    theta.resize(4 * max_scale, stream);
    uniform<float>(handle, state, theta.data(), theta.size(), 0.0f, 1.0f);
    normalize<float, float>(theta.data(),
                            theta.data(),
                            max_scale,
                            params.r_scale,
                            params.c_scale,
                            params.r_scale != params.c_scale,
                            params.theta_array,
                            stream);
    h_theta.resize(theta.size());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    raft::update_host(h_theta.data(), theta.data(), theta.size(), stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  void SetUp() override
  {
    using index_type = size_t;

    using out_view_type = raft::device_mdspan<index_type,
                                              raft::extents<index_type, raft::dynamic_extent, 2>,
                                              raft::row_major>;
    out_view_type out_view(out.data(), out.size());

    using out_src_view_type = raft::device_vector_view<index_type, index_type>;
    out_src_view_type out_src_view(out_src.data(), out_src.size());

    using out_dst_view_type = raft::device_vector_view<index_type, index_type>;
    out_dst_view_type out_dst_view(out_dst.data(), out_dst.size());

    if (params.theta_array) {
      raft::device_vector_view<const float, index_type> theta_view(theta.data(), theta.size());
      rmat_rectangular_gen(handle,
                           state,
                           theta_view,
                           out_view,
                           out_src_view,
                           out_dst_view,
                           params.r_scale,
                           params.c_scale);
    } else {
      rmat_rectangular_gen(handle,
                           state,
                           out_view,
                           out_src_view,
                           out_dst_view,
                           h_theta[0],
                           h_theta[1],
                           h_theta[2],
                           params.r_scale,
                           params.c_scale);
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

  void validate()
  {
    rmm::device_uvector<int> hist{theta.size(), stream};
    RAFT_CUDA_TRY(cudaMemsetAsync(hist.data(), 0, hist.size() * sizeof(int), stream));
    compute_hist<<<raft::ceildiv<size_t>(out.size() / 2, 256), 256, 0, stream>>>(
      hist.data(), out.data(), out.size(), max_scale, params.r_scale, params.c_scale);
    RAFT_CUDA_TRY(cudaGetLastError());
    rmm::device_uvector<float> computed_theta{theta.size(), stream};
    normalize<float, int>(computed_theta.data(),
                          hist.data(),
                          max_scale,
                          params.r_scale,
                          params.c_scale,
                          false,
                          true,
                          stream);
    RAFT_CUDA_TRY(cudaGetLastError());
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_TRUE(devArrMatchHost(
      h_theta.data(), computed_theta.data(), theta.size(), CompareApprox<float>(params.eps)));
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RmatInputs params;
  rmm::device_uvector<size_t> out, out_src, out_dst;
  rmm::device_uvector<float> theta;
  std::vector<float> h_theta;
  RngState state;
  size_t max_scale;
};

static const float TOLERANCE = 0.01f;

const std::vector<RmatInputs> inputs = {
  // square adjacency
  {16, 16, 100000, false, 123456ULL, TOLERANCE},
  {16, 16, 100000, true, 123456ULL, TOLERANCE},
  {16, 16, 200000, false, 123456ULL, TOLERANCE},
  {16, 16, 200000, true, 123456ULL, TOLERANCE},
  {18, 18, 100000, false, 123456ULL, TOLERANCE},
  {18, 18, 100000, true, 123456ULL, TOLERANCE},
  {18, 18, 200000, false, 123456ULL, TOLERANCE},
  {18, 18, 200000, true, 123456ULL, TOLERANCE},
  {16, 16, 100000, false, 456789ULL, TOLERANCE},
  {16, 16, 100000, true, 456789ULL, TOLERANCE},
  {16, 16, 200000, false, 456789ULL, TOLERANCE},
  {16, 16, 200000, true, 456789ULL, TOLERANCE},
  {18, 18, 100000, false, 456789ULL, TOLERANCE},
  {18, 18, 100000, true, 456789ULL, TOLERANCE},
  {18, 18, 200000, false, 456789ULL, TOLERANCE},
  {18, 18, 200000, true, 456789ULL, TOLERANCE},

  // rectangular adjacency
  {16, 18, 200000, false, 123456ULL, TOLERANCE},
  {16, 18, 200000, true, 123456ULL, TOLERANCE},
  {18, 16, 200000, false, 123456ULL, TOLERANCE},
  {18, 16, 200000, true, 123456ULL, TOLERANCE},
  {16, 18, 200000, false, 456789ULL, TOLERANCE},
  {16, 18, 200000, true, 456789ULL, TOLERANCE},
  {18, 16, 200000, false, 456789ULL, TOLERANCE},
  {18, 16, 200000, true, 456789ULL, TOLERANCE}};

TEST_P(RmatGenTest, Result) { validate(); }
INSTANTIATE_TEST_SUITE_P(RmatGenTests, RmatGenTest, ::testing::ValuesIn(inputs));

TEST_P(RmatGenMdspanTest, Result) { validate(); }
INSTANTIATE_TEST_SUITE_P(RmatGenMdspanTests, RmatGenMdspanTest, ::testing::ValuesIn(inputs));

}  // namespace random
}  // namespace raft
