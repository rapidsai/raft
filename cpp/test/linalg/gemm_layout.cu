/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <raft/cuda_utils.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"
#include "../fixture.hpp"

namespace raft {
namespace linalg {

template <typename T>
struct gemm_layout_inputs {
  int m;
  int n;
  int k;
  bool z_layout;
  bool x_layout;
  bool y_layout;
  T tolerance;
  uint64_t seed;
};

// Reference GEMM implementation.
template <typename T>
__global__ void naive_gemm(T *Z, T *X, T *Y, int M, int N, int K,
                           bool isZColMajor, bool isXColMajor,
                           bool isYColMajor) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;

  for (int m = tidy; m < M; m += (blockDim.y * gridDim.y)) {
    for (int n = tidx; n < N; n += (blockDim.x * gridDim.x)) {
      T temp = T(0.0);
      for (int k = 0; k < K; k++) {
        auto x_index = isXColMajor ? m + k * M : m * K + k;
        auto y_index = isYColMajor ? k + n * K : k * N + n;
        temp += X[x_index] * Y[y_index];
      }
      auto z_index = isZColMajor ? m + n * M : m * N + n;
      Z[z_index] = temp;
    }
  }
}

template <typename T>
class gemm_layout_test : public raft::fixture<gemm_layout_inputs<T>> {
 protected:
  void initialize() override {
    params_ = ::testing::TestWithParam<gemm_layout_inputs<T>>::GetParam();
    auto stream = this->handle().get_stream();
    raft::random::Rng r(params_.seed);
    // We compute Z = X * Y and compare against reference result
    // Dimensions of X : M x K
    // Dimensions of Y : K x N
    // Dimensions of Z : M x N
    size_t x_elems = params_.n * params_.k;
    size_t y_elems = params_.k * params_.n;
    size_t z_elems = params_.m * params_.n;
    allocate(x_, x_elems);
    allocate(y_, y_elems);
    allocate(ref_z_, z_elems);
    allocate(z_, z_elems);
    constexpr auto kTen = static_cast<T>(10.0);
    r.uniform(x_, x_elems, -kTen, kTen, stream);
    r.uniform(y_, y_elems, -kTen, kTen, stream);
    dim3 blocks(raft::ceildiv<int>(params_.m, 128),
                raft::ceildiv<int>(params_.m, 4), 1);
    dim3 threads(128, 4, 1);
    naive_gemm<<<blocks, threads>>>(ref_z_, x_, y_, params_.m, params_.n, params_.k,
                                    params_.z_layout, params_.x_layout,
                                    params_.y_layout);
    gemm(this->handle(), z_, x_, y_, params_.m, params_.n, params_.k, params_.z_layout,
         params_.x_layout, params_.y_layout, stream);
  }

  void finalize() override {
    CUDA_CHECK(cudaFree(ref_z_));
    CUDA_CHECK(cudaFree(z_));
    CUDA_CHECK(cudaFree(x_));
    CUDA_CHECK(cudaFree(y_));
  }

  void check() override {
    ASSERT_TRUE(raft::devArrMatch(ref_z_, z_, params_.m * params_.n,
                                  raft::compare_approx<T>(params_.tolerance)));
  }

 protected:
  gemm_layout_inputs<T> params_;
  T *ref_z_;  // Reference result for comparison
  T *z_;      // Computed result
  T *x_, *y_;
};

const std::vector<gemm_layout_inputs<float>> kInputsF = {
  {80, 70, 80, true, true, true, 1e-4f, 76433ULL},
  {80, 100, 40, true, true, false, 1e-4f, 426646ULL},
  {20, 100, 20, true, false, true, 1e-4f, 237703ULL},
  {100, 60, 30, true, false, false, 1e-4f, 538004ULL},
  {50, 10, 60, false, true, true, 1e-4f, 73012ULL},
  {90, 90, 30, false, true, false, 1e-4f, 538147ULL},
  {30, 100, 10, false, false, true, 1e-4f, 412352ULL},
  {40, 80, 100, false, false, false, 1e-4f, 297941ULL}};
RUN_TEST(gemm_layout, gemm_layout_test_f, gemm_layout_test<float>, kInputsF);

const std::vector<gemm_layout_inputs<double>> kInputsD = {
  {10, 70, 40, true, true, true, 1e-6, 535648ULL},
  {30, 30, 30, true, true, false, 1e-6, 956681ULL},
  {70, 80, 50, true, false, true, 1e-6, 875083ULL},
  {80, 90, 70, true, false, false, 1e-6, 50744ULL},
  {90, 90, 30, false, true, true, 1e-6, 506321ULL},
  {40, 100, 70, false, true, false, 1e-6, 638418ULL},
  {80, 50, 30, false, false, true, 1e-6, 701529ULL},
  {50, 80, 60, false, false, false, 1e-6, 893038ULL}};
RUN_TEST(gemm_layout, gemm_layout_test_d, gemm_layout_test<double>, kInputsD);

}  // end namespace linalg
}  // end namespace raft
