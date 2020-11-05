/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
#include <raft/cuda_utils.cuh>
#include <raft/linalg/coalesced_reduction.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"
#include "../fixture.hpp"
#include "reduce.cuh"

namespace raft {
namespace linalg {

template <typename T>
struct coalesced_reduction_inputs {
  T tolerance;
  int rows, cols;
  uint64_t seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os,
                           const coalesced_reduction_inputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void coalesced_reduction_launch(T *dots, const T *data, int cols, int rows,
                              cudaStream_t stream, bool inplace = false) {
  coalescedReduction(dots, data, cols, rows, static_cast<T>(0), stream, inplace,
                     [] __device__(T in, int i) { return in * in; });
}

template <typename T>
class coalesced_reduction_test : public fixture<coalesced_reduction_inputs<T>> {
 protected:
  void initialize() override {
    params_ = ::testing::TestWithParam<coalesced_reduction_inputs<T>>::GetParam();
    raft::random::Rng r(params_.seed);
    auto rows = params_.rows, cols = params_.cols;
    auto len = rows * cols;
    auto stream = this->handle().get_stream();
    raft::allocate(data_, len);
    raft::allocate(dots_exp_, rows);
    raft::allocate(dots_act_, rows);
    r.uniform(data_, len, static_cast<T>(-1.0), static_cast<T>(1.0), stream);
    naive_coalesced_reduction(dots_exp_, data_, cols, rows, stream);

    // Perform reduction with default inplace = false first
    coalesced_reduction_launch(dots_act_, data_, cols, rows, stream);
    // Add to result with inplace = true next
    coalesced_reduction_launch(dots_act_, data_, cols, rows, stream, true);
  }

  void finalize() override {
    CUDA_CHECK(cudaFree(data_));
    CUDA_CHECK(cudaFree(dots_exp_));
    CUDA_CHECK(cudaFree(dots_act_));
  }

  void check() override {
    ASSERT_TRUE(raft::devArrMatch(
                  dots_exp_, dots_act_, params_.rows,
                  raft::compare_approx<T>(params_.tolerance)));
  }

 protected:
  coalesced_reduction_inputs<T> params_;
  T *data_, *dots_exp_, *dots_act_;
};

const std::vector<coalesced_reduction_inputs<float>> kInputsF = {
  {0.000002f, 1024, 32, 1234ULL},
  {0.000002f, 1024, 64, 1234ULL},
  {0.000002f, 1024, 128, 1234ULL},
  {0.000002f, 1024, 256, 1234ULL}};
RUN_TEST(coalesced_reduction, coalesced_reduction_test_f,
         coalesced_reduction_test<float>, kInputsF);

const std::vector<coalesced_reduction_inputs<double>> kInputsD = {
  {0.000000001, 1024, 32, 1234ULL},
  {0.000000001, 1024, 64, 1234ULL},
  {0.000000001, 1024, 128, 1234ULL},
  {0.000000001, 1024, 256, 1234ULL}};
RUN_TEST(coalesced_reduction, coalesced_reduction_test_d,
         coalesced_reduction_test<double>, kInputsD);

}  // end namespace linalg
}  // end namespace raft
