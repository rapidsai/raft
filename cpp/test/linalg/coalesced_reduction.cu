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
#include "reduce.cuh"

namespace raft {
namespace linalg {

template <typename T>
struct coalescedReductionInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os,
                           const coalescedReductionInputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void coalescedReductionLaunch(T *dots, const T *data, int cols, int rows,
                              cudaStream_t stream, bool inplace = false) {
  coalescedReduction(dots, data, cols, rows, (T)0, stream, inplace,
                     [] __device__(T in, int i) { return in * in; });
}

template <typename T>
class coalescedReductionTest
  : public ::testing::TestWithParam<coalescedReductionInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<coalescedReductionInputs<T>>::GetParam();
    raft::random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    CUDA_CHECK(cudaStreamCreate(&stream));
    raft::allocate(data, len, stream);
    raft::allocate(dots_exp, rows, stream);
    raft::allocate(dots_act, rows, stream);
    r.uniform(data, len, T(-1.0), T(1.0), stream);
    naiveCoalescedReduction(dots_exp, data, cols, rows, stream);

    // Perform reduction with default inplace = false first
    coalescedReductionLaunch(dots_act, data, cols, rows, stream);
    // Add to result with inplace = true next
    coalescedReductionLaunch(dots_act, data, cols, rows, stream, true);

    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void TearDown() override {
    raft::deallocate_all(stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  coalescedReductionInputs<T> params;
  T *data, *dots_exp, *dots_act;
  cudaStream_t stream;
};

const std::vector<coalescedReductionInputs<float>> inputsf = {
  {0.000002f, 1024, 32, 1234ULL},
  {0.000002f, 1024, 64, 1234ULL},
  {0.000002f, 1024, 128, 1234ULL},
  {0.000002f, 1024, 256, 1234ULL}};

const std::vector<coalescedReductionInputs<double>> inputsd = {
  {0.000000001, 1024, 32, 1234ULL},
  {0.000000001, 1024, 64, 1234ULL},
  {0.000000001, 1024, 128, 1234ULL},
  {0.000000001, 1024, 256, 1234ULL}};

typedef coalescedReductionTest<float> coalescedReductionTestF;
TEST_P(coalescedReductionTestF, Result) {
  ASSERT_TRUE(raft::devArrMatch(dots_exp, dots_act, params.rows,
                                raft::CompareApprox<float>(params.tolerance),
                                stream));
}

typedef coalescedReductionTest<double> coalescedReductionTestD;
TEST_P(coalescedReductionTestD, Result) {
  ASSERT_TRUE(raft::devArrMatch(dots_exp, dots_act, params.rows,
                                raft::CompareApprox<double>(params.tolerance),
                                stream));
}

INSTANTIATE_TEST_CASE_P(coalescedReductionTests, coalescedReductionTestF,
                        ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(coalescedReductionTests, coalescedReductionTestD,
                        ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
