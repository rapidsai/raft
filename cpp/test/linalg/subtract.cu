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
#include <raft/linalg/subtract.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"

namespace raft {
namespace linalg {

template <typename Type>
__global__ void naiveSubtractElemKernel(Type *out, const Type *in1,
                                        const Type *in2, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = in1[idx] - in2[idx];
  }
}

template <typename Type>
void naiveSubtractElem(Type *out, const Type *in1, const Type *in2, int len,
                       cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = raft::ceildiv(len, TPB);
  naiveSubtractElemKernel<Type><<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
__global__ void naiveSubtractScalarKernel(Type *out, const Type *in1,
                                          const Type in2, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = in1[idx] - in2;
  }
}

template <typename Type>
void naiveSubtractScalar(Type *out, const Type *in1, const Type in2, int len,
                         cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = raft::ceildiv(len, TPB);
  naiveSubtractScalarKernel<Type>
    <<<nblks, TPB, 0, stream>>>(out, in1, in2, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct SubtractInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const SubtractInputs<T> &dims) {
  return os;
}

template <typename T>
class SubtractTest : public ::testing::TestWithParam<SubtractInputs<T>> {
 public:
  SubtractTest()
    : params(::testing::TestWithParam<SubtractInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      in1(params.len, stream),
      in2(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream) {}

 protected:
  void SetUp() override {
    raft::random::Rng r(params.seed);
    int len = params.len;
    r.uniform(in1.data(), len, T(-1.0), T(1.0), stream);
    r.uniform(in2.data(), len, T(-1.0), T(1.0), stream);

    naiveSubtractElem(out_ref.data(), in1.data(), in2.data(), len, stream);
    naiveSubtractScalar(out_ref.data(), out_ref.data(), T(1), len, stream);

    subtract(out.data(), in1.data(), in2.data(), len, stream);
    subtractScalar(out.data(), out.data(), T(1), len, stream);
    subtract(in1.data(), in1.data(), in2.data(), len, stream);
    subtractScalar(in1.data(), in1.data(), T(1), len, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

 protected:
  SubtractInputs<T> params;
  rmm::device_uvector<T> in1, in2, out_ref, out;
  raft::handle_t handle;
  cudaStream_t stream;
};

const std::vector<SubtractInputs<float>> inputsf2 = {
  {0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<SubtractInputs<double>> inputsd2 = {
  {0.00000001, 1024 * 1024, 1234ULL}};

typedef SubtractTest<float> SubtractTestF;
TEST_P(SubtractTestF, Result) {
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(), out.data(), params.len,
                                raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(out_ref.data(), in1.data(), params.len,
                                raft::CompareApprox<float>(params.tolerance)));
}

typedef SubtractTest<double> SubtractTestD;
TEST_P(SubtractTestD, Result) {
  ASSERT_TRUE(raft::devArrMatch(out_ref.data(), out.data(), params.len,
                                raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(out_ref.data(), in1.data(), params.len,
                                raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(SubtractTests, SubtractTestF,
                         ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(SubtractTests, SubtractTestD,
                         ::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
