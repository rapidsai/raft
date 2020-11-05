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
#include <raft/linalg/divide.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"
#include "unary_op.cuh"
#include "../fixture.hpp"

namespace raft {
namespace linalg {

template <typename Type>
__global__ void naive_divide_kernel(Type *out, const Type *in, Type scalar,
                                    int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    out[idx] = in[idx] / scalar;
  }
}

template <typename Type>
void naive_divide(Type *out, const Type *in, Type scalar, int len,
                 cudaStream_t stream) {
  static const int kTpb = 64;
  int nblks = raft::ceildiv(len, kTpb);
  naive_divide_kernel<Type><<<nblks, kTpb, 0, stream>>>(out, in, scalar, len);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
class divide_test : public raft::fixture<raft::linalg::unary_op_inputs<T>> {
 protected:
  void initialize() override {
    params_ =
      ::testing::TestWithParam<raft::linalg::unary_op_inputs<T>>::GetParam();
    raft::random::Rng r(params_.seed);
    int len = params_.len;
    auto stream = this->handle().get_stream();
    raft::allocate(in_, len);
    raft::allocate(out_ref_, len);
    raft::allocate(out_, len);
    constexpr auto kOne = static_cast<T>(1.0);
    r.uniform(in_, len, -kOne, kOne, stream);
    naive_divide(out_ref_, in_, params_.scalar, len, stream);
    divideScalar(out_, in_, params_.scalar, len, stream);
  }

  void finalize() override {
    CUDA_CHECK(cudaFree(in_));
    CUDA_CHECK(cudaFree(out_ref_));
    CUDA_CHECK(cudaFree(out_));
  }

  void check() override {
    ASSERT_TRUE(devArrMatch(out_ref_, out_, params_.len,
                            raft::compare_approx<T>(params_.tolerance)));
  }

 protected:
  unary_op_inputs<T> params_;
  T *in_, *out_ref_, *out_;
};

const std::vector<unary_op_inputs<float>> kInputsF = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
RUN_TEST(divide, divide_test_f, divide_test<float>, kInputsF);

const std::vector<unary_op_inputs<double>> kInputsD = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
RUN_TEST(divide, divide_test_d, divide_test<double>, kInputsD);

}  // end namespace linalg
}  // end namespace raft
