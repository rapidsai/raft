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
#include <raft/linalg/multiply.cuh>
#include <raft/random/rng.cuh>
#include "../fixture.hpp"
#include "../test_utils.h"
#include "unary_op.cuh"

namespace raft {
namespace linalg {

template <typename T>
class multiply_test : public raft::fixture<unary_op_inputs<T>> {
 protected:
  void initialize() override {
    params_ = ::testing::TestWithParam<unary_op_inputs<T>>::GetParam();
    raft::random::Rng r(params_.seed);
    int len = params_.len;
    auto stream = this->handle().get_stream();
    raft::allocate(in_, len);
    raft::allocate(out_ref_, len);
    raft::allocate(out_, len);
    constexpr auto kOne = static_cast<T>(1.0);
    r.uniform(in_, len, -kOne, kOne, stream);
    naive_scale(out_ref_, in_, params_.scalar, len, stream);
    multiplyScalar(out_, in_, params_.scalar, len, stream);
  }

  void finalize() override {
    CUDA_CHECK(cudaFree(in_));
    CUDA_CHECK(cudaFree(out_ref_));
    CUDA_CHECK(cudaFree(out_));
  }

  unary_op_inputs<T> params_;
  T *in_, *out_ref_, *out_;
};

const std::vector<unary_op_inputs<float>> kInputsF = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
RUN_TEST(multiply_test, multiply_test_f, multiply_test<float>, kInputsF);

const std::vector<unary_op_inputs<double>> kInputsD = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
RUN_TEST(multiply_test, multiply_test_d, multiply_test<double>, kInputsD);

}  // end namespace linalg
}  // end namespace raft
