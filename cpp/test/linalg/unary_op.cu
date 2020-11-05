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
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"
#include "unary_op.cuh"
#include "../fixture.hpp"

namespace raft {
namespace linalg {

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename IdxType = int, typename OutType = InType>
void unary_op_launch(OutType *out, const InType *in, InType scalar, IdxType len,
                   cudaStream_t stream) {
  if (in == nullptr) {
    auto op = [scalar] __device__(OutType * ptr, IdxType idx) {
      *ptr = static_cast<OutType>(scalar * idx);
    };
    writeOnlyUnaryOp<OutType, decltype(op), IdxType>(out, len, op, stream);
  } else {
    auto op = [scalar] __device__(InType in) {
      return static_cast<OutType>(in * scalar);
    };
    unaryOp<InType, decltype(op), IdxType, OutType>(out, in, len, op, stream);
  }
}

template <typename InType, typename IdxType, typename OutType = InType>
class unary_op_test : public raft::fixture<unary_op_inputs<InType, IdxType, OutType>> {
 protected:
  void initialize() override {
    params_ = ::testing::TestWithParam<
      unary_op_inputs<InType, IdxType, OutType>>::GetParam();
    raft::random::Rng r(params_.seed);
    auto stream = this->handle().get_stream();
    auto len = params_.len;
    allocate(in_, len);
    allocate(out_ref_, len);
    allocate(out_, len);
    constexpr auto kOne = static_cast<InType>(1.0);
    r.uniform(in_, len, -kOne, kOne, stream);
  }

  void finalize() override {
    CUDA_CHECK(cudaFree(in_));
    CUDA_CHECK(cudaFree(out_ref_));
    CUDA_CHECK(cudaFree(out_));
  }

  void check() override {
    auto len = params_.len;
    auto scalar = params_.scalar;
    auto stream = this->handle().get_stream();
    naive_scale(out_ref_, in_, scalar, len, stream);
    unary_op_launch(out_, in_, scalar, len, stream);
    ASSERT_TRUE(devArrMatch(out_ref_, out_, params_.len,
                            compare_approx<OutType>(params_.tolerance)));
  }

  unary_op_inputs<InType, IdxType, OutType> params_;
  InType *in_;
  OutType *out_ref_, *out_;
};

template <typename OutType, typename IdxType>
class write_only_unary_op_test : public unary_op_test<OutType, IdxType, OutType> {
 protected:
  void check() override {
    auto len = this->params_.len;
    auto scalar = this->params_.scalar;
    auto stream = this->handle().get_stream();
    naive_scale<OutType, IdxType, OutType>(this->out_ref_, nullptr, scalar, len, stream);
    unary_op_launch<OutType, IdxType, OutType>(this->out_, nullptr, scalar, len, stream);
    ASSERT_TRUE(devArrMatch(this->out_ref_, this->out_, this->params_.len,
                            compare_approx<OutType>(this->params_.tolerance)));
  }
};

const std::vector<unary_op_inputs<float, int>> kInputsFI32 = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
using unary_op_test_f_i32 = unary_op_test<float, int>;
RUN_TEST_BASE(unary_op, unary_op_test_f_i32, kInputsFI32);
using write_only_unary_op_test_f_i32 = write_only_unary_op_test<float, int>;
RUN_TEST_BASE(unary_op, write_only_unary_op_test_f_i32, kInputsFI32);

const std::vector<unary_op_inputs<float, size_t>> kInputsFI64 = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
using unary_op_test_f_i64 = unary_op_test<float, size_t>;
RUN_TEST_BASE(unary_op, unary_op_test_f_i64, kInputsFI64);
using write_only_unary_op_test_f_i64 = write_only_unary_op_test<float, size_t>;
RUN_TEST_BASE(unary_op, write_only_unary_op_test_f_i64, kInputsFI64);

const std::vector<unary_op_inputs<float, int, double>> kInputsFI32D = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
using unary_op_test_f_i32_d = unary_op_test<float, int, double>;
RUN_TEST_BASE(unary_op, unary_op_test_f_i32_d, kInputsFI32D);

const std::vector<unary_op_inputs<double, int>> kInputsDI32 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
using unary_op_test_d_i32 = unary_op_test<double, int>;
RUN_TEST_BASE(unary_op, unary_op_test_d_i32, kInputsDI32);
using write_only_unary_op_test_d_i32 = write_only_unary_op_test<double, int>;
RUN_TEST_BASE(unary_op, write_only_unary_op_test_d_i32, kInputsDI32);

const std::vector<unary_op_inputs<double, size_t>> kInputsDI64 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
using unary_op_test_d_i64 = unary_op_test<double, size_t>;
RUN_TEST_BASE(unary_op, unary_op_test_d_i64, kInputsDI64);
using write_only_unary_op_test_d_i64 = write_only_unary_op_test<double, size_t>;
RUN_TEST_BASE(unary_op, write_only_unary_op_test_d_i64, kInputsDI64);

}  // end namespace linalg
}  // end namespace raft
