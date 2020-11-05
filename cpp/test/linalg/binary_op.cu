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
#include <raft/linalg/binary_op.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"
#include "../fixture.hpp"
#include "binary_op.cuh"

namespace raft {
namespace linalg {

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename IdxType, typename OutType>
void binary_op_launch(OutType *out, const InType *in1, const InType *in2,
                      IdxType len, cudaStream_t stream) {
  binaryOp(
    out, in1, in2, len, [] __device__(InType a, InType b) { return a + b; },
    stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
class binary_op_test
  : public raft::fixture<binary_op_inputs<InType, IdxType, OutType>> {
 protected:
  void initialize() override {
    params_ = ::testing::TestWithParam<
      binary_op_inputs<InType, IdxType, OutType>>::GetParam();
    raft::random::Rng r(params_.seed);
    auto stream = this->handle().get_stream();
    IdxType len = params_.len;
    allocate(in1_, len);
    allocate(in2_, len);
    allocate(out_ref_, len);
    allocate(out_, len);
    constexpr auto kOne = static_cast<InType>(1.0);
    r.uniform(in1_, len, -kOne, kOne, stream);
    r.uniform(in2_, len, -kOne, kOne, stream);
    naive_add(out_ref_, in1_, in2_, len);
    binary_op_launch(out_, in1_, in2_, len, stream);
  }

  void finalize() override {
    CUDA_CHECK(cudaFree(in1_));
    CUDA_CHECK(cudaFree(in2_));
    CUDA_CHECK(cudaFree(out_ref_));
    CUDA_CHECK(cudaFree(out_));
  }

  void check() override {
    ASSERT_TRUE(devArrMatch(out_ref_, out_, params_.len,
                            compare_approx<OutType>(params_.tolerance)));
  }

 protected:
  binary_op_inputs<InType, IdxType, OutType> params_;
  InType *in1_, *in2_;
  OutType *out_ref_, *out_;
};

const std::vector<binary_op_inputs<float, int>> kInputsFI32 = {
  {0.000001f, 1024 * 1024, 1234ULL}};
using binary_op_test_f_i32 = binary_op_test<float, int>;
RUN_TEST_BASE(binary_op, binary_op_test_f_i32, kInputsFI32);

const std::vector<binary_op_inputs<float, size_t>> kInputsFI64 = {
  {0.000001f, 1024 * 1024, 1234ULL}};
using binary_op_test_f_i64 = binary_op_test<float, size_t>;
RUN_TEST_BASE(binary_op, binary_op_test_f_i64, kInputsFI64);

const std::vector<binary_op_inputs<float, int, double>> kInputsFI32D = {
  {0.000001f, 1024 * 1024, 1234ULL}};
using binary_op_test_f_i32_d = binary_op_test<float, int, double>;
RUN_TEST_BASE(binary_op, binary_op_test_f_i32_d, kInputsFI32D);

const std::vector<binary_op_inputs<double, int>> kInputsDI32 = {
  {0.00000001, 1024 * 1024, 1234ULL}};
using binary_op_test_d_i32 = binary_op_test<double, int>;
RUN_TEST_BASE(binary_op, binary_op_test_d_i32, kInputsDI32);

const std::vector<binary_op_inputs<double, size_t>> kInputsDI64 = {
  {0.00000001, 1024 * 1024, 1234ULL}};
using binary_op_test_d_i64 = binary_op_test<double, size_t>;
RUN_TEST_BASE(binary_op, binary_op_test_d_i64, kInputsDI64);

}  // namespace linalg
}  // namespace raft
