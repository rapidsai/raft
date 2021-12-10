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
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>
#include "../test_utils.h"
#include "binary_op.cuh"

namespace raft {
namespace linalg {

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename IdxType, typename OutType>
void binaryOpLaunch(
  OutType* out, const InType* in1, const InType* in2, IdxType len, cudaStream_t stream)
{
  binaryOp(
    out, in1, in2, len, [] __device__(InType a, InType b) { return a + b; }, stream);
}

template <typename InType, typename IdxType, typename OutType = InType>
class BinaryOpTest : public ::testing::TestWithParam<BinaryOpInputs<InType, IdxType, OutType>> {
 public:
  BinaryOpTest()
    : params(::testing::TestWithParam<BinaryOpInputs<InType, IdxType, OutType>>::GetParam()),
      stream(handle.get_stream()),
      in1(params.len, stream),
      in2(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::Rng r(params.seed);
    IdxType len = params.len;
    r.uniform(in1.data(), len, InType(-1.0), InType(1.0), stream);
    r.uniform(in2.data(), len, InType(-1.0), InType(1.0), stream);
    naiveAdd(out_ref.data(), in1.data(), in2.data(), len);
    binaryOpLaunch(out.data(), in1.data(), in2.data(), len, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  BinaryOpInputs<InType, IdxType, OutType> params;
  rmm::device_uvector<InType> in1;
  rmm::device_uvector<InType> in2;
  rmm::device_uvector<OutType> out_ref;
  rmm::device_uvector<OutType> out;
};

const std::vector<BinaryOpInputs<float, int>> inputsf_i32 = {{0.000001f, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<float, int> BinaryOpTestF_i32;
TEST_P(BinaryOpTestF_i32, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(BinaryOpTests, BinaryOpTestF_i32, ::testing::ValuesIn(inputsf_i32));

const std::vector<BinaryOpInputs<float, size_t>> inputsf_i64 = {{0.000001f, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<float, size_t> BinaryOpTestF_i64;
TEST_P(BinaryOpTestF_i64, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(BinaryOpTests, BinaryOpTestF_i64, ::testing::ValuesIn(inputsf_i64));

const std::vector<BinaryOpInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<float, int, double> BinaryOpTestF_i32_D;
TEST_P(BinaryOpTestF_i32_D, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(BinaryOpTests, BinaryOpTestF_i32_D, ::testing::ValuesIn(inputsf_i32_d));

const std::vector<BinaryOpInputs<double, int>> inputsd_i32 = {{0.00000001, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<double, int> BinaryOpTestD_i32;
TEST_P(BinaryOpTestD_i32, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(BinaryOpTests, BinaryOpTestD_i32, ::testing::ValuesIn(inputsd_i32));

const std::vector<BinaryOpInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 1234ULL}};
typedef BinaryOpTest<double, size_t> BinaryOpTestD_i64;
TEST_P(BinaryOpTestD_i64, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(BinaryOpTests, BinaryOpTestD_i64, ::testing::ValuesIn(inputsd_i64));

template <typename math_t>
class BinaryOpAlignment : public ::testing::Test {
 protected:
  BinaryOpAlignment()
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    handle.set_stream(stream);
  }
  void TearDown() override { RAFT_CUDA_TRY(cudaStreamDestroy(stream)); }

 public:
  void Misaligned()
  {
    // Test to trigger cudaErrorMisalignedAddress if veclen is incorrectly
    // chosen.
    int n = 1024;
    rmm::device_uvector<math_t> x(n, stream);
    rmm::device_uvector<math_t> y(n, stream);
    rmm::device_uvector<math_t> z(n, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(x.data(), 0, n * sizeof(math_t), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(y.data(), 0, n * sizeof(math_t), stream));
    raft::linalg::binaryOp(
      z.data() + 9,
      x.data() + 137,
      y.data() + 19,
      256,
      [] __device__(math_t x, math_t y) { return x + y; },
      stream);
  }

  raft::handle_t handle;
  cudaStream_t stream;
};
typedef ::testing::Types<float, double> FloatTypes;
TYPED_TEST_CASE(BinaryOpAlignment, FloatTypes);
TYPED_TEST(BinaryOpAlignment, Misaligned) { this->Misaligned(); }
}  // namespace linalg
}  // namespace raft
