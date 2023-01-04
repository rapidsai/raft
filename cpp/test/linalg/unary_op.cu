/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include "unary_op.cuh"
#include <gtest/gtest.h>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename IdxType = int, typename OutType = InType>
void unaryOpLaunch(OutType* out, const InType* in, InType scalar, IdxType len, cudaStream_t stream)
{
  raft::handle_t handle{stream};
  auto out_view = raft::make_device_vector_view(out, len);
  auto in_view  = raft::make_device_vector_view<const InType>(in, len);
  if (in == nullptr) {
    auto op = [scalar] __device__(OutType * ptr, IdxType idx) {
      *ptr = static_cast<OutType>(scalar * idx);
    };

    write_only_unary_op(handle, out_view, op);
  } else {
    auto op = [scalar] __device__(InType in) { return static_cast<OutType>(in * scalar); };
    unary_op(handle, in_view, out_view, op);
  }
}

template <typename InType, typename IdxType, typename OutType = InType>
class UnaryOpTest : public ::testing::TestWithParam<UnaryOpInputs<InType, IdxType, OutType>> {
 public:
  UnaryOpTest()
    : params(::testing::TestWithParam<UnaryOpInputs<InType, IdxType, OutType>>::GetParam()),
      stream(handle.get_stream()),
      in(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    auto len = params.len;
    uniform(handle, r, in.data(), len, InType(-1.0), InType(1.0));
    handle.sync_stream(stream);
  }

  virtual void DoTest()
  {
    auto len    = params.len;
    auto scalar = params.scalar;
    naiveScale(out_ref.data(), in.data(), scalar, len, stream);
    unaryOpLaunch(out.data(), in.data(), scalar, len, stream);
    handle.sync_stream(stream);
    ASSERT_TRUE(devArrMatch(
      out_ref.data(), out.data(), params.len, CompareApprox<OutType>(params.tolerance)));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  UnaryOpInputs<InType, IdxType, OutType> params;
  rmm::device_uvector<InType> in;
  rmm::device_uvector<OutType> out_ref, out;
};

template <typename OutType, typename IdxType>
class WriteOnlyUnaryOpTest : public UnaryOpTest<OutType, IdxType, OutType> {
 protected:
  void DoTest() override
  {
    auto len    = this->params.len;
    auto scalar = this->params.scalar;
    naiveScale(this->out_ref.data(), (OutType*)nullptr, scalar, len, this->stream);
    unaryOpLaunch(this->out.data(), (OutType*)nullptr, scalar, len, this->stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(this->stream));
    ASSERT_TRUE(devArrMatch(this->out_ref.data(),
                            this->out.data(),
                            this->params.len,
                            CompareApprox<OutType>(this->params.tolerance)));
  }
};

#define UNARY_OP_TEST(Name, inputs)  \
  TEST_P(Name, Result) { DoTest(); } \
  INSTANTIATE_TEST_SUITE_P(UnaryOpTests, Name, ::testing::ValuesIn(inputs))

const std::vector<UnaryOpInputs<float, int>> inputsf_i32 = {{0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef UnaryOpTest<float, int> UnaryOpTestF_i32;
UNARY_OP_TEST(UnaryOpTestF_i32, inputsf_i32);
typedef WriteOnlyUnaryOpTest<float, int> WriteOnlyUnaryOpTestF_i32;
UNARY_OP_TEST(WriteOnlyUnaryOpTestF_i32, inputsf_i32);

const std::vector<UnaryOpInputs<float, size_t>> inputsf_i64 = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef UnaryOpTest<float, size_t> UnaryOpTestF_i64;
UNARY_OP_TEST(UnaryOpTestF_i64, inputsf_i64);
typedef WriteOnlyUnaryOpTest<float, size_t> WriteOnlyUnaryOpTestF_i64;
UNARY_OP_TEST(WriteOnlyUnaryOpTestF_i64, inputsf_i64);

const std::vector<UnaryOpInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef UnaryOpTest<float, int, double> UnaryOpTestF_i32_D;
UNARY_OP_TEST(UnaryOpTestF_i32_D, inputsf_i32_d);

const std::vector<UnaryOpInputs<double, int>> inputsd_i32 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
typedef UnaryOpTest<double, int> UnaryOpTestD_i32;
UNARY_OP_TEST(UnaryOpTestD_i32, inputsd_i32);
typedef WriteOnlyUnaryOpTest<double, int> WriteOnlyUnaryOpTestD_i32;
UNARY_OP_TEST(WriteOnlyUnaryOpTestD_i32, inputsd_i32);

const std::vector<UnaryOpInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 2.0, 1234ULL}};
typedef UnaryOpTest<double, size_t> UnaryOpTestD_i64;
UNARY_OP_TEST(UnaryOpTestD_i64, inputsd_i64);
typedef WriteOnlyUnaryOpTest<double, size_t> WriteOnlyUnaryOpTestD_i64;
UNARY_OP_TEST(WriteOnlyUnaryOpTestD_i64, inputsd_i64);

}  // end namespace linalg
}  // end namespace raft
