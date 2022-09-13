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

#include "../test_utils.h"
#include "unary_op.cuh"
#include <gtest/gtest.h>
#include <raft/linalg/multiply.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename T>
class MultiplyTest : public ::testing::TestWithParam<UnaryOpInputs<T>> {
 public:
  MultiplyTest()
    : params(::testing::TestWithParam<UnaryOpInputs<T>>::GetParam()),
      stream(handle.get_stream()),
      in(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    params = ::testing::TestWithParam<UnaryOpInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int len = params.len;
    uniform(handle, r, in.data(), len, T(-1.0), T(1.0));
    naiveScale(out_ref.data(), in.data(), params.scalar, len, stream);
    multiplyScalar(out.data(), in.data(), params.scalar, len, stream);
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  UnaryOpInputs<T> params;
  rmm::device_uvector<T> in, out_ref, out;
};

const std::vector<UnaryOpInputs<float>> inputsf = {{0.000001f, 1024 * 1024, 2.f, 1234ULL}};
typedef MultiplyTest<float> MultiplyTestF;
TEST_P(MultiplyTestF, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MultiplyTests, MultiplyTestF, ::testing::ValuesIn(inputsf));

typedef MultiplyTest<double> MultiplyTestD;
const std::vector<UnaryOpInputs<double>> inputsd = {{0.000001f, 1024 * 1024, 2.f, 1234ULL}};
TEST_P(MultiplyTestD, Result)
{
  ASSERT_TRUE(devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MultiplyTests, MultiplyTestD, ::testing::ValuesIn(inputsd));

}  // end namespace linalg
}  // end namespace raft
