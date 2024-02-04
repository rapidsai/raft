/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/sqrt.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename Type>
RAFT_KERNEL naiveSqrtElemKernel(Type* out, const Type* in1, int len)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) { out[idx] = raft::sqrt(in1[idx]); }
}

template <typename Type>
void naiveSqrtElem(Type* out, const Type* in1, int len)
{
  static const int TPB = 64;
  int nblks            = raft::ceildiv(len, TPB);
  naiveSqrtElemKernel<Type><<<nblks, TPB>>>(out, in1, len);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename T>
struct SqrtInputs {
  T tolerance;
  int len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SqrtInputs<T>& dims)
{
  return os;
}

template <typename T>
class SqrtTest : public ::testing::TestWithParam<SqrtInputs<T>> {
 protected:
  SqrtTest()
    : in1(0, resource::get_cuda_stream(handle)),
      out_ref(0, resource::get_cuda_stream(handle)),
      out(0, resource::get_cuda_stream(handle))
  {
  }

  void SetUp() override
  {
    auto stream = resource::get_cuda_stream(handle);
    params      = ::testing::TestWithParam<SqrtInputs<T>>::GetParam();
    raft::random::RngState r(params.seed);
    int len = params.len;
    in1.resize(len, stream);
    out_ref.resize(len, stream);
    out.resize(len, stream);
    uniform(handle, r, in1.data(), len, T(1.0), T(2.0));

    naiveSqrtElem(out_ref.data(), in1.data(), len);
    auto out_view = raft::make_device_vector_view(out.data(), len);
    auto in_view  = raft::make_device_vector_view<const T>(in1.data(), len);
    auto in2_view = raft::make_device_vector_view(in1.data(), len);

    sqrt(handle, in_view, out_view);
    sqrt(handle, in_view, in2_view);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::resources handle;
  SqrtInputs<T> params;
  rmm::device_uvector<T> in1, out_ref, out;
  int device_count = 0;
};

const std::vector<SqrtInputs<float>> inputsf2 = {{0.000001f, 1024 * 1024, 1234ULL}};

const std::vector<SqrtInputs<double>> inputsd2 = {{0.00000001, 1024 * 1024, 1234ULL}};

typedef SqrtTest<float> SqrtTestF;
TEST_P(SqrtTestF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<float>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<float>(params.tolerance)));
}

typedef SqrtTest<double> SqrtTestD;
TEST_P(SqrtTestD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), out.data(), params.len, raft::CompareApprox<double>(params.tolerance)));

  ASSERT_TRUE(raft::devArrMatch(
    out_ref.data(), in1.data(), params.len, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(SqrtTests, SqrtTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(SqrtTests, SqrtTestD, ::testing::ValuesIn(inputsd2));

}  // namespace linalg
}  // namespace raft
