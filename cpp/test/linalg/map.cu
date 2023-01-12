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
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename InType, typename IdxType, typename OutType>
void mapLaunch(OutType* out,
               const InType* in1,
               const InType* in2,
               const InType* in3,
               InType scalar,
               IdxType len,
               cudaStream_t stream)
{
  raft::device_resources handle{stream};
  auto out_view = raft::make_device_vector_view(out, len);
  auto in1_view = raft::make_device_vector_view(in1, len);
  map(
    handle,
    in1_view,
    out_view,
    [=] __device__(InType a, InType b, InType c) { return a + b + c + scalar; },
    in2,
    in3);
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct MapInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
  InType scalar;
};

template <typename InType, typename IdxType, typename OutType = InType>
void create_ref(OutType* out_ref,
                const InType* in1,
                const InType* in2,
                const InType* in3,
                InType scalar,
                IdxType len,
                cudaStream_t stream)
{
  rmm::device_uvector<InType> tmp(len, stream);
  eltwiseAdd(tmp.data(), in1, in2, len, stream);
  eltwiseAdd(out_ref, tmp.data(), in3, len, stream);
  scalarAdd(out_ref, out_ref, (OutType)scalar, len, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
}

template <typename InType, typename IdxType, typename OutType = InType>
class MapTest : public ::testing::TestWithParam<MapInputs<InType, IdxType, OutType>> {
 public:
  MapTest()
    : params(::testing::TestWithParam<MapInputs<InType, IdxType, OutType>>::GetParam()),
      stream(handle.get_stream()),
      in1(params.len, stream),
      in2(params.len, stream),
      in3(params.len, stream),
      out_ref(params.len, stream),
      out(params.len, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);

    IdxType len = params.len;
    uniform(handle, r, in1.data(), len, InType(-1.0), InType(1.0));
    uniform(handle, r, in2.data(), len, InType(-1.0), InType(1.0));
    uniform(handle, r, in3.data(), len, InType(-1.0), InType(1.0));

    create_ref(out_ref.data(), in1.data(), in2.data(), in3.data(), params.scalar, len, stream);
    mapLaunch(out.data(), in1.data(), in2.data(), in3.data(), params.scalar, len, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  }

 protected:
  raft::device_resources handle;
  cudaStream_t stream;

  MapInputs<InType, IdxType, OutType> params;
  rmm::device_uvector<InType> in1, in2, in3;
  rmm::device_uvector<OutType> out_ref, out;
};

const std::vector<MapInputs<float, int>> inputsf_i32 = {{0.000001f, 1024 * 1024, 1234ULL, 3.2}};
typedef MapTest<float, int> MapTestF_i32;
TEST_P(MapTestF_i32, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestF_i32, ::testing::ValuesIn(inputsf_i32));

const std::vector<MapInputs<float, size_t>> inputsf_i64 = {{0.000001f, 1024 * 1024, 1234ULL, 9.4}};
typedef MapTest<float, size_t> MapTestF_i64;
TEST_P(MapTestF_i64, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestF_i64, ::testing::ValuesIn(inputsf_i64));

const std::vector<MapInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 1234ULL, 5.9}};
typedef MapTest<float, int, double> MapTestF_i32_D;
TEST_P(MapTestF_i32_D, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestF_i32_D, ::testing::ValuesIn(inputsf_i32_d));

const std::vector<MapInputs<double, int>> inputsd_i32 = {{0.00000001, 1024 * 1024, 1234ULL, 7.5}};
typedef MapTest<double, int> MapTestD_i32;
TEST_P(MapTestD_i32, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestD_i32, ::testing::ValuesIn(inputsd_i32));

const std::vector<MapInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 1234ULL, 5.2}};
typedef MapTest<double, size_t> MapTestD_i64;
TEST_P(MapTestD_i64, Result)
{
  ASSERT_TRUE(
    devArrMatch(out_ref.data(), out.data(), params.len, CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestD_i64, ::testing::ValuesIn(inputsd_i64));

}  // namespace linalg
}  // namespace raft
