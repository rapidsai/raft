/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/map.cuh>
#include <raft/random/rng.cuh>
#include "../test_utils.h"

namespace raft {
namespace linalg {

template <typename InType, typename IdxType, typename OutType>
void mapLaunch(OutType *out, const InType *in1, const InType *in2,
               const InType *in3, InType scalar, IdxType len,
               cudaStream_t stream) {
  map(
    out, len,
    [=] __device__(InType a, InType b, InType c) { return a + b + c + scalar; },
    stream, in1, in2, in3);
}

template <typename InType, typename IdxType = int, typename OutType = InType>
struct MapInputs {
  InType tolerance;
  IdxType len;
  unsigned long long int seed;
  InType scalar;
};

template <typename InType, typename IdxType, typename OutType = InType>
void create_ref(OutType *out_ref, const InType *in1, const InType *in2,
                const InType *in3, InType scalar, IdxType len,
                cudaStream_t stream) {
  InType *tmp;
  raft::allocate(tmp, len, stream);
  eltwiseAdd(tmp, in1, in2, len, stream);
  eltwiseAdd(out_ref, tmp, in3, len, stream);
  scalarAdd(out_ref, out_ref, (OutType)scalar, len, stream);
  CUDA_CHECK(cudaFree(tmp));
}

template <typename InType, typename IdxType, typename OutType = InType>
class MapTest
  : public ::testing::TestWithParam<MapInputs<InType, IdxType, OutType>> {
 protected:
  void SetUp() override {
    params =
      ::testing::TestWithParam<MapInputs<InType, IdxType, OutType>>::GetParam();
    raft::random::Rng r(params.seed);

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    IdxType len = params.len;
    raft::allocate(in1, len, stream);
    raft::allocate(in2, len, stream);
    raft::allocate(in3, len, stream);
    raft::allocate(out_ref, len, stream);
    raft::allocate(out, len, stream);
    r.uniform(in1, len, InType(-1.0), InType(1.0), stream);
    r.uniform(in2, len, InType(-1.0), InType(1.0), stream);
    r.uniform(in3, len, InType(-1.0), InType(1.0), stream);

    create_ref(out_ref, in1, in2, in3, params.scalar, len, stream);
    mapLaunch(out, in1, in2, in3, params.scalar, len, stream);
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in1));
    CUDA_CHECK(cudaFree(in2));
    CUDA_CHECK(cudaFree(in3));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

 protected:
  MapInputs<InType, IdxType, OutType> params;
  InType *in1, *in2, *in3;
  OutType *out_ref, *out;
};

const std::vector<MapInputs<float, int>> inputsf_i32 = {
  {0.000001f, 1024 * 1024, 1234ULL, 3.2}};
typedef MapTest<float, int> MapTestF_i32;
TEST_P(MapTestF_i32, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestF_i32,
                         ::testing::ValuesIn(inputsf_i32));

const std::vector<MapInputs<float, size_t>> inputsf_i64 = {
  {0.000001f, 1024 * 1024, 1234ULL, 9.4}};
typedef MapTest<float, size_t> MapTestF_i64;
TEST_P(MapTestF_i64, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestF_i64,
                         ::testing::ValuesIn(inputsf_i64));

const std::vector<MapInputs<float, int, double>> inputsf_i32_d = {
  {0.000001f, 1024 * 1024, 1234ULL, 5.9}};
typedef MapTest<float, int, double> MapTestF_i32_D;
TEST_P(MapTestF_i32_D, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestF_i32_D,
                         ::testing::ValuesIn(inputsf_i32_d));

const std::vector<MapInputs<double, int>> inputsd_i32 = {
  {0.00000001, 1024 * 1024, 1234ULL, 7.5}};
typedef MapTest<double, int> MapTestD_i32;
TEST_P(MapTestD_i32, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestD_i32,
                         ::testing::ValuesIn(inputsd_i32));

const std::vector<MapInputs<double, size_t>> inputsd_i64 = {
  {0.00000001, 1024 * 1024, 1234ULL, 5.2}};
typedef MapTest<double, size_t> MapTestD_i64;
TEST_P(MapTestD_i64, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MapTests, MapTestD_i64,
                         ::testing::ValuesIn(inputsd_i64));

}  // namespace linalg
}  // namespace raft
