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
#include "distance_base.cuh"

namespace raft {
namespace distance {

template <typename DataType>
class DistanceEucExpTest : public DistanceTest<raft::distance::DistanceType::L2Expanded, DataType> {
};

template <typename DataType>
class DistanceEucExpTestXequalY
  : public DistanceTestSameBuffer<raft::distance::DistanceType::L2Expanded, DataType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 2048, 4096, 128, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.003f, 1021, 1021, 1021, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
  {0.003f, 1021, 1021, 1021, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsXeqYf = {
  {0.01f, 2048, 4096, 128, true, 1234ULL},
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.03f, 1021, 1021, 1021, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
  {0.03f, 1021, 1021, 1021, false, 1234ULL},
};

typedef DistanceEucExpTest<float> DistanceEucExpTestF;
TEST_P(DistanceEucExpTestF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestF, ::testing::ValuesIn(inputsf));

typedef DistanceEucExpTestXequalY<float> DistanceEucExpTestXequalYF;
TEST_P(DistanceEucExpTestXequalYF, Result)
{
  int m = params.m;
  ASSERT_TRUE(raft::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                m,
                                raft::CompareApprox<float>(params.tolerance),
                                stream));
  ASSERT_TRUE(raft::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m / 2,
                                m,
                                raft::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests,
                        DistanceEucExpTestXequalYF,
                        ::testing::ValuesIn(inputsXeqYf));

const std::vector<DistanceInputs<double>> inputsd = {
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001, 1024, 1024, 32, false, 1234ULL},
  {0.001, 1024, 32, 1024, false, 1234ULL},
  {0.001, 32, 1024, 1024, false, 1234ULL},
  {0.003, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceEucExpTest<double> DistanceEucExpTestD;
TEST_P(DistanceEucExpTestD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceEucExpTestD, ::testing::ValuesIn(inputsd));

class BigMatrixEucExp : public BigMatrixDistanceTest<raft::distance::DistanceType::L2Expanded> {};
TEST_F(BigMatrixEucExp, Result) {}
}  // end namespace distance
}  // end namespace raft
