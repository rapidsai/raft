/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
class DistanceCorrelation
  : public DistanceTest<raft::distance::DistanceType::CorrelationExpanded, DataType> {};

template <typename DataType>
class DistanceCorrelationXequalY
  : public DistanceTestSameBuffer<raft::distance::DistanceType::CorrelationExpanded, DataType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceCorrelation<float> DistanceCorrelationF;
TEST_P(DistanceCorrelationF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(raft::devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceCorrelationF, ::testing::ValuesIn(inputsf));

typedef DistanceCorrelationXequalY<float> DistanceCorrelationXequalYF;
TEST_P(DistanceCorrelationXequalYF, Result)
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
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceCorrelationXequalYF, ::testing::ValuesIn(inputsf));

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
typedef DistanceCorrelation<double> DistanceCorrelationD;
TEST_P(DistanceCorrelationD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(raft::devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceCorrelationD, ::testing::ValuesIn(inputsd));

class BigMatrixCorrelation
  : public BigMatrixDistanceTest<raft::distance::DistanceType::CorrelationExpanded> {};
TEST_F(BigMatrixCorrelation, Result) {}
}  // end namespace distance
}  // end namespace raft
