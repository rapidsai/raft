/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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
class DistanceExpDice : public DistanceTest<raft::distance::DistanceType::DiceExpanded, DataType> {
};

template <typename DataType>
class DistanceExpDiceXequalY
  : public DistanceTestSameBuffer<raft::distance::DistanceType::DiceExpanded, DataType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 128, (65536 + 128) * 128, 8, true, 1234ULL},
  {0.001f, 1024, 1024, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsXeqYf = {
  {0.01f, 1024, 1024, 32, true, 1234ULL},
  {0.01f, 1024, 32, 1024, true, 1234ULL},
  {0.01f, 32, 1024, 1024, true, 1234ULL},
  {0.03f, 1024, 1024, 1024, true, 1234ULL},
  {0.01f, 1024, 1024, 32, false, 1234ULL},
  {0.01f, 1024, 32, 1024, false, 1234ULL},
  {0.01f, 32, 1024, 1024, false, 1234ULL},
  {0.03f, 1024, 1024, 1024, false, 1234ULL},
};

const std::vector<DistanceInputs<float>> inputsNaN = {
  {0.001f, (65536 + 128) * 128, 128, 8, false, 1234ULL}};

typedef DistanceExpDice<float> DistanceExpDiceNaN;
TEST_P(DistanceExpDiceNaN, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_FALSE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceNaN, ::testing::ValuesIn(inputsNaN));

typedef DistanceExpDice<float> DistanceExpDiceF;
TEST_P(DistanceExpDiceF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceF, ::testing::ValuesIn(inputsf));

typedef DistanceExpDiceXequalY<float> DistanceExpDiceXequalYF;
TEST_P(DistanceExpDiceXequalYF, Result)
{
  int m = params.m;
  int n = params.m;
  ASSERT_TRUE(raft::devArrMatch(dist_ref[0].data(),
                                dist[0].data(),
                                m,
                                n,
                                raft::CompareApprox<float>(params.tolerance),
                                stream));
  n = params.isRowMajor ? m : m / 2;
  m = params.isRowMajor ? m / 2 : m;

  ASSERT_TRUE(raft::devArrMatch(dist_ref[1].data(),
                                dist[1].data(),
                                m,
                                n,
                                raft::CompareApprox<float>(params.tolerance),
                                stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceXequalYF, ::testing::ValuesIn(inputsXeqYf));

const std::vector<DistanceInputs<float>> inputsd = {
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceExpDice<float> DistanceExpDiceD;
TEST_P(DistanceExpDiceD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;
  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceExpDiceD, ::testing::ValuesIn(inputsd));

class BigMatrixDice : public BigMatrixDistanceTest<raft::distance::DistanceType::DiceExpanded> {};
TEST_F(BigMatrixDice, Result) {}

}  // end namespace distance
}  // end namespace raft
