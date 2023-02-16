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
class DistanceInnerProduct
  : public DistanceTest<raft::distance::DistanceType::InnerProduct, DataType> {
};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 10, 5, 32, true, 1234ULL},
  {0.001f, 1024, 32, 1024, true, 1234ULL},
  {0.001f, 32, 1024, 1024, true, 1234ULL},
  {0.003f, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceInnerProduct<float> DistanceInnerProductF;
TEST_P(DistanceInnerProductF, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;

  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<float>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceInnerProductF, ::testing::ValuesIn(inputsf));

const std::vector<DistanceInputs<double>> inputsd = {
  {0.001, 1024, 1024, 32, true, 1234ULL},
  {0.001, 1024, 32, 1024, true, 1234ULL},
  {0.001, 32, 1024, 1024, true, 1234ULL},
  {0.003, 1024, 1024, 1024, true, 1234ULL},
  {0.001f, 1024, 1024, 32, false, 1234ULL},
  {0.001f, 1024, 32, 1024, false, 1234ULL},
  {0.001f, 32, 1024, 1024, false, 1234ULL},
  {0.003f, 1024, 1024, 1024, false, 1234ULL},
};
typedef DistanceInnerProduct<double> DistanceInnerProductD;
TEST_P(DistanceInnerProductD, Result)
{
  int m = params.isRowMajor ? params.m : params.n;
  int n = params.isRowMajor ? params.n : params.m;

  ASSERT_TRUE(devArrMatch(
    dist_ref.data(), dist.data(), m, n, raft::CompareApprox<double>(params.tolerance), stream));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceInnerProductD, ::testing::ValuesIn(inputsd));

class BigMatrixInnerProduct
  : public BigMatrixDistanceTest<raft::distance::DistanceType::InnerProduct> {
};
TEST_F(BigMatrixInnerProduct, Result) {}

}  // end namespace distance
}  // end namespace raft
