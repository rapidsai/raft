/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include "reduce.cuh"
#include <gtest/gtest.h>
#include <raft/linalg/reduce.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename InType, typename OutType>
struct ReduceInputs {
  OutType tolerance;
  int rows, cols;
  bool rowMajor, alongRows;
  unsigned long long int seed;
};

template <typename InType, typename OutType>
::std::ostream& operator<<(::std::ostream& os, const ReduceInputs<InType, OutType>& dims)
{
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename InType, typename OutType>
void reduceLaunch(OutType* dots,
                  const InType* data,
                  int cols,
                  int rows,
                  bool rowMajor,
                  bool alongRows,
                  bool inplace,
                  cudaStream_t stream)
{
  Apply apply     = alongRows ? Apply::ALONG_ROWS : Apply::ALONG_COLUMNS;
  int output_size = alongRows ? cols : rows;

  auto output_view_row_major = raft::make_device_vector_view(dots, output_size);
  auto input_view_row_major  = raft::make_device_matrix_view(data, rows, cols);

  auto output_view_col_major = raft::make_device_vector_view<OutType, uint32_t>(dots, output_size);
  auto input_view_col_major =
    raft::make_device_matrix_view<const InType, uint32_t, raft::col_major>(data, rows, cols);

  raft::handle_t handle{stream};

  if (rowMajor) {
    reduce(handle,
           input_view_row_major,
           output_view_row_major,
           (OutType)0,

           apply,
           inplace,
           [] __device__(InType in, int i) { return static_cast<OutType>(in * in); });
  } else {
    reduce(handle,
           input_view_col_major,
           output_view_col_major,
           (OutType)0,

           apply,
           inplace,
           [] __device__(InType in, int i) { return static_cast<OutType>(in * in); });
  }
}

template <typename InType, typename OutType>
class ReduceTest : public ::testing::TestWithParam<ReduceInputs<InType, OutType>> {
 public:
  ReduceTest()
    : params(::testing::TestWithParam<ReduceInputs<InType, OutType>>::GetParam()),
      stream(handle.get_stream()),
      data(params.rows * params.cols, stream),
      dots_exp(params.alongRows ? params.rows : params.cols, stream),
      dots_act(params.alongRows ? params.rows : params.cols, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int rows = params.rows, cols = params.cols;
    int len = rows * cols;
    outlen  = params.alongRows ? rows : cols;
    uniform(handle, r, data.data(), len, InType(-1.0), InType(1.0));
    naiveReduction(
      dots_exp.data(), data.data(), cols, rows, params.rowMajor, params.alongRows, stream);

    // Perform reduction with default inplace = false first
    reduceLaunch(
      dots_act.data(), data.data(), cols, rows, params.rowMajor, params.alongRows, false, stream);
    // Add to result with inplace = true next, which shouldn't affect
    // in the case of coalescedReduction!
    if (!(params.rowMajor ^ params.alongRows)) {
      reduceLaunch(
        dots_act.data(), data.data(), cols, rows, params.rowMajor, params.alongRows, true, stream);
    }
    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  ReduceInputs<InType, OutType> params;
  rmm::device_uvector<InType> data;
  rmm::device_uvector<OutType> dots_exp, dots_act;
  int outlen;
};

const std::vector<ReduceInputs<float, float>> inputsff = {
  {0.000002f, 1024, 32, true, true, 1234ULL},
  {0.000002f, 1024, 64, true, true, 1234ULL},
  {0.000002f, 1024, 128, true, true, 1234ULL},
  {0.000002f, 1024, 256, true, true, 1234ULL},
  {0.000002f, 1024, 32, true, false, 1234ULL},
  {0.000002f, 1024, 64, true, false, 1234ULL},
  {0.000002f, 1024, 128, true, false, 1234ULL},
  {0.000002f, 1024, 256, true, false, 1234ULL},
  {0.000002f, 1024, 32, false, true, 1234ULL},
  {0.000002f, 1024, 64, false, true, 1234ULL},
  {0.000002f, 1024, 128, false, true, 1234ULL},
  {0.000002f, 1024, 256, false, true, 1234ULL},
  {0.000002f, 1024, 32, false, false, 1234ULL},
  {0.000002f, 1024, 64, false, false, 1234ULL},
  {0.000002f, 1024, 128, false, false, 1234ULL},
  {0.000002f, 1024, 256, false, false, 1234ULL}};

const std::vector<ReduceInputs<double, double>> inputsdd = {
  {0.000000001, 1024, 32, true, true, 1234ULL},
  {0.000000001, 1024, 64, true, true, 1234ULL},
  {0.000000001, 1024, 128, true, true, 1234ULL},
  {0.000000001, 1024, 256, true, true, 1234ULL},
  {0.000000001, 1024, 32, true, false, 1234ULL},
  {0.000000001, 1024, 64, true, false, 1234ULL},
  {0.000000001, 1024, 128, true, false, 1234ULL},
  {0.000000001, 1024, 256, true, false, 1234ULL},
  {0.000000001, 1024, 32, false, true, 1234ULL},
  {0.000000001, 1024, 64, false, true, 1234ULL},
  {0.000000001, 1024, 128, false, true, 1234ULL},
  {0.000000001, 1024, 256, false, true, 1234ULL},
  {0.000000001, 1024, 32, false, false, 1234ULL},
  {0.000000001, 1024, 64, false, false, 1234ULL},
  {0.000000001, 1024, 128, false, false, 1234ULL},
  {0.000000001, 1024, 256, false, false, 1234ULL}};

const std::vector<ReduceInputs<float, double>> inputsfd = {
  {0.000002f, 1024, 32, true, true, 1234ULL},
  {0.000002f, 1024, 64, true, true, 1234ULL},
  {0.000002f, 1024, 128, true, true, 1234ULL},
  {0.000002f, 1024, 256, true, true, 1234ULL},
  {0.000002f, 1024, 32, true, false, 1234ULL},
  {0.000002f, 1024, 64, true, false, 1234ULL},
  {0.000002f, 1024, 128, true, false, 1234ULL},
  {0.000002f, 1024, 256, true, false, 1234ULL},
  {0.000002f, 1024, 32, false, true, 1234ULL},
  {0.000002f, 1024, 64, false, true, 1234ULL},
  {0.000002f, 1024, 128, false, true, 1234ULL},
  {0.000002f, 1024, 256, false, true, 1234ULL},
  {0.000002f, 1024, 32, false, false, 1234ULL},
  {0.000002f, 1024, 64, false, false, 1234ULL},
  {0.000002f, 1024, 128, false, false, 1234ULL},
  {0.000002f, 1024, 256, false, false, 1234ULL}};

typedef ReduceTest<float, float> ReduceTestFF;
TEST_P(ReduceTestFF, Result)
{
  ASSERT_TRUE(devArrMatch(
    dots_exp.data(), dots_act.data(), outlen, raft::CompareApprox<float>(params.tolerance)));
}

typedef ReduceTest<double, double> ReduceTestDD;
TEST_P(ReduceTestDD, Result)
{
  ASSERT_TRUE(devArrMatch(
    dots_exp.data(), dots_act.data(), outlen, raft::CompareApprox<double>(params.tolerance)));
}

typedef ReduceTest<float, double> ReduceTestFD;
TEST_P(ReduceTestFD, Result)
{
  ASSERT_TRUE(devArrMatch(
    dots_exp.data(), dots_act.data(), outlen, raft::CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(ReduceTests, ReduceTestFF, ::testing::ValuesIn(inputsff));

INSTANTIATE_TEST_CASE_P(ReduceTests, ReduceTestDD, ::testing::ValuesIn(inputsdd));

INSTANTIATE_TEST_CASE_P(ReduceTests, ReduceTestFD, ::testing::ValuesIn(inputsfd));

}  // end namespace linalg
}  // end namespace raft
