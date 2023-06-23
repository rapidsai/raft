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

#include "../linalg/matrix_vector_op.cuh"
#include "../test_utils.cuh"
#include <gtest/gtest.h>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/stats/mean.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace stats {

template <typename T, typename IdxType>
struct MeanCenterInputs {
  T tolerance, mean;
  IdxType rows, cols;
  bool sample, rowMajor, bcastAlongRows;
  unsigned long long int seed;
};

template <typename T, typename IdxType>
::std::ostream& operator<<(::std::ostream& os, const MeanCenterInputs<T, IdxType>& dims)
{
  return os;
}

template <typename T, typename IdxType>
class MeanCenterTest : public ::testing::TestWithParam<MeanCenterInputs<T, IdxType>> {
 public:
  MeanCenterTest()
    : params(::testing::TestWithParam<MeanCenterInputs<T, IdxType>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      rows(params.rows),
      cols(params.cols),
      out(rows * cols, stream),
      out_ref(rows * cols, stream),
      data(rows * cols, stream),
      meanVec(params.bcastAlongRows ? cols : rows, stream)
  {
  }

 protected:
  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    auto len         = rows * cols;
    auto meanVecSize = params.bcastAlongRows ? cols : rows;
    normal(handle, r, data.data(), len, params.mean, (T)1.0);
    raft::stats::mean(
      meanVec.data(), data.data(), cols, rows, params.sample, params.rowMajor, stream);
    if (params.rowMajor) {
      using layout = raft::row_major;
      mean_center(handle,
                  raft::make_device_matrix_view<const T, int, layout>(data.data(), rows, cols),
                  raft::make_device_vector_view<const T, int>(meanVec.data(), meanVecSize),
                  raft::make_device_matrix_view<T, int, layout>(out.data(), rows, cols),
                  params.bcastAlongRows);
    } else {
      using layout = raft::col_major;
      mean_center(handle,
                  raft::make_device_matrix_view<const T, int, layout>(data.data(), rows, cols),
                  raft::make_device_vector_view<const T, int>(meanVec.data(), meanVecSize),
                  raft::make_device_matrix_view<T, int, layout>(out.data(), rows, cols),
                  params.bcastAlongRows);
    }
    raft::linalg::naiveMatVec(out_ref.data(),
                              data.data(),
                              meanVec.data(),
                              cols,
                              rows,
                              params.rowMajor,
                              params.bcastAlongRows,
                              (T)-1.0,
                              stream);
    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  MeanCenterInputs<T, IdxType> params;
  int rows, cols;
  rmm::device_uvector<T> data, meanVec, out, out_ref;
};

const std::vector<MeanCenterInputs<float, int>> inputsf_i32 = {
  {0.05f, 1.f, 1024, 32, true, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, true, false, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, true, false, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, true, false, 1234ULL}};
typedef MeanCenterTest<float, int> MeanCenterTestF_i32;
TEST_P(MeanCenterTestF_i32, Result)
{
  ASSERT_TRUE(devArrMatch(
    out.data(), out_ref.data(), params.cols, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MeanCenterTests, MeanCenterTestF_i32, ::testing::ValuesIn(inputsf_i32));

const std::vector<MeanCenterInputs<float, size_t>> inputsf_i64 = {
  {0.05f, 1.f, 1024, 32, true, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, false, true, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, false, true, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, true, true, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, true, true, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, false, false, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, false, false, 1234ULL},
  {0.05f, 1.f, 1024, 32, true, true, false, 1234ULL},
  {0.05f, 1.f, 1024, 64, true, true, false, 1234ULL},
  {0.05f, 1.f, 1024, 128, true, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 32, false, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 64, false, true, false, 1234ULL},
  {0.05f, -1.f, 1024, 128, false, true, false, 1234ULL}};
typedef MeanCenterTest<float, size_t> MeanCenterTestF_i64;
TEST_P(MeanCenterTestF_i64, Result)
{
  ASSERT_TRUE(devArrMatch(
    out.data(), out_ref.data(), params.cols, raft::CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MeanCenterTests, MeanCenterTestF_i64, ::testing::ValuesIn(inputsf_i64));

const std::vector<MeanCenterInputs<double, int>> inputsd_i32 = {
  {0.05, 1.0, 1024, 32, true, false, true, 1234ULL},
  {0.05, 1.0, 1024, 64, true, false, true, 1234ULL},
  {0.05, 1.0, 1024, 128, true, false, true, 1234ULL},
  {0.05, -1.0, 1024, 32, false, false, true, 1234ULL},
  {0.05, -1.0, 1024, 64, false, false, true, 1234ULL},
  {0.05, -1.0, 1024, 128, false, false, true, 1234ULL},
  {0.05, 1.0, 1024, 32, true, true, true, 1234ULL},
  {0.05, 1.0, 1024, 64, true, true, true, 1234ULL},
  {0.05, 1.0, 1024, 128, true, true, true, 1234ULL},
  {0.05, -1.0, 1024, 32, false, true, true, 1234ULL},
  {0.05, -1.0, 1024, 64, false, true, true, 1234ULL},
  {0.05, -1.0, 1024, 128, false, true, true, 1234ULL},
  {0.05, 1.0, 1024, 32, true, false, false, 1234ULL},
  {0.05, 1.0, 1024, 64, true, false, false, 1234ULL},
  {0.05, 1.0, 1024, 128, true, false, false, 1234ULL},
  {0.05, -1.0, 1024, 32, false, false, false, 1234ULL},
  {0.05, -1.0, 1024, 64, false, false, false, 1234ULL},
  {0.05, -1.0, 1024, 128, false, false, false, 1234ULL},
  {0.05, 1.0, 1024, 32, true, true, false, 1234ULL},
  {0.05, 1.0, 1024, 64, true, true, false, 1234ULL},
  {0.05, 1.0, 1024, 128, true, true, false, 1234ULL},
  {0.05, -1.0, 1024, 32, false, true, false, 1234ULL},
  {0.05, -1.0, 1024, 64, false, true, false, 1234ULL},
  {0.05, -1.0, 1024, 128, false, true, false, 1234ULL}};
typedef MeanCenterTest<double, int> MeanCenterTestD_i32;
TEST_P(MeanCenterTestD_i32, Result)
{
  ASSERT_TRUE(devArrMatch(
    out.data(), out_ref.data(), params.cols, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MeanCenterTests, MeanCenterTestD_i32, ::testing::ValuesIn(inputsd_i32));

const std::vector<MeanCenterInputs<double, size_t>> inputsd_i64 = {
  {0.05, 1.0, 1024, 32, true, false, true, 1234ULL},
  {0.05, 1.0, 1024, 64, true, false, true, 1234ULL},
  {0.05, 1.0, 1024, 128, true, false, true, 1234ULL},
  {0.05, -1.0, 1024, 32, false, false, true, 1234ULL},
  {0.05, -1.0, 1024, 64, false, false, true, 1234ULL},
  {0.05, -1.0, 1024, 128, false, false, true, 1234ULL},
  {0.05, 1.0, 1024, 32, true, true, true, 1234ULL},
  {0.05, 1.0, 1024, 64, true, true, true, 1234ULL},
  {0.05, 1.0, 1024, 128, true, true, true, 1234ULL},
  {0.05, -1.0, 1024, 32, false, true, true, 1234ULL},
  {0.05, -1.0, 1024, 64, false, true, true, 1234ULL},
  {0.05, -1.0, 1024, 128, false, true, true, 1234ULL},
  {0.05, 1.0, 1024, 32, true, false, false, 1234ULL},
  {0.05, 1.0, 1024, 64, true, false, false, 1234ULL},
  {0.05, 1.0, 1024, 128, true, false, false, 1234ULL},
  {0.05, -1.0, 1024, 32, false, false, false, 1234ULL},
  {0.05, -1.0, 1024, 64, false, false, false, 1234ULL},
  {0.05, -1.0, 1024, 128, false, false, false, 1234ULL},
  {0.05, 1.0, 1024, 32, true, true, false, 1234ULL},
  {0.05, 1.0, 1024, 64, true, true, false, 1234ULL},
  {0.05, 1.0, 1024, 128, true, true, false, 1234ULL},
  {0.05, -1.0, 1024, 32, false, true, false, 1234ULL},
  {0.05, -1.0, 1024, 64, false, true, false, 1234ULL},
  {0.05, -1.0, 1024, 128, false, true, false, 1234ULL}};
typedef MeanCenterTest<double, size_t> MeanCenterTestD_i64;
TEST_P(MeanCenterTestD_i64, Result)
{
  ASSERT_TRUE(devArrMatch(
    out.data(), out_ref.data(), params.cols, raft::CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_SUITE_P(MeanCenterTests, MeanCenterTestD_i64, ::testing::ValuesIn(inputsd_i64));

}  // end namespace stats
}  // end namespace raft
