/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

namespace raft {
namespace linalg {

template <typename T, typename IdxT>
struct RowNormalizeInputs {
  T tolerance;
  int rows, cols;
  unsigned long long int seed;
};

template <typename T, typename IdxT>
void rowNormalizeRef(T* out, const T* in, IdxT cols, IdxT rows, cudaStream_t stream)
{
  rmm::device_uvector<T> norm(rows, stream);
  raft::linalg::rowNorm(
    norm.data(), in, cols, rows, raft::linalg::L2Norm, true, stream, [] __device__(T a) {
      return sqrt(a);
    });
  raft::linalg::matrixVectorOp(
    out,
    in,
    norm.data(),
    cols,
    rows,
    true,
    false,
    [] __device__(T a, T b) { return a / b; },
    stream);
}

template <typename T, typename IdxT>
class RowNormalizeTest : public ::testing::TestWithParam<RowNormalizeInputs<T, IdxT>> {
 public:
  RowNormalizeTest()
    : params(::testing::TestWithParam<RowNormalizeInputs<T, IdxT>>::GetParam()),
      stream(handle.get_stream()),
      data(params.rows * params.cols, stream),
      out_exp(params.rows * params.cols, stream),
      out_act(params.rows * params.cols, stream)
  {
  }

  void SetUp() override
  {
    raft::random::RngState r(params.seed);
    int len = params.rows * params.cols;
    uniform(handle, r, data.data(), len, T(-10.0), T(10.0));

    rowNormalizeRef(out_exp.data(), data.data(), params.cols, params.rows, stream);

    auto input_view = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      data.data(), params.rows, params.cols);
    auto output_view = raft::make_device_matrix_view<T, IdxT, raft::row_major>(
      out_act.data(), params.rows, params.cols);
    raft::linalg::rowNormalize(handle, input_view, output_view);

    handle.sync_stream(stream);
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  RowNormalizeInputs<T, IdxT> params;
  rmm::device_uvector<T> data, out_exp, out_act;
};

const std::vector<RowNormalizeInputs<float, int>> inputsf_i32 =
  raft::util::itertools::product<RowNormalizeInputs<float, int>>(
    {0.00001f}, {11, 101, 1000, 12345}, {2, 3, 7, 12, 33, 125, 254});
const std::vector<RowNormalizeInputs<double, int>> inputsd_i32 =
  raft::util::itertools::product<RowNormalizeInputs<double, int>>(
    {0.00000001}, {11, 101, 1000, 12345}, {2, 3, 7, 12, 33, 125, 254});

typedef RowNormalizeTest<float, int> RowNormalizeTestF_I32;
TEST_P(RowNormalizeTestF_I32, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_exp.data(),
                                out_act.data(),
                                params.rows * params.cols,
                                raft::CompareApprox(params.tolerance)));
}

typedef RowNormalizeTest<double, int> RowNormalizeTestD_I32;
TEST_P(RowNormalizeTestD_I32, Result)
{
  ASSERT_TRUE(raft::devArrMatch(out_exp.data(),
                                out_act.data(),
                                params.rows * params.cols,
                                raft::CompareApprox(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(RowNormalizeTests, RowNormalizeTestF_I32, ::testing::ValuesIn(inputsf_i32));

INSTANTIATE_TEST_CASE_P(RowNormalizeTests, RowNormalizeTestD_I32, ::testing::ValuesIn(inputsd_i32));

}  // end namespace linalg
}  // end namespace raft
