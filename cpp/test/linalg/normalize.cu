/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/normalize.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/itertools.hpp>

namespace raft {
namespace linalg {

template <typename T, typename IdxT>
struct RowNormalizeInputs {
  T tolerance;
  IdxT rows, cols;
  raft::linalg::NormType norm_type;
  unsigned long long int seed;
};

template <typename T, typename IdxT>
::std::ostream& operator<<(::std::ostream& os, const RowNormalizeInputs<T, IdxT>& I)
{
  os << "{ " << I.tolerance << ", " << I.rows << ", " << I.cols << ", " << I.norm_type << ", "
     << I.seed << '}' << std::endl;
  return os;
}

template <typename T, typename IdxT>
void rowNormalizeRef(
  T* out, const T* in, IdxT cols, IdxT rows, raft::linalg::NormType norm_type, cudaStream_t stream)
{
  rmm::device_uvector<T> norm(rows, stream);
  if (norm_type == raft::linalg::L2Norm) {
    raft::linalg::rowNorm(norm.data(), in, cols, rows, norm_type, true, stream, raft::sqrt_op());
  } else {
    raft::linalg::rowNorm(
      norm.data(), in, cols, rows, norm_type, true, stream, raft::identity_op());
  }
  raft::linalg::matrixVectorOp(
    out, in, norm.data(), cols, rows, true, false, raft::div_op{}, stream);
}

template <typename T, typename IdxT>
class RowNormalizeTest : public ::testing::TestWithParam<RowNormalizeInputs<T, IdxT>> {
 public:
  RowNormalizeTest()
    : params(::testing::TestWithParam<RowNormalizeInputs<T, IdxT>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
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

    rowNormalizeRef(
      out_exp.data(), data.data(), params.cols, params.rows, params.norm_type, stream);

    auto input_view = raft::make_device_matrix_view<const T, IdxT, raft::row_major>(
      data.data(), params.rows, params.cols);
    auto output_view = raft::make_device_matrix_view<T, IdxT, raft::row_major>(
      out_act.data(), params.rows, params.cols);
    raft::linalg::row_normalize(handle, input_view, output_view, params.norm_type);

    resource::sync_stream(handle, stream);
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  RowNormalizeInputs<T, IdxT> params;
  rmm::device_uvector<T> data, out_exp, out_act;
};

const std::vector<RowNormalizeInputs<float, int>> inputsf_i32 =
  raft::util::itertools::product<RowNormalizeInputs<float, int>>(
    {0.00001f},
    {11, 101, 12345},
    {2, 3, 7, 12, 33, 125, 254},
    {raft::linalg::L1Norm, raft::linalg::L2Norm, raft::linalg::LinfNorm},
    {1234ULL});
const std::vector<RowNormalizeInputs<double, int>> inputsd_i32 =
  raft::util::itertools::product<RowNormalizeInputs<double, int>>(
    {0.00000001},
    {11, 101, 12345},
    {2, 3, 7, 12, 33, 125, 254},
    {raft::linalg::L1Norm, raft::linalg::L2Norm, raft::linalg::LinfNorm},
    {1234ULL});
const std::vector<RowNormalizeInputs<float, uint32_t>> inputsf_u32 =
  raft::util::itertools::product<RowNormalizeInputs<float, uint32_t>>(
    {0.00001f},
    {11u, 101u, 12345u},
    {2u, 3u, 7u, 12u, 33u, 125u, 254u},
    {raft::linalg::L1Norm, raft::linalg::L2Norm, raft::linalg::LinfNorm},
    {1234ULL});
const std::vector<RowNormalizeInputs<double, uint32_t>> inputsd_u32 =
  raft::util::itertools::product<RowNormalizeInputs<double, uint32_t>>(
    {0.00000001},
    {11u, 101u, 12345u},
    {2u, 3u, 7u, 12u, 33u, 125u, 254u},
    {raft::linalg::L1Norm, raft::linalg::L2Norm, raft::linalg::LinfNorm},
    {1234ULL});

#define ROWNORMALIZE_TEST(test_type, test_name, test_inputs)               \
  typedef RAFT_DEPAREN(test_type) test_name;                               \
  TEST_P(test_name, Result)                                                \
  {                                                                        \
    ASSERT_TRUE(raft::devArrMatch(out_exp.data(),                          \
                                  out_act.data(),                          \
                                  params.rows* params.cols,                \
                                  raft::CompareApprox(params.tolerance))); \
  }                                                                        \
  INSTANTIATE_TEST_CASE_P(RowNormalizeTests, test_name, ::testing::ValuesIn(test_inputs))

ROWNORMALIZE_TEST((RowNormalizeTest<float, int>), RowNormalizeTestFI32, inputsf_i32);
ROWNORMALIZE_TEST((RowNormalizeTest<double, int>), RowNormalizeTestDI32, inputsd_i32);
ROWNORMALIZE_TEST((RowNormalizeTest<float, uint32_t>), RowNormalizeTestFU32, inputsf_u32);
ROWNORMALIZE_TEST((RowNormalizeTest<double, uint32_t>), RowNormalizeTestDU32, inputsd_u32);

}  // end namespace linalg
}  // end namespace raft
