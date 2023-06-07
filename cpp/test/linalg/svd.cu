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
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/init.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace raft {
namespace linalg {

template <typename T>
struct SvdInputs {
  T tolerance;
  int len;
  int n_row;
  int n_col;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SvdInputs<T>& dims)
{
  return os;
}

template <typename T>
class SvdTest : public ::testing::TestWithParam<SvdInputs<T>> {
 public:
  SvdTest()
    : params(::testing::TestWithParam<SvdInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      data(params.len, stream),
      left_eig_vectors_qr(params.n_row * params.n_col, stream),
      right_eig_vectors_trans_qr(params.n_col * params.n_col, stream),
      sing_vals_qr(params.n_col, stream),
      left_eig_vectors_ref(params.n_row * params.n_col, stream),
      right_eig_vectors_ref(params.n_col * params.n_col, stream),
      sing_vals_ref(params.len, stream)
  {
  }

 protected:
  void test_API()
  {
    auto data_view = raft::make_device_matrix_view<const T, int, raft::col_major>(
      data.data(), params.n_row, params.n_col);
    auto sing_vals_view = raft::make_device_vector_view<T, int>(sing_vals_qr.data(), params.n_col);
    auto left_eig_vectors_view = raft::make_device_matrix_view<T, int, raft::col_major>(
      left_eig_vectors_qr.data(), params.n_row, params.n_col);
    auto right_eig_vectors_view = raft::make_device_matrix_view<T, int, raft::col_major>(
      right_eig_vectors_trans_qr.data(), params.n_col, params.n_col);
    raft::linalg::svd_eig(handle, data_view, sing_vals_view, right_eig_vectors_view, std::nullopt);
    raft::linalg::svd_qr(handle, data_view, sing_vals_view);
    raft::linalg::svd_qr(
      handle, data_view, sing_vals_view, std::make_optional(left_eig_vectors_view));
    raft::linalg::svd_qr(
      handle, data_view, sing_vals_view, std::nullopt, std::make_optional(right_eig_vectors_view));
    raft::linalg::svd_qr_transpose_right_vec(handle, data_view, sing_vals_view);
    raft::linalg::svd_qr_transpose_right_vec(
      handle, data_view, sing_vals_view, std::make_optional(left_eig_vectors_view));
    raft::linalg::svd_qr_transpose_right_vec(
      handle, data_view, sing_vals_view, std::nullopt, std::make_optional(right_eig_vectors_view));
  }

  void test_qr()
  {
    auto data_view = raft::make_device_matrix_view<const T, int, raft::col_major>(
      data.data(), params.n_row, params.n_col);
    auto sing_vals_qr_view =
      raft::make_device_vector_view<T, int>(sing_vals_qr.data(), params.n_col);
    auto left_eig_vectors_qr_view =
      std::optional(raft::make_device_matrix_view<T, int, raft::col_major>(
        left_eig_vectors_qr.data(), params.n_row, params.n_col));
    auto right_eig_vectors_trans_qr_view =
      std::make_optional(raft::make_device_matrix_view<T, int, raft::col_major>(
        right_eig_vectors_trans_qr.data(), params.n_col, params.n_col));

    svd_qr_transpose_right_vec(handle,
                               data_view,
                               sing_vals_qr_view,
                               left_eig_vectors_qr_view,
                               right_eig_vectors_trans_qr_view);
    resource::sync_stream(handle, stream);
  }

  void SetUp() override
  {
    int len = params.len;

    ASSERT(params.n_row == 3, "This test only supports nrows=3!");
    ASSERT(params.len == 6, "This test only supports len=6!");
    T data_h[] = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
    raft::update_device(data.data(), data_h, len, stream);

    int left_evl  = params.n_row * params.n_col;
    int right_evl = params.n_col * params.n_col;

    T left_eig_vectors_ref_h[] = {-0.308219, -0.906133, -0.289695, 0.488195, 0.110706, -0.865685};

    T right_eig_vectors_ref_h[] = {-0.638636, -0.769509, -0.769509, 0.638636};

    T sing_vals_ref_h[] = {7.065283, 1.040081};

    raft::update_device(left_eig_vectors_ref.data(), left_eig_vectors_ref_h, left_evl, stream);
    raft::update_device(right_eig_vectors_ref.data(), right_eig_vectors_ref_h, right_evl, stream);
    raft::update_device(sing_vals_ref.data(), sing_vals_ref_h, params.n_col, stream);

    test_API();
    raft::update_device(data.data(), data_h, len, stream);
    test_qr();
  }

 protected:
  raft::resources handle;
  cudaStream_t stream;

  SvdInputs<T> params;
  rmm::device_uvector<T> data, left_eig_vectors_qr, right_eig_vectors_trans_qr, sing_vals_qr,
    left_eig_vectors_ref, right_eig_vectors_ref, sing_vals_ref;
};

const std::vector<SvdInputs<float>> inputsf2 = {{0.00001f, 3 * 2, 3, 2, 1234ULL}};

const std::vector<SvdInputs<double>> inputsd2 = {{0.00001, 3 * 2, 3, 2, 1234ULL}};

typedef SvdTest<float> SvdTestValF;
TEST_P(SvdTestValF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double> SvdTestValD;
TEST_P(SvdTestValD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(sing_vals_ref.data(),
                                sing_vals_qr.data(),
                                params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef SvdTest<float> SvdTestLeftVecF;
TEST_P(SvdTestLeftVecF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double> SvdTestLeftVecD;
TEST_P(SvdTestLeftVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(left_eig_vectors_ref.data(),
                                left_eig_vectors_qr.data(),
                                params.n_row * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

typedef SvdTest<float> SvdTestRightVecF;
TEST_P(SvdTestRightVecF, Result)
{
  ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_trans_qr.data(),
                                params.n_col * params.n_col,
                                raft::CompareApproxAbs<float>(params.tolerance)));
}

typedef SvdTest<double> SvdTestRightVecD;
TEST_P(SvdTestRightVecD, Result)
{
  ASSERT_TRUE(raft::devArrMatch(right_eig_vectors_ref.data(),
                                right_eig_vectors_trans_qr.data(),
                                params.n_col * params.n_col,
                                raft::CompareApproxAbs<double>(params.tolerance)));
}

INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestValF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestValD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestRightVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestRightVecD, ::testing::ValuesIn(inputsd2));

// INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestRightVecF,
// ::testing::ValuesIn(inputsf2));

// INSTANTIATE_TEST_SUITE_P(SvdTests, SvdTestRightVecD,
//::testing::ValuesIn(inputsd2));

}  // end namespace linalg
}  // end namespace raft
