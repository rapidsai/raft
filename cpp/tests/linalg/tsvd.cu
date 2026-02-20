/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../test_utils.cuh"

#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/linalg/tsvd.cuh>
#include <raft/random/rng.cuh>
#include <raft/util/cudart_utils.hpp>

#include <gtest/gtest.h>
#include <test_utils.h>

#include <vector>

namespace raft::linalg {

template <typename T>
struct TsvdInputs {
  T tolerance;
  int n_row;
  int n_col;
  int n_row2;
  int n_col2;
  float redundancy;
  unsigned long long int seed;
  int algo;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const TsvdInputs<T>& dims)
{
  return os;
}

template <typename T>
class TsvdTest : public ::testing::TestWithParam<TsvdInputs<T>> {
 public:
  TsvdTest()
    : params(::testing::TestWithParam<TsvdInputs<T>>::GetParam()),
      stream(resource::get_cuda_stream(handle)),
      components(0, stream),
      components_ref(0, stream),
      data2(0, stream),
      data2_back(0, stream)
  {
    basicTest();
    advancedTest();
  }

 protected:
  void basicTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.n_row * params.n_col;

    rmm::device_uvector<T> data(len, stream);

    std::vector<T> data_h = {1.0, 2.0, 4.0, 2.0, 4.0, 5.0, 5.0, 4.0, 2.0, 1.0, 6.0, 4.0};
    data_h.resize(len);
    raft::update_device(data.data(), data_h.data(), len, stream);

    int len_comp = params.n_col * params.n_col;
    components.resize(len_comp, stream);
    rmm::device_uvector<T> singular_vals(params.n_col, stream);

    std::vector<T> components_ref_h = {
      0.3951, 0.1532, 0.9058, 0.7111, -0.6752, -0.1959, 0.5816, 0.7215, -0.3757};
    components_ref_h.resize(len_comp);

    components_ref.resize(len_comp, stream);
    raft::update_device(components_ref.data(), components_ref_h.data(), len_comp, stream);

    paramsTSVD prms;
    prms.n_cols       = params.n_col;
    prms.n_rows       = params.n_row;
    prms.n_components = params.n_col;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else
      prms.algorithm = solver::COV_EIG_JACOBI;

    auto input_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      data.data(), prms.n_rows, prms.n_cols);
    auto components_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      components.data(), prms.n_components, prms.n_cols);
    auto singular_vals_view =
      raft::make_device_vector_view<T, std::size_t>(singular_vals.data(), prms.n_components);

    tsvd_fit(handle, prms, input_view, components_view, singular_vals_view);
  }

  void advancedTest()
  {
    raft::random::Rng r(params.seed, raft::random::GenPC);
    int len = params.n_row2 * params.n_col2;

    paramsTSVD prms;
    prms.n_cols       = params.n_col2;
    prms.n_rows       = params.n_row2;
    prms.n_components = params.n_col2;
    if (params.algo == 0)
      prms.algorithm = solver::COV_EIG_DQ;
    else if (params.algo == 1)
      prms.algorithm = solver::COV_EIG_JACOBI;
    else
      prms.n_components = params.n_col2 - 15;

    data2.resize(len, stream);
    int redundant_cols = int(params.redundancy * params.n_col2);
    int redundant_len  = params.n_row2 * redundant_cols;

    int informative_cols = params.n_col2 - redundant_cols;
    int informative_len  = params.n_row2 * informative_cols;

    r.uniform(data2.data(), informative_len, T(-1.0), T(1.0), stream);
    RAFT_CUDA_TRY(cudaMemcpyAsync(data2.data() + informative_len,
                                  data2.data(),
                                  redundant_len * sizeof(T),
                                  cudaMemcpyDeviceToDevice,
                                  stream));
    rmm::device_uvector<T> data2_trans(prms.n_rows * prms.n_components, stream);

    int len_comp = params.n_col2 * prms.n_components;
    rmm::device_uvector<T> components2(len_comp, stream);
    rmm::device_uvector<T> explained_vars2(prms.n_components, stream);
    rmm::device_uvector<T> explained_var_ratio2(prms.n_components, stream);
    rmm::device_uvector<T> singular_vals2(prms.n_components, stream);

    auto input_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      data2.data(), prms.n_rows, prms.n_cols);
    auto trans_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      data2_trans.data(), prms.n_rows, prms.n_components);
    auto comp_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      components2.data(), prms.n_components, prms.n_cols);
    auto ev_view =
      raft::make_device_vector_view<T, std::size_t>(explained_vars2.data(), prms.n_components);
    auto evr_view =
      raft::make_device_vector_view<T, std::size_t>(explained_var_ratio2.data(), prms.n_components);
    auto sv_view =
      raft::make_device_vector_view<T, std::size_t>(singular_vals2.data(), prms.n_components);

    tsvd_fit_transform(handle, prms, input_view, trans_view, comp_view, ev_view, evr_view, sv_view);

    data2_back.resize(len, stream);

    auto trans_in_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      data2_trans.data(), prms.n_rows, prms.n_components);
    auto comp_in_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      components2.data(), prms.n_components, prms.n_cols);
    auto output_view = raft::make_device_matrix_view<T, std::size_t, raft::col_major>(
      data2_back.data(), prms.n_rows, prms.n_cols);

    tsvd_inverse_transform(handle, prms, trans_in_view, comp_in_view, output_view);
  }

 protected:
  raft::device_resources handle;
  cudaStream_t stream;

  TsvdInputs<T> params;
  rmm::device_uvector<T> components, components_ref, data2, data2_back;
};

const std::vector<TsvdInputs<float>> inputsf2 = {{0.01f, 4, 3, 1024, 128, 0.25f, 1234ULL, 0},
                                                 {0.01f, 4, 3, 1024, 128, 0.25f, 1234ULL, 1},
                                                 {0.04f, 4, 3, 512, 64, 0.25f, 1234ULL, 2},
                                                 {0.04f, 4, 3, 512, 64, 0.25f, 1234ULL, 2}};

const std::vector<TsvdInputs<double>> inputsd2 = {{0.01, 4, 3, 1024, 128, 0.25f, 1234ULL, 0},
                                                  {0.01, 4, 3, 1024, 128, 0.25f, 1234ULL, 1},
                                                  {0.05, 4, 3, 512, 64, 0.25f, 1234ULL, 2},
                                                  {0.05, 4, 3, 512, 64, 0.25f, 1234ULL, 2}};

typedef TsvdTest<float> TsvdTestLeftVecF;
TEST_P(TsvdTestLeftVecF, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_col),
                          raft::CompareApprox<float>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef TsvdTest<double> TsvdTestLeftVecD;
TEST_P(TsvdTestLeftVecD, Result)
{
  ASSERT_TRUE(devArrMatch(components.data(),
                          components_ref.data(),
                          (params.n_col * params.n_col),
                          raft::CompareApprox<double>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef TsvdTest<float> TsvdTestDataVecF;
TEST_P(TsvdTestDataVecF, Result)
{
  ASSERT_TRUE(devArrMatch(data2.data(),
                          data2_back.data(),
                          (params.n_col2 * params.n_col2),
                          raft::CompareApprox<float>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

typedef TsvdTest<double> TsvdTestDataVecD;
TEST_P(TsvdTestDataVecD, Result)
{
  ASSERT_TRUE(devArrMatch(data2.data(),
                          data2_back.data(),
                          (params.n_col2 * params.n_col2),
                          raft::CompareApprox<double>(params.tolerance),
                          resource::get_cuda_stream(handle)));
}

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestLeftVecD, ::testing::ValuesIn(inputsd2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(TsvdTests, TsvdTestDataVecD, ::testing::ValuesIn(inputsd2));

}  // end namespace raft::linalg
