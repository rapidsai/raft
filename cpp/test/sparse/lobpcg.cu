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

#include <gtest/gtest.h>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/sparse/linalg/add.cuh>
#include <raft/sparse/solver/lobpcg.cuh>
#include <raft/spectral/matrix_wrappers.hpp>

#include "../test_utils.cuh"
#include "../test_utils.h"
#include <raft/util/cudart_utils.hpp>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename math_t, typename idx_t>
struct CSRMatrixVal {
  std::vector<idx_t> row_ind_ptr;
  std::vector<idx_t> row_ind;
  std::vector<math_t> values;
};

template <typename math_t, typename idx_t>
struct LOBPCGInputs {
  CSRMatrixVal<math_t, idx_t> matrix_a;
  std::vector<math_t> init_eigvecs;
  std::vector<math_t> exp_eigvals;
  std::vector<math_t> exp_eigvecs;
  idx_t n_components;
};

// Helper for b_orthonormalize optional arguments
template <typename value_t,
          typename index_t,
          typename b_opt_t,
          typename vbv_opt_t,
          typename v_max_opt_t>
void b_orthonormalize(const raft::handle_t& handle,
                      raft::device_matrix_view<value_t, index_t, raft::col_major> V,
                      raft::device_matrix_view<value_t, index_t, raft::col_major> BV,
                      b_opt_t&& B_opt         = std::nullopt,
                      vbv_opt_t&& VBV_opt     = std::nullopt,
                      v_max_opt_t&& V_max_opt = std::nullopt,
                      bool bv_is_empty        = true)
{
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> b =
    std::forward<b_opt_t>(B_opt);
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> vbv =
    std::forward<vbv_opt_t>(VBV_opt);
  std::optional<raft::device_vector_view<value_t, index_t>> v_max =
    std::forward<v_max_opt_t>(V_max_opt);
  raft::sparse::solver::detail::b_orthonormalize(handle, V, BV, b, vbv, v_max, bv_is_empty);
}

template <typename math_t, typename idx_t>
class LOBPCGTest : public ::testing::TestWithParam<LOBPCGInputs<math_t, idx_t>> {
 public:
  LOBPCGTest()
    : params(::testing::TestWithParam<LOBPCGInputs<math_t, idx_t>>::GetParam()),
      stream(handle.get_stream()),
      ind_a(params.matrix_a.row_ind.size(), stream),
      ind_ptr_a(params.matrix_a.row_ind_ptr.size(), stream),
      values_a(params.matrix_a.values.size(), stream),
      exp_eigvals(params.exp_eigvals.size(), stream),
      exp_eigvecs(params.exp_eigvecs.size(), stream),
      act_eigvals(params.exp_eigvals.size(), stream),
      act_eigvecs(params.exp_eigvecs.size(), stream)
  {
  }

 protected:
  void SetUp() override
  {
    n_rows_a = params.matrix_a.row_ind_ptr.size() - 1;
    nnz_a    = params.matrix_a.values.size();
  }

  void test_selectcolsif()
  {
    auto a = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 5, 8);
    auto c = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 5, 4);
    auto b = raft::make_device_vector<idx_t, idx_t>(handle, 8);
    raft::linalg::range(a.data_handle(), a.size(), handle.get_stream());
    std::vector<idx_t> select_h{0, 1, 1, 1, 0, 0, 0, 1};
    raft::copy(b.data_handle(), select_h.data(), 8, handle.get_stream());
    raft::sparse::solver::detail::selectColsIf(handle, a.view(), b.view(), c.view());
    std::vector<math_t> res(c.size());
    std::vector<math_t> expected{5,  6,  7,  8,  9,  10, 11, 12, 13, 14,
                                 15, 16, 17, 18, 19, 35, 36, 37, 38, 39};
    raft::copy(res.data(), c.data_handle(), c.size(), handle.get_stream());

    ASSERT_TRUE(hostVecMatch(expected, res, raft::CompareApprox<math_t>(0.0001)));
  }

  void test_bmat()
  {
    auto total = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 6, 6);
    auto x1    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x2    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x3    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x4    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x5    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x6    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x7    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x8    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto x9    = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    raft::linalg::range(x1.data_handle(), 0, 4, handle.get_stream());
    raft::linalg::range(x2.data_handle(), 4, 8, handle.get_stream());
    raft::linalg::range(x3.data_handle(), 8, 12, handle.get_stream());
    raft::linalg::range(x4.data_handle(), 12, 16, handle.get_stream());
    raft::linalg::range(x5.data_handle(), 16, 20, handle.get_stream());
    raft::linalg::range(x6.data_handle(), 20, 24, handle.get_stream());
    raft::linalg::range(x7.data_handle(), 24, 28, handle.get_stream());
    raft::linalg::range(x8.data_handle(), 28, 32, handle.get_stream());
    raft::linalg::range(x9.data_handle(), 32, 36, handle.get_stream());
    std::vector<raft::device_matrix_view<math_t, idx_t, col_major>> xs = {x1.view(),
                                                                          x2.view(),
                                                                          x3.view(),
                                                                          x4.view(),
                                                                          x5.view(),
                                                                          x6.view(),
                                                                          x7.view(),
                                                                          x8.view(),
                                                                          x9.view()};
    raft::sparse::solver::detail::bmat(handle, total.view(), xs, 3);
    std::vector<math_t> res(total.size());
    std::vector<math_t> expected{0, 1, 12, 13, 24, 25, 2,  3,  14, 15, 26, 27,
                                 4, 5, 16, 17, 28, 29, 6,  7,  18, 19, 30, 31,
                                 8, 9, 20, 21, 32, 33, 10, 11, 22, 23, 34, 35};
    raft::copy(res.data(), total.data_handle(), total.size(), handle.get_stream());
    handle.sync_stream();
    ASSERT_TRUE(hostVecMatch(expected, res, raft::CompareApprox<math_t>(0.0001)));
  }

  void test_b_orthonormalize()
  {
    idx_t n_rows_v     = n_rows_a;
    idx_t n_features_v = params.n_components;
    raft::update_device(act_eigvecs.data(), params.init_eigvecs.data(), act_eigvecs.size(), stream);
    auto v = raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      act_eigvecs.data(), n_rows_v, n_features_v);
    auto bv =
      raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, n_rows_v, n_features_v);
    auto vbv =
      raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, n_features_v, n_features_v);
    b_orthonormalize(
      handle, v, bv.view(), std::nullopt, std::make_optional(vbv.view()), std::nullopt, true);
    std::vector<math_t> vbv_inv_expected{0.76298383, 0.0, -1.20276028, 1.0791533};
    std::vector<math_t> vbv_inv_actual(4);
    raft::copy(vbv_inv_actual.data(), vbv.data_handle(), vbv_inv_actual.size(), stream);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_TRUE(
      hostVecMatch(vbv_inv_expected, vbv_inv_actual, raft::CompareApprox<math_t>(0.0001)));
  }

  void test_eigh()
  {
    std::vector<math_t> in_cpu{1.73969722, 0.98719877, 0.73374337, 0.211756781};
    std::vector<math_t> lambda_cpu{-0.27255666,  2.22401026};
    std::vector<math_t> vector_cpu{-0.44044489,  0.89777965,  0.89777965,  0.44044489};
    auto in_gpu = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    auto lambda_gpu = raft::make_device_vector<math_t, idx_t>(handle, 2);
    auto vector_gpu = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, 2, 2);
    std::optional<raft::device_matrix_view<math_t, idx_t, raft::col_major>> empty_matrix_opt = std::nullopt;
    
    raft::copy(in_gpu.data_handle(), in_cpu.data(), 4, handle.get_stream());
    raft::sparse::solver::detail::eigh(handle, in_gpu.view(), empty_matrix_opt, vector_gpu.view(), lambda_gpu.view());

    ASSERT_TRUE(devArrMatchHost(lambda_cpu.data(), lambda_gpu.data_handle(), lambda_cpu.size(),  raft::CompareApprox<math_t>(0.0001), handle.get_stream()));
    ASSERT_TRUE(devArrMatchHost(vector_cpu.data(), vector_gpu.data_handle(), vector_cpu.size(),  raft::CompareApprox<math_t>(0.0001), handle.get_stream()));
  }

  void Run()
  {
    test_eigh();
    test_bmat();
    test_selectcolsif();
    test_b_orthonormalize();
    raft::update_device(ind_a.data(), params.matrix_a.row_ind.data(), params.matrix_a.row_ind.size(), stream);
    raft::update_device(ind_ptr_a.data(), params.matrix_a.row_ind_ptr.data(), params.matrix_a.row_ind_ptr.size(), stream);
    raft::update_device(values_a.data(), params.matrix_a.values.data(), params.matrix_a.values.size(), stream);

    raft::update_device(act_eigvecs.data(), params.init_eigvecs.data(), act_eigvecs.size(), stream);

    auto matA = raft::spectral::matrix::sparse_matrix_t(
      handle, ind_ptr_a.data(), ind_a.data(), values_a.data(), n_rows_a, n_rows_a, nnz_a);
    raft::sparse::solver::lobpcg(
      handle,
      matA,
      raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
        act_eigvecs.data(), n_rows_a, params.n_components),
      raft::make_device_vector_view<math_t, idx_t>(act_eigvals.data(), n_rows_a));

    std::vector<math_t> X_CPU(n_rows_a * params.n_components);
    std::vector<math_t> W_CPU(n_rows_a);
    raft::copy(X_CPU.data(), act_eigvecs.data(), X_CPU.size(), stream);
    raft::copy(W_CPU.data(), act_eigvals.data(), W_CPU.size(), stream);
    ASSERT_TRUE(raft::devArrMatch<math_t>(exp_eigvecs.data(),
                                          act_eigvecs.data(),
                                          exp_eigvecs.size(),
                                          raft::CompareApprox<math_t>(0.0001),
                                          stream));
    ASSERT_TRUE(raft::devArrMatch<math_t>(exp_eigvals.data(),
                                          act_eigvals.data(),
                                          exp_eigvals.size(),
                                          raft::CompareApprox<math_t>(0.0001),
                                          stream));
  }

 protected:
  raft::handle_t handle;
  cudaStream_t stream;

  LOBPCGInputs<math_t, idx_t> params;
  idx_t n_rows_a, nnz_a;
  rmm::device_uvector<idx_t> ind_a, ind_ptr_a;
  rmm::device_uvector<math_t> values_a, exp_eigvals, exp_eigvecs, act_eigvals, act_eigvecs;
};

using LOBPCGTestF = LOBPCGTest<float, int>;
TEST_P(LOBPCGTestF, Result) { Run(); }

using LOBPCGTestD = LOBPCGTest<double, int>;
TEST_P(LOBPCGTestD, Result) { Run(); }

const std::vector<LOBPCGInputs<float, int>> lobpcg_inputs_f = {
  {{{0, 4, 10, 14, 19, 24, 28},
    {0, 2, 3, 5, 0, 1, 2, 3, 4, 5, 0, 2, 3, 5, 1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 2, 3, 4},
    {0.37911922, 0.11567201, 0.5135106,  0.08968836, 0.73450965, 0.26432646, 0.21985123,
     0.74888277, 0.34753734, 0.11204864, 0.82902676, 0.53023521, 0.24047095, 0.37913592,
     0.60975031, 0.60746519, 0.96833343, 0.30845102, 0.88653955, 0.43530847, 0.32938903,
     0.82477561, 0.20858375, 0.24755519, 0.23677223, 0.73957246, 0.09050876, 0.86530489}},
   {0.08319983,
    0.17758466,
    0.93301819,
    0.67171826,
    0.19967821,
    0.30873092,
    0.35005079,
    0.56035486,
    0.64176631,
    0.93904784,
    0.38935935,
    0.97182089},
   {2.61153278, 0.85782948},
   {-0.38272064,
    -0.25160901,
    -0.48684676,
    -0.50752949,
    -0.43005954,
    -0.33265696,
    -0.39778489,
    0.2539629,
    -0.37506003,
    0.72637041,
    0.02727131,
    -0.32900198},
   2}};
const std::vector<LOBPCGInputs<double, int>> lobpcg_inputs_d = {
  {{{0, 4, 10, 14, 19, 24, 28},
    {0, 2, 3, 5, 0, 1, 2, 3, 4, 5, 0, 2, 3, 5, 1, 2, 3, 4, 5, 0, 2, 3, 4, 5, 0, 2, 3, 4},
    {0.37911922, 0.11567201, 0.5135106,  0.08968836, 0.73450965, 0.26432646, 0.21985123,
     0.74888277, 0.34753734, 0.11204864, 0.82902676, 0.53023521, 0.24047095, 0.37913592,
     0.60975031, 0.60746519, 0.96833343, 0.30845102, 0.88653955, 0.43530847, 0.32938903,
     0.82477561, 0.20858375, 0.24755519, 0.23677223, 0.73957246, 0.09050876, 0.86530489}},
   {0.08319983,
    0.17758466,
    0.93301819,
    0.67171826,
    0.19967821,
    0.30873092,
    0.35005079,
    0.56035486,
    0.64176631,
    0.93904784,
    0.38935935,
    0.97182089},
   {2.61153278, 0.85782948},
   {-0.38272064,
    -0.25160901,
    -0.48684676,
    -0.50752949,
    -0.43005954,
    -0.33265696,
    -0.39778489,
    0.2539629,
    -0.37506003,
    0.72637041,
    0.02727131,
    -0.32900198},
   2}};

INSTANTIATE_TEST_CASE_P(LOBPCGTest, LOBPCGTestF, ::testing::ValuesIn(lobpcg_inputs_f));
INSTANTIATE_TEST_CASE_P(LOBPCGTest, LOBPCGTestD, ::testing::ValuesIn(lobpcg_inputs_d));

}  // namespace sparse
}  // namespace raft

/*

a=cupyx.scipy.sparse.random(6,6, 0.8,'csr')
a.indptr = array([ 0,  4, 10, 14, 19, 24, 28], dtype=int32)

a.indices = array([0, 2, 3, 5, 0, 1, 2, 3, 4, 5, 0, 2, 3, 5, 1, 2, 3, 4, 5, 0, 2, 3,
       4, 5, 0, 2, 3, 4], dtype=int32)

a.data = array([0.37911922, 0.11567201, 0.5135106 , 0.08968836, 0.73450965,
       0.26432646, 0.21985123, 0.74888277, 0.34753734, 0.11204864,
       0.82902676, 0.53023521, 0.24047095, 0.37913592, 0.60975031,
       0.60746519, 0.96833343, 0.30845102, 0.88653955, 0.43530847,
       0.32938903, 0.82477561, 0.20858375, 0.24755519, 0.23677223,
       0.73957246, 0.09050876, 0.86530489])

a.todense() =
np.matrix([[0.37911922, 0.        , 0.11567201, 0.5135106 , 0.        , 0.08968836],
        [0.73450965, 0.26432646, 0.21985123, 0.74888277, 0.34753734, 0.11204864],
        [0.82902676, 0.        , 0.53023521, 0.24047095, 0.        , 0.37913592],
        [0.        , 0.60975031, 0.60746519, 0.96833343, 0.30845102, 0.88653955],
        [0.43530847, 0.        , 0.32938903, 0.82477561, 0.20858375, 0.24755519],
        [0.23677223, 0.        , 0.73957246, 0.09050876, 0.86530489, 0.        ]])
x = np.random.rand(6,2)
x = np.array([[0.08319983, 0.35005079],
           [0.17758466, 0.56035486],
           [0.93301819, 0.64176631],
           [0.67171826, 0.93904784],
           [0.19967821, 0.38935935],
           [0.30873092, 0.97182089]])

lobpcg(a, x) =  (array([2.61153278, 0.85782948]),
                array([[-0.38272064, -0.39778489],
                        [-0.25160901,  0.2539629 ],
                        [-0.48684676, -0.37506003],
                        [-0.50752949,  0.72637041],
                        [-0.43005954,  0.02727131],
                        [-0.33265696, -0.32900198]]))
 */