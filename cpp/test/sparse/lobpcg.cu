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

#include "../test_utils.h"
#include <raft/util/cudart_utils.hpp>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename math_t, typename idx_t>
struct CSRMatrixVal {
  std::vector<idx_t> row_ind;
  std::vector<idx_t> row_ind_ptr;
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

template <typename math_t, typename idx_t>
class LOBPCGTest : public ::testing::TestWithParam<LOBPCGInputs<math_t, idx_t>> {
 public:
  LOBPCGTest()
    : params(::testing::TestWithParam<LOBPCGInputs<math_t, idx_t>>::GetParam()),
      stream(handle.get_stream()),
      ind_a(params.matrix_a.row_ind.size(), stream),
      ind_ptr_a(params.matrix_a.row_ind_ptr.size(), stream),
      values_a(params.matrix_a.row_ind_ptr.size(), stream),
      exp_eigvals(params.exp_eigvals.size(), stream),
      exp_eigvecs(params.exp_eigvecs.size(), stream),
      act_eigvals(params.exp_eigvals.size(), stream),
      act_eigvecs(params.exp_eigvecs.size(), stream)
  {
  }

 protected:
  void SetUp() override
  {
    n_rows_a = params.matrix_a.row_ind.size() - 1;
    nnz_a    = params.matrix_a.row_ind_ptr.size();
  }

  void test_b_orthonormalize()
  {
    idx_t n_rows_v = n_rows_a;
    idx_t n_features_v = params.n_components;
    raft::update_device(act_eigvecs.data(), params.init_eigvecs.data(), act_eigvecs.size(), stream);
    auto v = raft::make_device_matrix_view<math_t, idx_t, raft::col_major>(
      act_eigvecs.data(), n_rows_v, n_features_v);
    auto bv = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, n_rows_v, n_features_v);
    auto vbv = raft::make_device_matrix<math_t, idx_t, raft::col_major>(handle, n_features_v, n_features_v);
    raft::sparse::solver::detail::b_orthonormalize(handle,
      v,
      bv.view(),
      std::nullopt,
      std::make_optional(vbv.view()),
      std::nullopt,
      true
    );
    std::vector<math_t> vbv_inv_expected{0.76298383, 0.0, -1.20276028, 1.0791533};
    std::vector<math_t> vbv_inv_actual(4);
    raft::copy(vbv_inv_actual.data(), vbv.data_handle(), vbv_inv_actual.size(), stream);

    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    ASSERT_TRUE(hostVecMatch(vbv_inv_expected, vbv_inv_actual, raft::CompareApprox<math_t>(0.0001)));
  }

  void Run()
  {
    test_b_orthonormalize();
    raft::update_device(ind_a.data(), params.matrix_a.row_ind.data(), n_rows_a, stream);
    raft::update_device(ind_ptr_a.data(), params.matrix_a.row_ind_ptr.data(), nnz_a, stream);
    raft::update_device(values_a.data(), params.matrix_a.values.data(), nnz_a, stream);

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
    ASSERT_TRUE(raft::devArrMatch<math_t>(
      exp_eigvecs.data(), act_eigvecs.data(), exp_eigvecs.size(), raft::CompareApprox<math_t>(0.0001), stream));
    ASSERT_TRUE(raft::devArrMatch<math_t>(
      exp_eigvals.data(), act_eigvals.data(), exp_eigvals.size(), raft::CompareApprox<math_t>(0.0001), stream));
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