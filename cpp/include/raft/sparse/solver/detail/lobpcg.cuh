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

#pragma once

#include <cmath>
#include <optional>

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <raft/linalg/eig.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/init.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/reverse.cuh>
#include <raft/matrix/slice.cuh>
#include <raft/matrix/triangular.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/spectral/matrix_wrappers.hpp>
#include <raft/util/cudart_utils.hpp>

template <typename value_t, typename index_t>
auto make_transpose_layout_view(raft::device_matrix_view<value_t, index_t, raft::row_major> mds)
{
  return raft::make_device_matrix_view<value_t, index_t, raft::col_major>(mds.data_handle(), mds.extent(1), mds.extent(0));
}
template <typename value_t, typename index_t>
auto make_transpose_layout_view(raft::device_matrix_view<value_t, index_t, raft::col_major> mds)
{
  return raft::make_device_matrix_view<value_t, index_t, raft::row_major>(mds.data_handle(), mds.extent(1), mds.extent(0));
}

namespace raft::sparse::solver::detail {

/**
 * @brief stucture that defines the reduction Lambda to find minimum between elements
 */
template <typename DataT>
struct MaxOp {
  HDI DataT operator()(DataT a, DataT b) { return maxPrim(a, b); }
};

template <typename DataT>
struct isnan_test {
  HDI int operator()(const DataT a) { return isnan(a); }
};

/**
 * @tparam value_t floating point type used for elements
 * @tparam index_t integer type used for indexing
 * Assemble a matrix from a list of blocks
 */
template <typename value_t, typename index_t>
void bmat(const raft::handle_t& handle,
          raft::device_matrix_view<value_t, index_t, col_major> out,
          const std::vector<raft::device_matrix_view<value_t, index_t, col_major>>& ins,
          index_t n_blocks)
{
  RAFT_EXPECTS(n_blocks * n_blocks == ins.size(), "inconsistent number of blocks");
  std::vector<index_t> cumulative_row(n_blocks);
  std::vector<index_t> cumulative_col(n_blocks);
  for (index_t i = 0; i < n_blocks; i++) {
    for (index_t j = 0; j < n_blocks; j++) {
      raft::matrix::slice_insert(
        handle,
        ins[j + i * n_blocks],
        out,
        raft::matrix::slice_coordinates(cumulative_row[j],
                                        cumulative_col[i],
                                        cumulative_row[j] + ins[j + i * n_blocks].extent(0),
                                        cumulative_col[i] + ins[j + i * n_blocks].extent(1)));
      cumulative_col[i] += ins[j + i * n_blocks].extent(0);
      cumulative_row[j] += ins[j + i * n_blocks].extent(1);
    }
  }
}

/* Modification of copyRows to reindex columns, col_major only
 * On a 4x3 matrix, indices could be [0, 2] to select col 0 and 2
 */
template <typename m_t, typename idx_array_t = int, typename idx_t = size_t>
void selectCols(const m_t* in,
                idx_t n_rows,
                idx_t n_cols,
                m_t* out,
                const idx_array_t* indices,
                idx_t n_cols_indices,
                cudaStream_t stream)
{
  idx_t size    = n_cols_indices * n_rows;
  auto counting = thrust::make_counting_iterator<idx_t>(0);

  thrust::for_each(rmm::exec_policy(stream), counting, counting + size, [=] __device__(idx_t idx) {
    idx_t row                   = idx % n_rows;
    idx_t new_col               = idx / n_rows;
    idx_t old_col               = indices[new_col];
    out[new_col * n_rows + row] = in[old_col * n_rows + row];
  });
}

template <typename value_t, typename index_t>
void selectColsIf(const raft::handle_t& handle,
                  raft::device_matrix_view<value_t, index_t, col_major> in,
                  raft::device_vector_view<index_t, index_t> mask,
                  raft::device_matrix_view<value_t, index_t, col_major> out)
{
  auto stream     = handle.get_stream();
  auto in_n_cols  = in.extent(1);
  auto out_n_cols = out.extent(1);
  auto rangeVec   = raft::make_device_vector<index_t, index_t>(handle, in_n_cols);
  raft::linalg::range(rangeVec.data_handle(), in_n_cols, stream);
  raft::linalg::map(
    handle,
    raft::make_const_mdspan(mask),
    raft::make_const_mdspan(rangeVec.view()),
    rangeVec.view(),
    [] __device__(index_t mask_value, index_t idx) { return mask_value == 1 ? idx : -1; });
  thrust::sort(rmm::exec_policy(stream),
               rangeVec.data_handle(),
               rangeVec.data_handle() + rangeVec.size(),
               thrust::less<index_t>());
  selectCols(in.data_handle(),
             in.extent(0),
             in.extent(1),
             out.data_handle(),
             rangeVec.data_handle() + rangeVec.size() - out_n_cols,
             out_n_cols,
             stream);
}

/**
 * Reverse if needed the eigenvalues/vectors and truncate the columns to fit eigVectorTrunc
 */
template <typename value_t, typename index_t>
void truncEig(
  const raft::handle_t& handle,
  raft::device_matrix_view<value_t, index_t, raft::col_major> eigVectorin,
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> eigVectorTrunc,
  raft::device_vector_view<value_t, index_t> eigLambda,
  bool largest)
{
  // The eigenvalues are already sorted in ascending order with syevd
  auto nrows = eigVectorin.extent(0);
  auto ncols = eigVectorin.extent(1);
  if (largest) {
    raft::matrix::col_reverse(handle, eigVectorin);
    raft::matrix::col_reverse(
      handle, raft::make_device_matrix_view(eigLambda.data_handle(), 1, eigLambda.extent(0)));
  }
  if (eigVectorTrunc.has_value() && ncols > eigVectorTrunc->extent(1))
    raft::matrix::truncZeroOrigin(eigVectorin.data_handle(),
                                  nrows,
                                  eigVectorTrunc->data_handle(),
                                  nrows,
                                  eigVectorTrunc->extent(1),
                                  handle.get_stream());
}

// C = A * B
template <typename value_t, typename index_t>
void spmm(const raft::handle_t& handle,
          raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A,
          raft::device_matrix_view<value_t, index_t, raft::col_major> B,
          raft::device_matrix_view<value_t, index_t, raft::col_major> C,
          bool transpose_a = false,
          bool transpose_b = false)
{
  auto stream          = handle.get_stream();
  auto* A_values_      = const_cast<value_t*>(A.values_);
  auto* A_row_offsets_ = const_cast<index_t*>(A.row_offsets_);
  auto* A_col_indices_ = const_cast<index_t*>(A.col_indices_);
  cusparseSpMatDescr_t sparse_A;
  cusparseDnMatDescr_t dense_B;
  cusparseDnMatDescr_t dense_C;
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatecsr(
    &sparse_A, A.nrows_, A.ncols_, A.nnz_, A_row_offsets_, A_col_indices_, A_values_));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
    &dense_B, B.extent(0), B.extent(1), B.extent(0), B.data_handle(), CUSPARSE_ORDER_COL));
  RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsecreatednmat(
    &dense_C, C.extent(0), C.extent(1), C.extent(0), C.data_handle(), CUSPARSE_ORDER_COL));
  // a * b
  value_t alpha    = 1;
  value_t beta     = 0;
  size_t buff_size = 0;
  auto opA         = transpose_a ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  auto opB         = transpose_b ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  raft::sparse::detail::cusparsespmm_bufferSize(handle.get_cusparse_handle(),
                                                opA,
                                                opB,
                                                &alpha,
                                                sparse_A,
                                                dense_B,
                                                &beta,
                                                dense_C,
                                                CUSPARSE_SPMM_ALG_DEFAULT,
                                                &buff_size,
                                                stream);
  rmm::device_uvector<value_t> dev_buffer(buff_size / sizeof(value_t), stream);
  raft::sparse::detail::cusparsespmm(handle.get_cusparse_handle(),
                                     opA,
                                     opB,
                                     &alpha,
                                     sparse_A,
                                     dense_B,
                                     &beta,
                                     dense_C,
                                     CUSPARSE_SPMM_ALG_DEFAULT,
                                     dev_buffer.data(),
                                     stream);

  cusparseDestroySpMat(sparse_A);
  cusparseDestroyDnMat(dense_B);
  cusparseDestroyDnMat(dense_C);
}

/**
 * Solve the linear equation A x = b, given the Cholesky factorization of A
 * The operation is in-place, i.e. matrix X overwrites matrix B.
 */
template <typename value_t, typename index_t>
void cho_solve(const raft::handle_t& handle,
               raft::device_matrix_view<const value_t, index_t, raft::col_major> A,
               raft::device_matrix_view<value_t, index_t, raft::col_major> B,
               bool lower = true)
{
  auto thrust_exec_policy = handle.get_thrust_policy();
  auto stream             = handle.get_stream();
  auto lda                = A.extent(0);
  auto dim                = A.extent(0);
  cublasFillMode_t uplo   = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  rmm::device_uvector<int> info(1, stream);
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrs(handle.get_cusolver_dn_handle(),
                                                          uplo,
                                                          dim,
                                                          B.extent(1),
                                                          A.data_handle(),
                                                          lda,
                                                          B.data_handle(),
                                                          dim,
                                                          info.data(),
                                                          stream));
}

template <typename value_t, typename index_t>
bool cholesky(const raft::handle_t& handle,
              raft::device_matrix_view<value_t, index_t, raft::col_major> P,
              bool lower = true)
{
  auto thrust_exec_policy = handle.get_thrust_policy();
  auto stream             = handle.get_stream();
  int Lwork               = 0;
  auto lda                = P.extent(0);
  auto dim                = P.extent(0);
  cublasFillMode_t uplo   = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

  auto P_copy =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, P.extent(0), P.extent(1));
  raft::copy(P_copy.data_handle(), P.data_handle(), P.size(), stream);

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf_bufferSize(
    handle.get_cusolver_dn_handle(), uplo, dim, P_copy.data_handle(), lda, &Lwork));

  rmm::device_uvector<value_t> workspace_decomp(Lwork / sizeof(value_t), stream);
  rmm::device_uvector<int> info(1, stream);
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf(handle.get_cusolver_dn_handle(),
                                                          uplo,
                                                          dim,
                                                          P_copy.data_handle(),
                                                          lda,
                                                          workspace_decomp.data(),
                                                          Lwork,
                                                          info.data(),
                                                          stream));
  int info_h = 0;
  raft::update_host(&info_h, info.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  ASSERT(info_h == 0, "lobpcg: error in potrf, info=%d | expected=0", info_h);

  int h_hasnan = thrust::transform_reduce(thrust_exec_policy,
                                          P_copy.data_handle(),
                                          P_copy.data_handle() + P_copy.size(),
                                          isnan_test<value_t>(),
                                          0,
                                          thrust::plus<int>());

  if (h_hasnan != 0)  // "lobpcg: error in cholesky, NaN in outputs"
    return false;

  raft::matrix::fill(handle, P, value_t(0));
  if (lower) {
    raft::matrix::lower_triangular(handle, raft::make_const_mdspan(P_copy.view()), P);
  } else {
    raft::matrix::upper_triangular(handle, raft::make_const_mdspan(P_copy.view()), P);
  }
  return true;
}

template <typename value_t, typename index_t>
void inverse(const raft::handle_t& handle,
             raft::device_matrix_view<value_t, index_t, raft::col_major> P,
             raft::device_matrix_view<value_t, index_t, raft::col_major> Pinv,
             bool lower = true)
{
  auto stream             = handle.get_stream();
  int Lwork               = 0;
  auto lda                = P.extent(0);
  auto dim                = P.extent(0);
  int info_h              = 0;
  cublasOperation_t trans = CUBLAS_OP_N;
  raft::matrix::eye(handle, Pinv);

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDngetrf_bufferSize(
    handle.get_cusolver_dn_handle(), dim, dim, P.data_handle(), lda, &Lwork));

  rmm::device_uvector<value_t> workspace_decomp(Lwork, stream);
  rmm::device_uvector<int> info(1, stream);
  auto ipiv = raft::make_device_vector<index_t, index_t>(handle, dim);

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDngetrf(handle.get_cusolver_dn_handle(),
                                                          dim,
                                                          dim,
                                                          P.data_handle(),
                                                          lda,
                                                          workspace_decomp.data(),
                                                          ipiv.data_handle(),
                                                          info.data(),
                                                          stream));

  raft::update_host(&info_h, info.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  ASSERT(info_h == 0, "lobpcg: error in getrf, info=%d | expected=0", info_h);

  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDngetrs(handle.get_cusolver_dn_handle(),
                                                          trans,
                                                          dim,
                                                          dim,
                                                          P.data_handle(),
                                                          lda,
                                                          ipiv.data_handle(),
                                                          Pinv.data_handle(),
                                                          lda,
                                                          info.data(),
                                                          stream));

  raft::update_host(&info_h, info.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  ASSERT(info_h == 0, "lobpcg: error in getrs, info=%d | expected=0", info_h);
}

template <typename value_t, typename index_t>
void apply_constraints(const raft::handle_t& handle,
                       raft::device_matrix_view<value_t, index_t, raft::col_major> V,
                       raft::device_matrix_view<value_t, index_t, raft::col_major> YBY,
                       raft::device_matrix_view<value_t, index_t, raft::col_major> BY,
                       raft::device_matrix_view<value_t, index_t, raft::col_major> Y)
{
  auto stream   = handle.get_stream();
  auto YBY_copy = raft::make_device_matrix<value_t, index_t, raft::col_major>(
    handle, YBY.extent(0), YBY.extent(1));
  raft::copy(YBY_copy.data_handle(), YBY.data_handle(), YBY.size(), stream);
  // TODO: Use mdspan gemm with row-major to transpose
  auto YBV =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, BY.extent(1), V.extent(1));
  value_t zero = 0;
  value_t one  = 1;
  raft::linalg::gemm(handle,
                     true,
                     false,
                     YBV.extent(0),
                     YBV.extent(1),
                     BY.extent(0),
                     &one,
                     BY.data_handle(),
                     BY.extent(0),
                     V.data_handle(),
                     V.extent(0),
                     &zero,
                     YBV.data_handle(),
                     YBV.extent(0),
                     stream);

  cholesky(handle, YBY_copy.view());
  cho_solve(handle, raft::make_const_mdspan(YBY_copy.view()), YBV.view());
  auto BV =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, V.extent(0), YBV.extent(1));
  raft::linalg::gemm(handle, Y, YBV.view(), BV.view());
  raft::linalg::subtract(handle, raft::make_const_mdspan(Y), raft::make_const_mdspan(BV.view()), Y);
}

/**
 * Helper function for converting a generalized eigenvalue problem
 * A(X) = lambda(B(X)) to standard eigen value problem using cholesky
 * transformation
 */
template <typename value_t, typename index_t>
bool eigh(const raft::handle_t& handle,
          raft::device_matrix_view<value_t, index_t, raft::col_major> A,
          std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> B_opt,
          raft::device_matrix_view<value_t, index_t, raft::col_major> eigVecs,
          raft::device_vector_view<value_t, index_t> eigVals)
{
  auto dim = A.extent(0);
  auto AT  = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  raft::linalg::transpose(handle, A, AT.view());
  if (!B_opt.has_value()) {
    raft::linalg::eig_dc(handle, raft::make_const_mdspan(AT.view()), eigVecs, eigVals);
    return true;
  }
  auto RTi         = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto Ri          = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto RT          = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto F           = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, dim, dim);
  auto B           = B_opt.value();
  bool cho_success = cholesky(handle, B, false);

  raft::linalg::transpose(handle, B, RT.view());
  inverse(handle, RT.view(), Ri.view());
  inverse(handle, B, RTi.view());

  // Reuse the memory of matrix
  auto& ARi   = B;
  auto& Fvecs = RT;
  raft::linalg::gemm(handle, AT.view(), Ri.view(), ARi);
  raft::linalg::gemm(handle, RTi.view(), ARi, F.view());

  raft::linalg::eig_dc(handle, raft::make_const_mdspan(F.view()), Fvecs.view(), eigVals);
  raft::linalg::gemm(handle, Ri.view(), Fvecs.view(), eigVecs);
  return cho_success;
}

/**
 * B-orthonormalize the given block vector using Cholesky
 *
 * @tparam value_t floating point type used for elements
 * @tparam index_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[inout] V: dense matrix to normalize
 * @param[inout] BV: dense matrix. Use with parameter `bv_is_empty`.
 * @param[in] B_opt: optional sparse matrix for normalization
 * @param[out] VBV_opt: optional dense matrix containing inverse matrix (shape v[1] * v[1])
 * @param[out] V_max_opt: optional vector containing normalization of V (shape v[1])
 * @param[in] bv_is_empty: True if BV is used as input
 * @return success status
 */
template <typename value_t, typename index_t>
bool b_orthonormalize(
  const raft::handle_t& handle,
  raft::device_matrix_view<value_t, index_t, raft::col_major> V,
  raft::device_matrix_view<value_t, index_t, raft::col_major> BV,
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B_opt     = std::nullopt,
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> VBV_opt = std::nullopt,
  std::optional<raft::device_vector_view<value_t, index_t>> V_max_opt                = std::nullopt,
  bool bv_is_empty                                                                   = true)
{
  auto stream        = handle.get_stream();
  auto V_max_buffer  = rmm::device_uvector<value_t>(0, stream);
  value_t* V_max_ptr = nullptr;
  if (!V_max_opt) {  // allocate normalization buffer
    V_max_buffer.resize(V.extent(1), stream);
    V_max_ptr = V_max_buffer.data();
  } else {
    V_max_ptr = V_max_opt.value().data_handle();
  }
  auto V_max       = raft::make_device_vector_view<value_t, index_t>(V_max_ptr, V.extent(1));
  auto V_max_const = raft::make_const_mdspan(V_max);

  //
  /*raft::linalg::reduce(handle,
                       raft::make_device_matrix_view<const value_t, index_t, raft::col_major>(
                         V.data_handle(), V.extent(1), V.extent(0)),
                       V_max,
                       value_t(0),
                       raft::linalg::Apply::ALONG_ROWS,
                       false,
                       raft::identity_op(),
                       MaxOp<value_t>());
  */
  raft::linalg::reduce(V_max.data_handle(),
                       V.data_handle(),
                       V.extent(0),
                       V.extent(1),
                       value_t(0),
                       false,
                       true,
                       handle.get_stream(),
                       false,
                       raft::identity_op(),
                       MaxOp<value_t>());
  raft::linalg::binary_div_skip_zero(handle, V, V_max_const, raft::linalg::Apply::ALONG_ROWS);

  if (!bv_is_empty) {
    raft::linalg::binary_div_skip_zero(handle, BV, V_max_const, raft::linalg::Apply::ALONG_ROWS);
  } else {
    if (B_opt)
      spmm(handle, B_opt.value(), V, BV);
    else
      raft::copy(BV.data_handle(), V.data_handle(), V.size(), stream);
  }
  auto VBV_buffer  = rmm::device_uvector<value_t>(0, stream);
  value_t* VBV_ptr = nullptr;
  if (!VBV_opt) {  // allocate normalization buffer
    VBV_buffer.resize(V.extent(1) * V.extent(1), stream);
    VBV_ptr = VBV_buffer.data();
  } else {
    VBV_ptr = VBV_opt.value().data_handle();
  }
  auto VBV = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
    VBV_ptr, V.extent(1), V.extent(1));
  auto VBVBuffer = raft::make_device_matrix<value_t, index_t, raft::col_major>(
    handle, VBV.extent(0), VBV.extent(1));
  auto VT =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, V.extent(1), V.extent(0));
  raft::linalg::transpose(handle, V, VT.view());

  raft::linalg::gemm(handle, VT.view(), BV, VBV);
  bool cholesky_success = cholesky(handle, VBV, false);
  if (!cholesky_success) { return cholesky_success; }

  inverse(handle, VBV, VBVBuffer.view());
  raft::copy(VBV.data_handle(), VBVBuffer.data_handle(), VBV.size(), stream);
  raft::linalg::gemm(handle, V, VBV, V);
  if (B_opt) raft::linalg::gemm(handle, BV, VBV, BV);
  return true;
}

template <typename value_t, typename index_t>
void lobpcg(
  const raft::handle_t& handle,
  // IN
  raft::spectral::matrix::sparse_matrix_t<index_t, value_t> A,    // shape=(n,n)
  raft::device_matrix_view<value_t, index_t, raft::col_major> X,  // shape=(n,k) IN OUT Eigvectors
  raft::device_vector_view<value_t, index_t> W,                   // shape=(k) OUT Eigvals
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B_opt,    // shape=(n,n)
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> M_opt,    // shape=(n,n)
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> Y_opt,  // Constraint
  // matrix shape=(n,Y)
  value_t tol           = 0,
  std::int32_t max_iter = 20,
  bool largest          = true,
  int verbosityLevel    = 0)
{
  cudaStream_t stream     = handle.get_stream();
  auto thrust_exec_policy = handle.get_thrust_policy();
  // auto size_y         = 0;
  // if (Y_opt.has_value()) size_y = Y_opt.value().extent(1);
  auto n      = X.extent(0);
  auto size_x = X.extent(1);

  /* TODO:  DENSE SOLUTION
  if ((n - size_y) < (5 * size_x)) {
    return;
  } */
  if (tol <= 0) { tol = raft::mySqrt(1e-15) * n; }
  // Apply constraints to X
  /*
  auto matrix_BY = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_y);
  if (Y_opt.has_value())
  {
      if (B_opt.has_value())
      {
          auto B = B_opt.value();
          spmm(handle, Y_opt.value(), B, matrix_BY.view(), false, false);
          // TODO
      } else {
          raft::copy(matrix_BY.data_handle(), Y_opt.value().data_handle(), n * size_y,
  handle.get_stream());
      }
      // GramYBY
      // ApplyConstraints
  }*/
  auto BX     = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
  auto BXView = BX.view();
  b_orthonormalize(handle, X, BXView, B_opt);
  // Compute the initial Ritz vectors: solve the eigenproblem.
  auto AX = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
  spmm(handle, A, X, AX.view());
  auto gramXAX =
    raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, size_x);
  auto XTRowView = make_transpose_layout_view(X);
  raft::linalg::gemm(handle, 
    XTRowView,
    AX.view(), gramXAX.view());
  auto eigVectorBuffer = rmm::device_uvector<value_t>(size_x * size_x, stream); // rmm because of resize
  auto eigVectorView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(eigVectorBuffer.data(), size_x, size_x);
  auto eigLambda = raft::make_device_vector<value_t, index_t>(handle, size_x);
  std::optional<raft::device_matrix_view<value_t, index_t, raft::col_major>> empty_matrix_opt = std::nullopt;
  eigh(handle, gramXAX.view(), empty_matrix_opt, eigVectorView, eigLambda.view());

  truncEig(handle, eigVectorView, empty_matrix_opt, eigLambda.view(), largest);
  // Slice not needed for first eigh
  // raft::matrix::slice(handle, eigVectorFull, eigVector, raft::matrix::slice_coordinates(0, 0,
  // eigVectorFull.extent(0), size_x));

  raft::linalg::gemm(handle, X, eigVectorView, X);
  raft::linalg::gemm(handle, AX.view(), eigVectorView, AX.view());
  if (B_opt) raft::linalg::gemm(handle, BXView, eigVectorView, BXView);

  // Active index set
  // TODO: use uint8_t
  auto active_mask       = raft::make_device_vector<index_t, index_t>(handle, size_x);
  auto previousBlockSize = size_x;

  auto ident  = rmm::device_uvector<value_t>(size_x * size_x, stream);
  auto identView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
    ident.data(), size_x, size_x);
  raft::matrix::eye(handle, identView);
  auto identSizeX = raft::make_device_matrix<value_t, index_t, raft::col_major>(
    handle, size_x, size_x);
  raft::matrix::eye(handle, identSizeX.view());

  auto Pbuffer  = rmm::device_uvector<value_t>(0, stream);
  auto APbuffer = rmm::device_uvector<value_t>(0, stream);
  auto BPbuffer = rmm::device_uvector<value_t>(0, stream);
  auto PView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(Pbuffer.data(), 0, 0);
  auto APView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(APbuffer.data(), 0, 0);
  auto BPView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(BPbuffer.data(), 0, 0);
  auto activePbuffer  = rmm::device_uvector<value_t>(0, stream);
  auto activeAPbuffer = rmm::device_uvector<value_t>(0, stream);
  auto activeBPbuffer = rmm::device_uvector<value_t>(0, stream);
  auto activePView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(activePbuffer.data(), 0, 0);
  auto activeAPView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(activeAPbuffer.data(), 0, 0);
  auto activeBPView =
    raft::make_device_matrix_view<value_t, index_t, raft::col_major>(activeBPbuffer.data(), 0, 0);
  auto R = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);

  auto aux = raft::make_device_matrix<value_t, index_t, raft::col_major>(
    handle, n, size_x);
  //auto aux_sum = raft::make_device_vector<value_t, index_t>(handle, size_x);
  auto residual_norms = raft::make_device_vector<value_t, index_t>(handle, size_x);
  std::int32_t iteration_number = -1;
  bool restart                  = true;
  bool explicitGramFlag         = false;
  while (iteration_number < max_iter + 1) {
    iteration_number += 1;
    if (B_opt) {
      raft::matrix::copy(handle, raft::make_const_mdspan(BXView), aux.view());
    } else {
      raft::matrix::copy(handle, raft::make_const_mdspan(X), aux.view());
    }
    raft::linalg::binary_mult_skip_zero(handle,
                                        aux.view(),
                                        raft::make_const_mdspan(eigLambda.view()),
                                        raft::linalg::Apply::ALONG_ROWS);

    raft::linalg::subtract(
      handle, raft::make_const_mdspan(AX.view()), raft::make_const_mdspan(aux.view()), R.view());

    raft::linalg::reduce(
      residual_norms.data_handle(),
      R.data_handle(),
      size_x,
      n,
      value_t(0),
      false,
      true,
      stream,
      false,
      raft::sq_op());

    // TODO check sqop of reduce raft::linalg::sqrt(handle, raft::make_const_mdspan(aux_sum.view()), residual_norms.view());

    // cupy where & active_mask
    raft::linalg::unary_op(handle,
                           raft::make_const_mdspan(residual_norms.view()),
                           active_mask.view(),
                           [tol] __device__(value_t rn) { return rn > tol; });
    if (verbosityLevel > 2) {
      print_device_vector("active_mask", active_mask.data_handle(), active_mask.size(), std::cout);
    }
    index_t currentBlockSize = thrust::count_if(thrust::cuda::par.on(stream),
                                                active_mask.data_handle(),
                                                active_mask.data_handle() + active_mask.size(),
                                                [] __device__(value_t v) {return v > 0; });
    handle.sync_stream();
    if (currentBlockSize != previousBlockSize) {
      previousBlockSize = currentBlockSize;
      ident.resize(currentBlockSize * currentBlockSize, stream);
      identView         = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(
        ident.data(), currentBlockSize, currentBlockSize);
      raft::matrix::eye(handle, identView);
    }

    if (currentBlockSize == 0) break;
    if (verbosityLevel > 0) {
      // TODO add verb
    }
    auto activeR =
      raft::make_device_matrix<value_t, index_t, col_major>(handle, n, currentBlockSize);

    selectColsIf(handle, R.view(), active_mask.view(), activeR.view());

    if (iteration_number > 0) {
      activePbuffer.resize(n * currentBlockSize, stream);
      activeAPbuffer.resize(n * currentBlockSize, stream);
      activeBPbuffer.resize(n * currentBlockSize, stream);
      activePView = raft::make_device_matrix_view<value_t, index_t, col_major>(activePbuffer.data(), n, currentBlockSize);
      activeAPView = raft::make_device_matrix_view<value_t, index_t, col_major>(activeAPbuffer.data(), n, currentBlockSize);
      selectColsIf(handle, PView, active_mask.view(), activePView);
      selectColsIf(handle, APView, active_mask.view(), activeAPView);
      if (B_opt.has_value()) {
        activeBPView = raft::make_device_matrix_view<value_t, index_t, col_major>(activeBPbuffer.data(), n, currentBlockSize);
        selectColsIf(handle, BPView, active_mask.view(), activeBPView);
      }
    }
    if (M_opt.has_value()) {
      // Apply preconditioner T to the active residuals.
      auto MRtemp = raft::make_device_matrix<value_t, index_t, col_major>(
        handle, R.extent(0), currentBlockSize);
      spmm(handle, M_opt.value(), activeR.view(), MRtemp.view());
      raft::copy(activeR.data_handle(), MRtemp.data_handle(), MRtemp.size(), stream);
    }
    // Apply constraints to the preconditioned residuals.
    if (Y_opt.has_value()) {
      // TODO Constraint
      // apply_constraints(handle, X, gramYBY.view(), BY.view(), Y_opt.value());
    }
    // B-orthogonalize the preconditioned residuals to X.
    if (B_opt.has_value()) {
      auto BXTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, BX.extent(1), activeR.extent(1));
      auto XBXTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, X.extent(0), BXTR.extent(1));

      raft::linalg::gemm(handle, 
        make_transpose_layout_view(BX.view()),
        activeR.view(), BXTR.view());
      raft::linalg::gemm(handle, X, BXTR.view(), XBXTR.view());
      raft::linalg::subtract(handle,
                             raft::make_const_mdspan(activeR.view()),
                             raft::make_const_mdspan(XBXTR.view()),
                             activeR.view());
    } else {
      auto XTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, X.extent(1), activeR.extent(1));
      auto XXTR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, X.extent(0), XTR.extent(1));
      raft::linalg::gemm(handle, XTRowView, activeR.view(), XTR.view());
      raft::linalg::gemm(handle, X, XTR.view(), XXTR.view());
      raft::linalg::subtract(handle,
                             raft::make_const_mdspan(activeR.view()),
                             raft::make_const_mdspan(XXTR.view()),
                             activeR.view());
    }
    // B-orthonormalize the preconditioned residuals.
    auto activeBR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
      handle, activeR.extent(0), activeR.extent(1));
    auto activeBRView = activeBR.view();
    b_orthonormalize(handle, activeR.view(), activeBRView, B_opt);

    auto activeAR =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, activeR.extent(1));
    spmm(handle, A, activeR.view(), activeAR.view());

    if (iteration_number > 0) {
      auto invR = raft::make_device_matrix<value_t, index_t, raft::col_major>(
        handle, activePView.extent(1), activePView.extent(1));
      auto normal = raft::make_device_vector<value_t, index_t>(handle, activePView.extent(1));
      bool b_orth_success = true;
      if (B_opt.has_value()) {
        auto BP = raft::make_device_matrix<value_t, index_t, raft::col_major>(
          handle, activePView.extent(0), activePView.extent(1));
        b_orth_success = b_orthonormalize(handle,
                                          activePView,
                                          BP.view(),
                                          B_opt,
                                          std::make_optional(invR.view()),
                                          std::make_optional(normal.view()));
      } else {
        b_orth_success = b_orthonormalize(handle,
                                          activePView,
                                          activeBPView,
                                          B_opt,
                                          std::make_optional(invR.view()),
                                          std::make_optional(normal.view()),
                                          false);
      }
      if (!b_orth_success) {
        restart = true;
      } else {
        raft::linalg::binary_div_skip_zero(handle,
                                           activeAPView,
                                           raft::make_const_mdspan(normal.view()),
                                           raft::linalg::Apply::ALONG_ROWS);
        raft::linalg::gemm(handle, activeAPView, invR.view(), activeAPView);
        restart = false;
      }

      // Perform the Rayleigh Ritz Procedure:
      // Compute symmetric Gram matrices:
      value_t myeps = 1;  // TODO: std::is_same_t<value_t, float> ? 1e-4 : 1e-8;
      if (!explicitGramFlag) {
        value_t* residual_norms_max_elem =
          thrust::max_element(thrust_exec_policy,
                              residual_norms.data_handle(),
                              residual_norms.data_handle() + residual_norms.size());
        value_t residual_norms_max = 0;
        raft::copy(&residual_norms_max, residual_norms_max_elem, 1, stream);
        explicitGramFlag = residual_norms_max <= myeps;
      }

      if (!B_opt.has_value()) {
        // Shared memory assignments to simplify the code
        BXView = X;
        activeBRView = activeR.view();
        if (!restart)
          activeBPView = activePView;
      }
    }
    // Common submatrices
    auto gramXAR =
      raft::make_device_matrix<value_t, index_t, col_major>(handle, size_x, currentBlockSize);
    auto gramRAR = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramXBX =
      raft::make_device_matrix<value_t, index_t, col_major>(handle, size_x, currentBlockSize);
    auto gramRBR = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramXBR =
      raft::make_device_matrix<value_t, index_t, col_major>(handle, size_x, currentBlockSize);
    raft::linalg::gemm(handle,
                       XTRowView,
                       activeAR.view(),
                       gramXAR.view());

    raft::linalg::gemm(
      handle,
      make_transpose_layout_view(activeR.view()),
      activeAR.view(),
      gramRAR.view());

    auto device_half = raft::make_device_scalar<value_t>(handle, 0.5);
    if (explicitGramFlag) {
      raft::linalg::gemm(
        handle,
        make_transpose_layout_view(gramRAR.view()),
        identView,
        gramRAR.view(),
        std::make_optional(device_half.view()),
        std::make_optional(device_half.view()));
      raft::linalg::gemm(handle,
                         XTRowView,
                         AX.view(),
                         gramXAX.view());
      raft::linalg::gemm(
        handle,
        make_transpose_layout_view(gramXAX.view()),
        identView,
        gramXAX.view(),
        std::make_optional(device_half.view()),
        std::make_optional(device_half.view()));

      raft::linalg::gemm(handle,
                         XTRowView,
                         BX.view(),
                         gramXBX.view());
      raft::linalg::gemm(
        handle,
        make_transpose_layout_view(activeR.view()),
        activeBRView,
        gramRBR.view());
      raft::linalg::gemm(handle,
                         XTRowView,
                         activeBRView,
                         gramXBR.view());
    } else {
      raft::matrix::fill(handle, gramXAX.view(), value_t(0));
      raft::matrix::set_diagonal(handle, make_const_mdspan(eigLambda.view()), gramXAX.view());

      raft::matrix::eye(handle, gramXBX.view());
      raft::matrix::eye(handle, gramRBR.view());
      raft::matrix::fill(handle, gramXBR.view(), value_t(0));
    }
    auto gramDim = gramXAX.extent(1) + gramXAR.extent(1) + currentBlockSize;
    auto gramA   = raft::make_device_matrix<value_t, index_t, col_major>(handle, gramDim, gramDim);
    auto gramB   = raft::make_device_matrix<value_t, index_t, col_major>(handle, gramDim, gramDim);
    auto gramAView     = gramA.view();
    auto gramBView     = gramB.view();
    auto eigLambdaTemp = raft::make_device_vector<value_t, index_t>(handle, gramDim);
    auto eigVectorTemp =
      raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, gramDim, gramDim);
    auto eigLambdaTempView = eigLambdaTemp.view();
    auto eigVectorTempView = eigVectorTemp.view();
    eigVectorBuffer.resize(gramDim * size_x, stream);
    eigVectorView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(eigVectorBuffer.data(), gramDim, size_x);
    auto gramXAP =
      raft::make_device_matrix<value_t, index_t, col_major>(handle, size_x, currentBlockSize);
    auto gramRAP = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramPAP = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramXBP =
      raft::make_device_matrix<value_t, index_t, col_major>(handle, size_x, currentBlockSize);
    auto gramRBP = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, currentBlockSize, currentBlockSize);
    auto gramPBP = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, currentBlockSize, currentBlockSize);
    // create transpose mat
    auto gramXAPT = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, gramXAP.extent(1), gramXAP.extent(0));
    auto gramXART = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, gramXAR.extent(1), gramXAR.extent(0));
    auto gramRAPT = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, gramRAP.extent(1), gramRAP.extent(0));
    auto gramXBPT = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, gramXBP.extent(1), gramXBP.extent(0));
    auto gramXBRT = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, gramXBR.extent(1), gramXBR.extent(0));
    auto gramRBPT = raft::make_device_matrix<value_t, index_t, col_major>(
      handle, gramRBP.extent(1), gramRBP.extent(0));
    raft::linalg::transpose(handle, gramXAR.view(), gramXART.view());
    raft::linalg::transpose(handle, gramXBR.view(), gramXBRT.view());

    if (!restart) {
      raft::linalg::gemm(handle,
                         XTRowView,
                         activeAPView,
                         gramXAP.view());
      raft::linalg::gemm(
        handle,
        make_transpose_layout_view(activeR.view()),
        activeAPView,
        gramRAP.view());
      raft::linalg::gemm(handle,
                         make_transpose_layout_view(activePView),
                         activeAPView,
                         gramPAP.view());
      raft::linalg::gemm(handle,
                         XTRowView,
                         activeBPView,
                         gramXBP.view());
      raft::linalg::gemm(
        handle,
        make_transpose_layout_view(activeR.view()),
        activeBPView,
        gramRBP.view());

      if (explicitGramFlag) {
        raft::linalg::gemm(
          handle,
          make_transpose_layout_view(gramPAP.view()),
          identView,
          gramPAP.view(),
          std::make_optional(device_half.view()),
          std::make_optional(device_half.view()));
        raft::linalg::gemm(handle,
                           make_transpose_layout_view(activePView),
                           activeBPView,
                           gramPBP.view());
      } else {
        raft::matrix::eye(handle, gramPBP.view());
      }
      raft::linalg::transpose(handle, gramXAP.view(), gramXAPT.view());
      raft::linalg::transpose(handle, gramRAP.view(), gramRAPT.view());
      raft::linalg::transpose(handle, gramXBP.view(), gramXBPT.view());
      raft::linalg::transpose(handle, gramRBP.view(), gramRBPT.view());

      std::vector<raft::device_matrix_view<value_t, index_t, col_major>> A_blocks = {
        gramXAX.view(), gramXAR.view(), gramXAP.view(), gramXART.view(), gramRAR.view(), gramRAP.view(), gramXAPT.view(), gramRAPT.view(), gramPAP.view()};
      std::vector<raft::device_matrix_view<value_t, index_t, col_major>> B_blocks = {
        gramXBX.view(), gramXBR.view(), gramXBP.view(), gramXBRT.view(), gramRBR.view(), gramRBP.view(), gramXBPT.view(), gramRBPT.view(), gramPBP.view()};
      gramAView =
        raft::make_device_matrix_view<value_t, index_t, col_major>(gramA.data_handle(), n, n);
      gramBView =
        raft::make_device_matrix_view<value_t, index_t, col_major>(gramB.data_handle(), n, n);

      bmat(handle, gramAView, A_blocks, 3);
      bmat(handle, gramBView, B_blocks, 3);

      bool eig_sucess =
        eigh(handle, gramAView, std::make_optional(gramBView), eigVectorTempView, eigLambdaTempView);
      if (!eig_sucess) restart = true;
    }
    if (restart) {
      gramDim = gramXAX.extent(1) + gramXAR.extent(1);
      std::vector<raft::device_matrix_view<value_t, index_t, col_major>> A_blocks = {
        gramXAX.view(), gramXAR.view(), gramXART.view(), gramRAR.view()};
      std::vector<raft::device_matrix_view<value_t, index_t, col_major>> B_blocks = {
        gramXBX.view(), gramXBR.view(), gramXBRT.view(), gramRBR.view()};
      gramAView = raft::make_device_matrix_view<value_t, index_t, col_major>(
        gramA.data_handle(), gramDim, gramDim);
      gramBView = raft::make_device_matrix_view<value_t, index_t, col_major>(
        gramB.data_handle(), gramDim, gramDim);
      eigLambdaTempView =
        raft::make_device_vector_view<value_t, index_t>(eigLambdaTempView.data_handle(), gramDim);
      eigVectorTempView = raft::make_device_matrix_view<value_t, index_t, col_major>(
        eigVectorTempView.data_handle(), gramDim, gramDim);
      bmat(handle, gramAView, A_blocks, 2);
      bmat(handle, gramBView, B_blocks, 2);
      bool eig_sucess = eigh(
        handle, gramAView, std::make_optional(gramBView), eigVectorTempView, eigLambdaTempView);
      ASSERT(eig_sucess, "lobpcg: eigh has failed in lobpcg iterations");
    }
    truncEig(
      handle, eigVectorTempView, std::make_optional(eigVectorView), eigLambdaTempView, largest);
    raft::copy(eigLambda.data_handle(), eigLambdaTempView.data_handle(), size_x, stream);

    // Verbosity print

    // Compute Ritz vectors.
    auto d_one = raft::make_device_scalar<value_t>(handle, 1);
    auto one = std::make_optional(d_one.view());
    auto eigBlockVectorX = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, size_x, size_x);
    auto eigBlockVectorR = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, currentBlockSize, size_x);
    auto eigBlockVectorP = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, gramDim - (size_x + currentBlockSize), size_x);
    auto pp = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
    auto app = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
    if (B_opt.has_value()) {
      auto bpp = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
      raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorX.view(),
        raft::matrix::slice_coordinates<index_t>(0, 0, size_x, size_x));
      if (!restart) {
        raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorR.view(),
          raft::matrix::slice_coordinates<index_t>(size_x, 0, size_x + currentBlockSize, size_x));
        raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorP.view(),
          raft::matrix::slice_coordinates<index_t>(size_x + currentBlockSize, 0, gramDim, size_x));
      } else {
        raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorR.view(),
          raft::matrix::slice_coordinates<index_t>(size_x, 0, gramDim, size_x));
      }

      raft::linalg::gemm(handle, activeR.view(), eigBlockVectorR.view(), pp.view());
      raft::linalg::gemm(handle, activeAR.view(), eigBlockVectorR.view(), app.view());
      raft::linalg::gemm(handle, activeBRView, eigBlockVectorR.view(), bpp.view());
      if (!restart) {
        raft::linalg::gemm(handle, activePView, eigBlockVectorP.view(), pp.view(), one, one);
        raft::linalg::gemm(handle, activeAPView, eigBlockVectorP.view(), app.view(), one, one);
        raft::linalg::gemm(handle, activeBPView, eigBlockVectorP.view(), bpp.view(), one, one);
      }
      Pbuffer.resize(n * size_x, stream);
      APbuffer.resize(n * size_x, stream);
      BPbuffer.resize(n * size_x, stream);
      PView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(Pbuffer.data(), n, size_x);
      APView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(APbuffer.data(), n, size_x);
      BPView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(BPbuffer.data(), n, size_x);
  
      raft::copy(PView.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(APView.data_handle(), app.data_handle(), app.size(), stream);
      raft::copy(BPView.data_handle(), bpp.data_handle(), bpp.size(), stream);
  
      raft::linalg::gemm(handle, X, eigBlockVectorX.view(), pp.view(), one, one);
      raft::linalg::gemm(handle, AX.view(), eigBlockVectorX.view(), app.view(), one, one);
      raft::linalg::gemm(handle, BXView, eigBlockVectorX.view(), bpp.view(), one, one);
  
      raft::copy(X.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(AX.data_handle(), app.data_handle(), app.size(), stream);
      raft::copy(BXView.data_handle(), bpp.data_handle(), bpp.size(), stream);
    } else {
      raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorX.view(),
        raft::matrix::slice_coordinates<index_t>(0, 0, size_x, size_x));
      if (!restart) {
        raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorR.view(),
          raft::matrix::slice_coordinates<index_t>(size_x, 0, size_x + currentBlockSize, size_x));
        raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorP.view(),
          raft::matrix::slice_coordinates<index_t>(size_x + currentBlockSize, 0, gramDim, size_x));
      } else {
        raft::matrix::slice(handle, make_const_mdspan(eigVectorView), eigBlockVectorR.view(),
          raft::matrix::slice_coordinates<index_t>(size_x, 0, gramDim, size_x));
      }

      raft::linalg::gemm(handle, activeR.view(), eigBlockVectorR.view(), pp.view());
      raft::linalg::gemm(handle, activeAR.view(), eigBlockVectorR.view(), app.view());
      if (!restart) {
        raft::linalg::gemm(handle, activePView, eigBlockVectorP.view(), pp.view(), one, one);
        raft::linalg::gemm(handle, activeAPView, eigBlockVectorP.view(), app.view(), one, one);
      }
      Pbuffer.resize(n * size_x, stream);
      APbuffer.resize(n * size_x, stream);
      PView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(Pbuffer.data(), n, size_x);
      APView = raft::make_device_matrix_view<value_t, index_t, raft::col_major>(APbuffer.data(), n, size_x);
  
      raft::copy(PView.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(APView.data_handle(), app.data_handle(), app.size(), stream);
  
      raft::linalg::gemm(handle, X, eigBlockVectorX.view(), pp.view(), one, one);
      raft::linalg::gemm(handle, AX.view(), eigBlockVectorX.view(), app.view(), one, one);
  
      raft::copy(X.data_handle(), pp.data_handle(), pp.size(), stream);
      raft::copy(AX.data_handle(), app.data_handle(), app.size(), stream);
    }
  }
  
  if (B_opt.has_value()) {
    raft::copy(aux.data_handle(), BXView.data_handle(), BXView.size(), stream);
  } else {
    raft::copy(aux.data_handle(), X.data_handle(), X.size(), stream);
  }
  raft::linalg::binary_mult_skip_zero(handle, aux.view(), make_const_mdspan(eigLambda.view()), raft::linalg::Apply::ALONG_ROWS);

  raft::linalg::subtract(
    handle, raft::make_const_mdspan(AX.view()), raft::make_const_mdspan(aux.view()), R.view());

  raft::linalg::reduce(
    residual_norms.data_handle(),
    R.data_handle(),
    size_x,
    n,
    value_t(0),
    false,
    true,
    stream,
    false,
    raft::sq_op());
  // TODO check reduce sqrt postop raft::linalg::sqrt(handle, raft::make_const_mdspan(aux_sum.view()), residual_norms.view());

  if (verbosityLevel > 0) {
    /// TODO add verb
  }
}
};  // namespace raft::sparse::solver::detail