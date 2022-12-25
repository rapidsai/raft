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

#include <raft/core/device_mdarray.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/detail/cusolver_wrappers.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/reduce.cuh>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/diagonal.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/triangular.cuh>
#include <raft/sparse/detail/cusparse_wrappers.h>
#include <raft/spectral/matrix_wrappers.hpp>

namespace raft::sparse::solver::detail {

/**
 * @brief stucture that defines the reduction Lambda to find minimum between elements
 */
template <typename DataT>
struct MaxOp {
  HDI DataT operator()(DataT a, DataT b) { return maxPrim(a, b); }
};

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

template <typename value_t, typename index_t>
void cholesky(const raft::handle_t& handle,
              raft::device_matrix_view<value_t, index_t, raft::col_major> P,
              bool lower = true)
{
  auto stream           = handle.get_stream();
  int Lwork             = 0;
  auto lda              = P.extent(0);
  auto dim              = P.extent(0);
  cublasFillMode_t uplo = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf_bufferSize(
    handle.get_cusolver_dn_handle(), uplo, dim, P.data_handle(), lda, &Lwork));

  rmm::device_uvector<value_t> workspace_decomp(Lwork / sizeof(value_t), stream);
  rmm::device_uvector<int> info(1, stream);
  RAFT_CUSOLVER_TRY(raft::linalg::detail::cusolverDnpotrf(handle.get_cusolver_dn_handle(),
                                                          uplo,
                                                          dim,
                                                          P.data_handle(),
                                                          lda,
                                                          workspace_decomp.data(),
                                                          Lwork,
                                                          info.data(),
                                                          stream));
  int info_h = 0;
  raft::update_host(&info_h, info.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  ASSERT(info_h == 0, "lobpcg: error in potrf, info=%d | expected=0", info_h);
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
  // make Pinv an identity matrix
  auto diag = raft::make_device_vector<value_t, index_t>(handle, dim);
  raft::matrix::fill(handle, diag.view(), value_t(1));
  raft::matrix::fill(handle, Pinv, value_t(0));
  raft::matrix::set_diagonal(
    handle, raft::make_device_vector_view<const value_t, index_t>(diag.data_handle(), dim), Pinv);

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

/**
 * B-orthonormalize the given block vector using Cholesky
 *
 * @tparam value_t floating point type used for elements
 * @tparam index_t integer type used for indexing
 * @param[in] handle: raft handle
 * @param[in] B_opt: optional sparse matrix for normalization
 * @param[inout] V: dense matrix to normalize
 * @param[inout] BV: dense matrix. Use with parameter `bv_is_empty`.
 * @param[out] VBV_opt: optional dense matrix containing inverse matrix
 * @param[out] V_max_opt: optional vector containing normalization of V
 * @param[in] bv_is_empty: True if BV is used as input
 */
template <typename value_t, typename index_t>
void b_orthonormalize(
  const raft::handle_t& handle,
  std::optional<raft::spectral::matrix::sparse_matrix_t<index_t, value_t>> B_opt,
  raft::device_matrix_view<value_t, index_t, raft::col_major> V,
  raft::device_matrix_view<value_t, index_t, raft::col_major> BV,
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
  auto V_max_const = raft::make_device_vector_view<const value_t, index_t>(V_max_ptr, V.extent(1));

  //
  /*raft::linalg::reduce(handle,
                       raft::make_device_matrix_view<const value_t, index_t, raft::col_major>(
                         V.data_handle(), V.extent(1), V.extent(0)),
                       V_max,
                       value_t(0),
                       raft::linalg::Apply::ALONG_ROWS,
                       false,
                       raft::Nop<value_t>(),
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
                       raft::Nop<value_t>(),
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

  raft::linalg::gemm(handle, VT.view(), BV, VBVBuffer.view());
  cholesky(handle, VBVBuffer.view(), false);
  // Reset VBV before copying upper triangular
  raft::matrix::fill(handle, VBV, value_t(0));
  raft::matrix::upper_triangular(
    handle,
    raft::make_device_matrix_view<const value_t, index_t, raft::col_major>(
      VBVBuffer.data_handle(), VBVBuffer.extent(0), VBV.extent(1)),
    VBV);

  inverse(handle, VBV, VBVBuffer.view());
  raft::copy(VBV.data_handle(), VBVBuffer.data_handle(), VBV.size(), stream);
  raft::linalg::gemm(handle, V, VBV, V);
  if (B_opt) raft::linalg::gemm(handle, BV, VBV, BV);
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
  value_t tol            = 0,
  std::uint32_t max_iter = 20,
  bool largest           = true)
{
  cudaStream_t stream = handle.get_stream();
  // auto size_y         = 0;
  // if (Y_opt.has_value()) size_y = Y_opt.value().extent(1);
  auto n      = X.extent(0);
  auto size_x = X.extent(1);

  /*  DENSE SOLUTION
  if ((n - size_y) < (5 * size_x)) {
    return;
  } */
  if (tol == 0) { tol = raft::mySqrt(1e-15) * n; }
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
      cusparseDestroyDnMat(denseY);
      // GramYBY
      // ApplyConstraints
  }*/
  auto BX = raft::make_device_matrix<value_t, index_t, raft::col_major>(handle, n, size_x);
  b_orthonormalize(handle, B_opt, X, BX.view());
  return;
  // TODO
}
};  // namespace raft::sparse::solver::detail