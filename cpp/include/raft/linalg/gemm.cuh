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
#ifndef __GEMM_H
#define __GEMM_H

#pragma once

#include "detail/gemm.hpp"

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/util/input_validation.hpp>

namespace raft {
namespace linalg {

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: C = alpha .* opA(A) * opB(B) + beta .* C
 *
 * @tparam math_t the element type
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @param [in] handle raft handle
 * @param [in] trans_a cublas transpose op for A
 * @param [in] trans_b cublas transpose op for B
 * @param [in] m number of rows of C
 * @param [in] n number of columns of C
 * @param [in] k number of rows of opB(B) / number of columns of opA(A)
 * @param [in] alpha host or device scalar
 * @param [in] A such a matrix that the shape of column-major opA(A) is [m, k]
 * @param [in] lda leading dimension of A
 * @param [in] B such a matrix that the shape of column-major opA(B) is [k, n]
 * @param [in] ldb leading dimension of B
 * @param [in] beta host or device scalar
 * @param [inout] C column-major matrix of size [m, n]
 * @param [in] ldc leading dimension of C
 * @param [in] stream
 */
template <typename math_t, bool DevicePointerMode = false>
void gemm(raft::resources const& handle,
          const bool trans_a,
          const bool trans_b,
          const int m,
          const int n,
          const int k,
          const math_t* alpha,
          const math_t* A,
          const int lda,
          const math_t* B,
          const int ldb,
          const math_t* beta,
          math_t* C,
          const int ldc,
          cudaStream_t stream)
{
  detail::gemm<math_t, DevicePointerMode>(
    handle, trans_a, trans_b, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, stream);
}

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam math_t the type of input/output matrices
 * @param handle raft handle
 * @param a input matrix
 * @param n_rows_a number of rows of A
 * @param n_cols_a number of columns of A
 * @param b input matrix
 * @param c output matrix
 * @param n_rows_c number of rows of C
 * @param n_cols_c number of columns of C
 * @param trans_a cublas transpose op for A
 * @param trans_b cublas transpose op for B
 * @param alpha scalar
 * @param beta scalar
 * @param stream cuda stream
 */
template <typename math_t>
void gemm(raft::resources const& handle,
          const math_t* a,
          int n_rows_a,
          int n_cols_a,
          const math_t* b,
          math_t* c,
          int n_rows_c,
          int n_cols_c,
          cublasOperation_t trans_a,
          cublasOperation_t trans_b,
          math_t alpha,
          math_t beta,
          cudaStream_t stream)
{
  detail::gemm(
    handle, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, alpha, beta, stream);
}

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam math_t the type of input/output matrices
 * @param handle raft handle
 * @param a input matrix
 * @param n_rows_a number of rows of A
 * @param n_cols_a number of columns of A
 * @param b input matrix
 * @param c output matrix
 * @param n_rows_c number of rows of C
 * @param n_cols_c number of columns of C
 * @param trans_a cublas transpose op for A
 * @param trans_b cublas transpose op for B
 * @param stream cuda stream
 */
template <typename math_t>
void gemm(raft::resources const& handle,
          const math_t* a,
          int n_rows_a,
          int n_cols_a,
          const math_t* b,
          math_t* c,
          int n_rows_c,
          int n_cols_c,
          cublasOperation_t trans_a,
          cublasOperation_t trans_b,
          cudaStream_t stream)
{
  detail::gemm(handle, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, stream);
}

/**
 * @brief A wrapper for CUBLS GEMM function designed for handling all possible
 * combinations of operand layouts.
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * @tparam T Data type of input/output matrices (float/double)
 * @param handle raft handle
 * @param z output matrix of size M rows x N columns
 * @param x input matrix of size M rows x K columns
 * @param y input matrix of size K rows x N columns
 * @param _M number of rows of X and Z
 * @param _N number of columns of Y and columns of Z
 * @param _K number of columns of X and rows of Y
 * @param isZColMajor Storage layout of Z. true = col major, false = row major
 * @param isXColMajor Storage layout of X. true = col major, false = row major
 * @param isYColMajor Storage layout of Y. true = col major, false = row major
 * @param stream cuda stream
 * @param alpha scalar
 * @param beta scalar
 */
template <typename T>
void gemm(raft::resources const& handle,
          T* z,
          T* x,
          T* y,
          int _M,
          int _N,
          int _K,
          bool isZColMajor,
          bool isXColMajor,
          bool isYColMajor,
          cudaStream_t stream,
          T alpha = T(1.0),
          T beta  = T(0.0))
{
  detail::gemm(
    handle, z, x, y, _M, _N, _K, isZColMajor, isXColMajor, isYColMajor, stream, &alpha, &beta);
}

/**
 * @defgroup gemm Matrix-Matrix Multiplication
 * @{
 */

/**
 * @brief GEMM function designed for handling all possible
 * combinations of operand layouts (raft::row_major or raft::col_major)
 * with scalars alpha and beta on the host or device
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * If alpha is not provided, it is assumed to be 1.0
 * If beta is not provided, it is assumed to be 0.0
 * @tparam ValueType Data type of input/output matrices (float/double)
 * @tparam IndexType Type of index
 * @tparam LayoutPolicyX layout of X
 * @tparam LayoutPolicyY layout of Y
 * @tparam LayoutPolicyZ layout of Z
 * @param[in] handle raft handle
 * @param[in] x input raft::device_matrix_view of size M rows x K columns
 * @param[in] y input raft::device_matrix_view of size K rows x N columns
 * @param[out] z output raft::device_matrix_view of size M rows x N columns
 * @param[in] alpha optional raft::host_scalar_view or raft::device_scalar_view, default 1.0
 * @param[in] beta optional raft::host_scalar_view or raft::device_scalar_view, default 0.0
 */
template <typename ValueType,
          typename IndexType,
          typename LayoutPolicyX,
          typename LayoutPolicyY,
          typename LayoutPolicyZ,
          typename ScalarIdxType  = std::uint32_t,
          typename ScalarViewType = raft::host_scalar_view<ValueType, ScalarIdxType>,
          typename                = std::enable_if_t<std::disjunction_v<
            std::is_same<ScalarViewType, raft::host_scalar_view<ValueType, ScalarIdxType>>,
            std::is_same<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>>>>
void gemm(raft::device_resources const& handle,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyX> x,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyY> y,
          raft::device_matrix_view<ValueType, IndexType, LayoutPolicyZ> z,
          std::optional<ScalarViewType> alpha = std::nullopt,
          std::optional<ScalarViewType> beta  = std::nullopt)
{
  RAFT_EXPECTS(raft::is_row_or_column_major(x), "X is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(y), "Y is not contiguous");
  RAFT_EXPECTS(raft::is_row_or_column_major(z), "Z is not contiguous");

  RAFT_EXPECTS(x.extent(0) == z.extent(0), "Number of rows of X and Z should be equal");
  RAFT_EXPECTS(y.extent(1) == z.extent(1), "Number of columns of Y and Z should be equal");
  RAFT_EXPECTS(x.extent(1) == y.extent(0), "Number of columns of X and rows of Y should be equal");

  constexpr auto is_x_col_major =
    std::is_same_v<typename decltype(x)::layout_type, raft::col_major>;
  constexpr auto is_y_col_major =
    std::is_same_v<typename decltype(y)::layout_type, raft::col_major>;
  constexpr auto is_z_col_major =
    std::is_same_v<typename decltype(z)::layout_type, raft::col_major>;

  constexpr auto device_mode =
    std::is_same_v<ScalarViewType, raft::device_scalar_view<ValueType, ScalarIdxType>>;

  ValueType alpha_value = 1;
  ValueType beta_value  = 0;

  auto alpha_device = raft::make_device_scalar(handle, alpha_value);
  auto beta_device  = raft::make_device_scalar(handle, beta_value);

  auto alpha_host = raft::make_host_scalar(alpha_value);
  auto beta_host  = raft::make_host_scalar(beta_value);

  if constexpr (device_mode) {
    if (!alpha) { alpha = alpha_device.view(); }
    if (!beta) { beta = beta_device.view(); }
  } else {
    if (!alpha) { alpha = alpha_host.view(); }
    if (!beta) { beta = beta_host.view(); }
  }

  detail::gemm<ValueType, device_mode>(handle,
                                       z.data_handle(),
                                       x.data_handle(),
                                       y.data_handle(),
                                       x.extent(0),
                                       y.extent(1),
                                       x.extent(1),
                                       is_z_col_major,
                                       is_x_col_major,
                                       is_y_col_major,
                                       handle.get_stream(),
                                       alpha.value().data_handle(),
                                       beta.value().data_handle());
}

/** @} */  // end of gemm

}  // end namespace linalg
}  // end namespace raft

#endif
