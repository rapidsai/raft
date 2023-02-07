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

#pragma once

#include <cublas_v2.h>

#include "cublas_wrappers.hpp"

#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resources.hpp>

namespace raft {
namespace linalg {
namespace detail {

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
  auto cublas_h = raft::resource::get_cublas_handle(handle);
  cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);
  RAFT_CUBLAS_TRY(cublasgemm(cublas_h,
                             trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                             trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                             m,
                             n,
                             k,
                             alpha,
                             A,
                             lda,
                             B,
                             ldb,
                             beta,
                             C,
                             ldc,
                             stream));
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
  auto cublas_h = raft::resource::get_cublas_handle(handle);

  int m   = n_rows_c;
  int n   = n_cols_c;
  int k   = trans_a == CUBLAS_OP_T ? n_rows_a : n_cols_a;
  int lda = trans_a == CUBLAS_OP_T ? k : m;
  int ldb = trans_b == CUBLAS_OP_T ? n : k;
  int ldc = m;
  RAFT_CUBLAS_TRY(
    cublasgemm(cublas_h, trans_a, trans_b, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc, stream));
}

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
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  gemm(
    handle, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, alpha, beta, stream);
}

template <typename T, bool DevicePointerMode = false>
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
          T* alpha,
          T* beta)
{
  auto cublas_h = raft::resource::get_cublas_handle(handle);
  cublas_device_pointer_mode<DevicePointerMode> pmode(cublas_h);

  cublasOperation_t trans_a, trans_b;
  T *a, *b, *c;
  int lda, ldb, ldc;
  int M, N, K;
  // This function performs c = a * b. Based on the required output layout,
  // either a = x,  b = y or a = y, b = x. In either case c = z.
  if (isZColMajor == true) {
    // Result c is required in column major layout. Thus we perform,
    // z = x * y
    // Using BLAS call c = a * b. Therefore a = x, b = y and c = z

    a = x;
    // If x is in row major layout, cublas needs to transpose x first,
    // therefore trans_x needs to be CUBLAS_OP_T. If x is in column major
    // layout, trans_b needs to be CUBLAS_OP_N.
    trans_a = isXColMajor == true ? CUBLAS_OP_N : CUBLAS_OP_T;
    // Set leading dimension appropriately
    lda = isXColMajor == true ? _M : _K;

    b = y;
    // If y is in row major layout, cublas needs to transpose y first,
    // therefore trans_x needs to be CUBLAS_OP_T. If x is in column major
    // layout, trans_b needs to be CUBLAS_OP_N.
    trans_b = isYColMajor == true ? CUBLAS_OP_N : CUBLAS_OP_T;
    ldb     = isYColMajor == true ? _K : _N;

    c   = z;
    ldc = _M;
    M   = _M;
    N   = _N;
    K   = _K;
  } else {
    // Result c is required in row major layout Thus we pick
    // a = y, b = x and c = a * b = y * x
    // cublas produces output matrix only in column major layout. To get output
    // matrix on row major layout, we need to produce transpose of output
    // in column major layout. Therefore we perform,
    // tr(z) = tr(y) * tr(x)
    // we model this using cublas call for c = a * b
    // therefore a = tr(y), b = tr(x) and c = tr(z)

    a = y;
    // If y is in row major layout, it can be/ interpreted as tr(y) on column
    // major layout. Therefore we can pass trans_a as CUBLAS_OP_N. If y is in
    // column major layout, cublas needs to transpose y first, therefore
    // trans_a needs to be CUBLAS_OP_T
    trans_a = isYColMajor == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    // Set leading dimension appropriately
    lda = isYColMajor == true ? _K : _N;

    b = x;
    // If x is in row major layout, it can be interpreted as tr(x) on column
    // major layout. Therefore we can pass trans_b as CUBLAS_OP_N. If x is in
    // column major layout, cublas needs to trasponse x first, therefore
    // trans_b needs to be CUBLAS_OP_T
    trans_b = isXColMajor == true ? CUBLAS_OP_T : CUBLAS_OP_N;
    // Set leading dimension appropriately
    ldb = isXColMajor == true ? _M : _K;

    c   = z;
    ldc = _N;

    M = _N;
    N = _M;
    K = _K;
  }
  // Actual cuBLAS call
  RAFT_CUBLAS_TRY(
    cublasgemm(cublas_h, trans_a, trans_b, M, N, K, alpha, a, lda, b, ldb, beta, c, ldc, stream));
}

}  // namespace detail
}  // namespace linalg
}  // namespace raft
