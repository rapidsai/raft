/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <raft/linalg/cublas_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <raft/handle.hpp>

namespace raft {
namespace linalg {

template <typename math_t>
void gemv(const raft::handle_t& handle,
          const math_t* A,
          const int n_rows,
          const int n_cols,
          const math_t* x,
          const int incx,
          math_t* y,
          const int incy,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          cudaStream_t stream)
{
  cublasHandle_t cublas_h = handle.get_cublas_handle();
  cublasOperation_t op_a  = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  RAFT_CUBLAS_TRY(
    cublasgemv(cublas_h, op_a, n_rows, n_cols, &alpha, A, n_rows, x, incx, &beta, y, incy, stream));
}

/**
 * y = alpha * op(A) * x + beta * y
 *
 * where
 *
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 *
 * @param lda is the leading dimension of A (number of rows); lda must be not smaller than n_rows_a.
 *     set it when you need to use only the first n_rows_a rows of the matrix A, which has
 *     (perhaps, due to padding) lda rows.
 *
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 *
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 */
template <typename math_t>
void gemv(const raft::handle_t& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          cudaStream_t stream)
{
  gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
}

/**
 * y = op(A) * x
 *
 * where
 *
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 *
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 *
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 */
template <typename math_t>
void gemv(const raft::handle_t& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          cudaStream_t stream)
{
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);

  gemv(handle, A, n_rows_a, n_cols_a, x, 1, y, 1, trans_a, alpha, beta, stream);
}

/**
 * y = alpha * op(A) * x + beta * y
 *
 * where
 *
 * @param alpha is a scalar scale of Ax.
 *
 * @param beta is a scalar scale of y.
 *
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 *
 * @param lda is the leading dimension of A (number of rows); lda must be not smaller than n_rows_a.
 *     set it when you need to use only the first n_rows_a rows of the matrix A, which has
 *     (perhaps, due to padding) lda rows.
 *
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 *
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 */
template <typename math_t>
void gemv(const raft::handle_t& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const int lda,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          const math_t alpha,
          const math_t beta,
          cudaStream_t stream)
{
  cublasHandle_t cublas_h = handle.get_cublas_handle();
  cublasOperation_t op_a  = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
  RAFT_CUBLAS_TRY(
    cublasgemv(cublas_h, op_a, n_rows_a, n_cols_a, &alpha, A, lda, x, 1, &beta, y, 1, stream));
}

/**
 * y = op(A) * x
 *
 * where
 *
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 *
 * @param lda is the leading dimension of A (number of rows); lda must be not smaller than n_rows_a.
 *     set it when you need to use only the first n_rows_a rows of the matrix A, which has
 *     (perhaps, due to padding) lda rows.
 *
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 *
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 *
 */
template <typename math_t>
void gemv(const raft::handle_t& handle,
          const math_t* A,
          const int n_rows_a,
          const int n_cols_a,
          const int lda,
          const math_t* x,
          math_t* y,
          const bool trans_a,
          cudaStream_t stream)
{
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, alpha, beta, stream);
}

};  // namespace linalg
};  // namespace raft
