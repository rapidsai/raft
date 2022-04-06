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
/**
 * This file is deprecated and will be removed in release 22.06.
 * Please use the cuh version instead.
 */

#ifndef __GEMV_H
#define __GEMV_H

#pragma once

#include "detail/gemv.hpp"

namespace raft {
namespace linalg {

/**
 * @brief the wrapper of cublas gemv function
 *  It computes the following equation: y = alpha .* op(A) * x + beta .* y
 *
 * @tparam math_t the element type
 * @tparam DevicePointerMode whether pointers alpha, beta point to device memory
 * @param [in] handle raft handle
 * @param [in] trans_a cublas transpose op for A
 * @param [in] m number of rows of A
 * @param [in] n number of columns of A
 * @param [in] alpha host or device scalar
 * @param [in] A column-major matrix of size [m, n]
 * @param [in] lda leading dimension of A
 * @param [in] x vector of length n if trans_a else m
 * @param [in] incx stride between consecutive elements of x
 * @param [in] beta host or device scalar
 * @param [inout] y vector of length m if trans_a else n
 * @param [in] incy stride between consecutive elements of y
 * @param [in] stream
 */
template <typename math_t, bool DevicePointerMode = false>
void gemv(const raft::handle_t& handle,
          const bool trans_a,
          const int m,
          const int n,
          const math_t* alpha,
          const math_t* A,
          const int lda,
          const math_t* x,
          const int incx,
          const math_t* beta,
          math_t* y,
          const int incy,
          cudaStream_t stream)
{
  detail::gemv<math_t, DevicePointerMode>(
    handle, trans_a, m, n, alpha, A, lda, x, incx, beta, y, incy, stream);
}

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
  detail::gemv(handle, A, n_rows, n_cols, x, incx, y, incy, trans_a, alpha, beta, stream);
}

/**
 * y = alpha * op(A) * x + beta * y
 *
 * where
 *
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param alpha is a scalar scale of Ax.
 * @param beta is a scalar scale of y.
 * @param stream stream on which this function is run
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
  detail::gemv(handle, A, n_rows_a, n_cols_a, x, y, trans_a, alpha, beta, stream);
}

/**
 * y = op(A) * x
 *
 * where
 *
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param stream stream on which this function is run
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
  detail::gemv(handle, A, n_rows_a, n_cols_a, x, y, trans_a, stream);
}

/**
 * y = alpha * op(A) * x + beta * y
 *
 * where
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param lda is the leading dimension of A (number of rows); lda must be not smaller than n_rows_a.
 *     set it when you need to use only the first n_rows_a rows of the matrix A, which has
 *     (perhaps, due to padding) lda rows.
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param alpha is a scalar scale of Ax.
 * @param beta is a scalar scale of y.
 * @param stream stream on which this function is run
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
  detail::gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, alpha, beta, stream);
}

/**
 * y = op(A) * x
 *
 * where
 * @param handle raft handle
 * @param A is a column-major matrix of size n_rows_a * n_cols_a.
 *   op(A) is either the transpose operation (trans_a == true) or identity.
 * @param n_rows_a number of rows in A
 * @param n_cols_a number of cols in A
 * @param lda is the leading dimension of A (number of rows); lda must be not smaller than n_rows_a.
 *     set it when you need to use only the first n_rows_a rows of the matrix A, which has
 *     (perhaps, due to padding) lda rows.
 * @param x is a vector of size `trans_a ? n_rows_a : n_cols_a`.
 * @param y is a vector of size `trans_a ? n_cols_a : n_rows_a`.
 * @param trans_a whether to take transpose of a
 * @param stream stream on which this function is run
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
  detail::gemv(handle, A, n_rows_a, n_cols_a, lda, x, y, trans_a, stream);
}

};  // namespace linalg
};  // namespace raft

#endif