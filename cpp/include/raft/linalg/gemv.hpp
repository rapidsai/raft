/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include "detail/gemv.hpp"

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
