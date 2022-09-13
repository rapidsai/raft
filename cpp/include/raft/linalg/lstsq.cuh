/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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
#ifndef __LSTSQ_H
#define __LSTSQ_H

#pragma once

#include <raft/handle.hpp>
#include <raft/linalg/detail/lstsq.cuh>
namespace raft {
namespace linalg {

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S Vt` using default cuSOLVER routine.
 *
 * @param[in] handle raft handle
 * @param[inout] A input feature matrix.
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param[in] n_rows number of rows in A
 * @param[in] n_cols number of columns in A
 * @param[inout] b input target vector.
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
 * @param[out] w output coefficient vector
 * @param[in] stream cuda stream for ordering operations
 */
template <typename math_t>
void lstsqSvdQR(const raft::handle_t& handle,
                math_t* A,
                const int n_rows,
                const int n_cols,
                const math_t* b,
                math_t* w,
                cudaStream_t stream)
{
  detail::lstsqSvdQR(handle, A, n_rows, n_cols, b, w, stream);
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S V^T` using Jacobi iterations (cuSOLVER).
 *
 * @param[in] handle raft handle
 * @param[inout] A input feature matrix.
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param[in] n_rows number of rows in A
 * @param[in] n_cols number of columns in A
 * @param[inout] b input target vector.
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
 * @param[out] w output coefficient vector
 * @param[in] stream cuda stream for ordering operations
 */
template <typename math_t>
void lstsqSvdJacobi(const raft::handle_t& handle,
                    math_t* A,
                    const int n_rows,
                    const int n_cols,
                    const math_t* b,
                    math_t* w,
                    cudaStream_t stream)
{
  detail::lstsqSvdJacobi(handle, A, n_rows, n_cols, b, w, stream);
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  via eigenvalue decomposition of `A^T * A` (covariance matrix for dataset A).
 *  (`w = (A^T A)^-1  A^T b`)
 */
template <typename math_t>
void lstsqEig(const raft::handle_t& handle,
              const math_t* A,
              const int n_rows,
              const int n_cols,
              const math_t* b,
              math_t* w,
              cudaStream_t stream)
{
  detail::lstsqEig(handle, A, n_rows, n_cols, b, w, stream);
}

/** Solves the linear ordinary least squares problem `Aw = b`
 *  via QR decomposition of `A = QR`.
 *  (triangular system of equations `Rw = Q^T b`)
 *
 * @param[in] handle raft handle
 * @param[inout] A input feature matrix.
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param[in] n_rows number of rows in A
 * @param[in] n_cols number of columns in A
 * @param[inout] b input target vector.
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
 * @param[out] w output coefficient vector
 * @param[in] stream cuda stream for ordering operations
 */
template <typename math_t>
void lstsqQR(const raft::handle_t& handle,
             math_t* A,
             const int n_rows,
             const int n_cols,
             math_t* b,
             math_t* w,
             cudaStream_t stream)
{
  detail::lstsqQR(handle, A, n_rows, n_cols, b, w, stream);
}

/**
 * @defgroup lstsq Least Squares Methods
 * @{
 */

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 * Via SVD decomposition of `A = U S Vt`.
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::handle_t
 * @param[inout] A input raft::device_matrix_view
 *            Warning: the content of this matrix is modified.
 * @param[inout] b input target raft::device_vector_view
 *            Warning: the content of this vector is modified.
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_svd_qr(const raft::handle_t& handle,
                  raft::device_matrix_view<ValueType, IndexType, raft::col_major> A,
                  raft::device_vector_view<ValueType, IndexType> b,
                  raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(b.is_exhaustive(), "Eigen Vectors must be contiguous");
  RAFT_EXPECTS(w.is_exhaustive(), "Eigen Values must be contiguous");
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqSvdQR(handle,
             A.data_handle(),
             A.extent(0),
             A.extent(1),
             b.data_handle(),
             w.data_handle(),
             handle.get_stream());
}

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S V^T` using Jacobi iterations.
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::handle_t
 * @param[inout] A input raft::device_matrix_view
 *            Warning: the content of this matrix is modified.
 * @param[inout] b input target raft::device_vector_view
 *            Warning: the content of this vector is modified.
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_svd_jacobi(const raft::handle_t& handle,
                      raft::device_matrix_view<ValueType, IndexType, raft::col_major> A,
                      raft::device_vector_view<ValueType, IndexType> b,
                      raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(b.is_exhaustive(), "Eigen Vectors must be contiguous");
  RAFT_EXPECTS(w.is_exhaustive(), "Eigen Values must be contiguous");
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqSvdJacobi(handle,
                 A.data_handle(),
                 A.extent(0),
                 A.extent(1),
                 b.data_handle(),
                 w.data_handle(),
                 handle.get_stream());
}

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 *  via eigenvalue decomposition of `A^T * A` (covariance matrix for dataset A).
 *  (`w = (A^T A)^-1  A^T b`)
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::handle_t
 * @param[inout] A input raft::device_matrix_view
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param[inout] b input target raft::device_vector_view
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_eig(const raft::handle_t& handle,
               raft::device_matrix_view<ValueType, IndexType, raft::col_major> A,
               raft::device_vector_view<ValueType, IndexType> b,
               raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(b.is_exhaustive(), "Eigen Vectors must be contiguous");
  RAFT_EXPECTS(w.is_exhaustive(), "Eigen Values must be contiguous");
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqEig(handle,
           A.data_handle(),
           A.extent(0),
           A.extent(1),
           b.data_handle(),
           w.data_handle(),
           handle.get_stream());
}

/**
 * @brief Solves the linear ordinary least squares problem `Aw = b`
 *  via QR decomposition of `A = QR`.
 *  (triangular system of equations `Rw = Q^T b`)
 *
 * @tparam ValueType the data-type of input/output
 * @param[in] handle raft::handle_t
 * @param[inout] A input raft::device_matrix_view
 *            Warning: the content of this matrix is modified.
 * @param[inout] b input target raft::device_vector_view
 *            Warning: the content of this vector is modified.
 * @param[out] w output coefficient raft::device_vector_view
 */
template <typename ValueType, typename IndexType>
void lstsq_qr(const raft::handle_t& handle,
              raft::device_matrix_view<ValueType, IndexType, raft::col_major> A,
              raft::device_vector_view<ValueType, IndexType> b,
              raft::device_vector_view<ValueType, IndexType> w)
{
  RAFT_EXPECTS(A.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(b.is_exhaustive(), "Eigen Vectors must be contiguous");
  RAFT_EXPECTS(w.is_exhaustive(), "Eigen Values must be contiguous");
  RAFT_EXPECTS(A.extent(1) == w.size(), "Size mismatch between A and w");
  RAFT_EXPECTS(A.extent(0) == b.size(), "Size mismatch between A and b");

  lstsqQR(handle,
          A.data_handle(),
          A.extent(0),
          A.extent(1),
          b.data_handle(),
          w.data_handle(),
          handle.get_stream());
}

/** @} */  // end of lstsq

};  // namespace linalg
};  // namespace raft

#endif