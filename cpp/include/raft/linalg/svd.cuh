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
#ifndef __SVD_H
#define __SVD_H

#pragma once

#include "detail/svd.cuh"

#include <optional>

namespace raft {
namespace linalg {

/**
 * @brief singular value decomposition (SVD) on the column major float type
 * input matrix using QR method
 * @param handle: raft handle
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular values of input matrix
 * @param right_sing_vecs: right singular values of input matrix
 * @param trans_right: transpose right vectors or not
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param stream cuda stream
 */
template <typename T>
void svdQR(const raft::handle_t& handle,
           T* in,
           int n_rows,
           int n_cols,
           T* sing_vals,
           T* left_sing_vecs,
           T* right_sing_vecs,
           bool trans_right,
           bool gen_left_vec,
           bool gen_right_vec,
           cudaStream_t stream)
{
  detail::svdQR(handle,
                in,
                n_rows,
                n_cols,
                sing_vals,
                left_sing_vecs,
                right_sing_vecs,
                trans_right,
                gen_left_vec,
                gen_right_vec,
                stream);
}

template <typename T>
void svdEig(const raft::handle_t& handle,
            T* in,
            int n_rows,
            int n_cols,
            T* S,
            T* U,
            T* V,
            bool gen_left_vec,
            cudaStream_t stream)
{
  detail::svdEig(handle, in, n_rows, n_cols, S, U, V, gen_left_vec, stream);
}

/**
 * @brief on the column major input matrix using Jacobi method
 * @param handle: raft handle
 * @param in: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param sing_vals: singular values of input matrix
 * @param left_sing_vecs: left singular vectors of input matrix
 * @param right_sing_vecs: right singular vectors of input matrix
 * @param gen_left_vec: generate left eig vector. Not activated.
 * @param gen_right_vec: generate right eig vector. Not activated.
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param max_sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 * @param stream cuda stream
 */
template <typename math_t>
void svdJacobi(const raft::handle_t& handle,
               math_t* in,
               int n_rows,
               int n_cols,
               math_t* sing_vals,
               math_t* left_sing_vecs,
               math_t* right_sing_vecs,
               bool gen_left_vec,
               bool gen_right_vec,
               math_t tol,
               int max_sweeps,
               cudaStream_t stream)
{
  detail::svdJacobi(handle,
                    in,
                    n_rows,
                    n_cols,
                    sing_vals,
                    left_sing_vecs,
                    right_sing_vecs,
                    gen_left_vec,
                    gen_right_vec,
                    tol,
                    max_sweeps,
                    stream);
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param handle: raft handle
 * @param U: left singular vectors of size n_rows x k
 * @param S: square matrix with singular values on its diagonal, k x k
 * @param V: right singular vectors of size n_cols x k
 * @param out: reconstructed matrix to be returned
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values
 * @param stream cuda stream
 */
template <typename math_t>
void svdReconstruction(const raft::handle_t& handle,
                       math_t* U,
                       math_t* S,
                       math_t* V,
                       math_t* out,
                       int n_rows,
                       int n_cols,
                       int k,
                       cudaStream_t stream)
{
  detail::svdReconstruction(handle, U, S, V, out, n_rows, n_cols, k, stream);
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param handle: raft handle
 * @param A_d: input matrix
 * @param U: left singular vectors of size n_rows x k
 * @param S_vec: singular values as a vector
 * @param V: right singular vectors of size n_cols x k
 * @param n_rows: number rows of output matrix
 * @param n_cols: number columns of output matrix
 * @param k: number of singular values to be computed, 1.0 for normal SVD
 * @param tol: tolerance for the evaluation
 * @param stream cuda stream
 */
template <typename math_t>
bool evaluateSVDByL2Norm(const raft::handle_t& handle,
                         math_t* A_d,
                         math_t* U,
                         math_t* S_vec,
                         math_t* V,
                         int n_rows,
                         int n_cols,
                         int k,
                         math_t tol,
                         cudaStream_t stream)
{
  return detail::evaluateSVDByL2Norm(handle, A_d, U, S_vec, V, n_rows, n_cols, k, tol, stream);
}

/**
 * @defgroup svd Singular Value Decomposition
 * @{
 */

/**
 * @brief singular value decomposition (SVD) on a column major
 * matrix using QR decomposition
 * @param[in] handle raft::handle_t
 * @param[in] in input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] sing_vals singular values raft::device_vector_view of shape (K)
 * @param[out] left_sing_vecs optional left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, n)
 * @param[out] right_sing_vecs optional right singular values of raft::device_matrix_view with
 * layout raft::col_major and dimensions (n, n)
 */
template <typename ValueType, typename IndexType>
void svd_qr(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> sing_vals,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> left_sing_vecs =
    std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> right_sing_vecs =
    std::nullopt)
{
  if (left_sing_vecs) {
    RAFT_EXPECTS(in.extent(0) == left_sing_vecs.value().extent(0) &&
                   in.extent(1) == left_sing_vecs.value().extent(1),
                 "U should have dimensions m * n");
  }
  if (right_sing_vecs) {
    RAFT_EXPECTS(in.extent(1) == right_sing_vecs.value().extent(0) &&
                   in.extent(1) == right_sing_vecs.value().extent(1),
                 "V should have dimensions n * n");
  }
  svdQR(handle,
        const_cast<ValueType*>(in.data_handle()),
        in.extent(0),
        in.extent(1),
        sing_vals.data_handle(),
        left_sing_vecs.value().data_handle(),
        right_sing_vecs.value().data_handle(),
        false,
        left_sing_vecs.has_value(),
        right_sing_vecs.has_value(),
        handle.get_stream());
}

/**
 * @brief Overload of `svd_qr` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `svd_qr`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void svd_qr(const raft::handle_t& handle,
            raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
            raft::device_vector_view<ValueType, IndexType> sing_vals,
            UType&& U,
            VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  svd_qr(handle, in, sing_vals, U_optional, V_optional);
}

/**
 * @brief singular value decomposition (SVD) on a column major
 * matrix using QR decomposition. Right singular vector matrix is transposed before returning
 * @param[in] handle raft::handle_t
 * @param[in] in input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] sing_vals singular values raft::device_vector_view of shape (K)
 * @param[out] left_sing_vecs optional left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, n)
 * @param[out] right_sing_vecs optional right singular values of raft::device_matrix_view with
 * layout raft::col_major and dimensions (n, n)
 */
template <typename ValueType, typename IndexType>
void svd_qr_transpose_right_vec(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> sing_vals,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> left_sing_vecs =
    std::nullopt,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> right_sing_vecs =
    std::nullopt)
{
  if (left_sing_vecs) {
    RAFT_EXPECTS(in.extent(0) == left_sing_vecs.value().extent(0) &&
                   in.extent(1) == left_sing_vecs.value().extent(1),
                 "U should have dimensions m * n");
  }
  if (right_sing_vecs) {
    RAFT_EXPECTS(in.extent(1) == right_sing_vecs.value().extent(0) &&
                   in.extent(1) == right_sing_vecs.value().extent(1),
                 "V should have dimensions n * n");
  }
  svdQR(handle,
        const_cast<ValueType*>(in.data_handle()),
        in.extent(0),
        in.extent(1),
        sing_vals.data_handle(),
        left_sing_vecs.value().data_handle(),
        right_sing_vecs.value().data_handle(),
        true,
        left_sing_vecs.has_value(),
        right_sing_vecs.has_value(),
        handle.get_stream());
}

/**
 * @brief Overload of `svd_qr_transpose_right_vec` to help the
 *   compiler find the above overload, in case users pass in
 *   `std::nullopt` for one or both of the optional arguments.
 *
 * Please see above for documentation of `svd_qr_transpose_right_vec`.
 */
template <typename ValueType, typename IndexType, typename UType, typename VType>
void svd_qr_transpose_right_vec(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> sing_vals,
  UType&& U,
  VType&& V)
{
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U_optional =
    std::forward<UType>(U);
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> V_optional =
    std::forward<VType>(V);

  svd_qr_transpose_right_vec(handle, in, sing_vals, U_optional, V_optional);
}

/**
 * @brief singular value decomposition (SVD) on a column major
 * matrix using Eigen decomposition. A square symmetric covariance matrix is constructed for the SVD
 * @param[in] handle raft::handle_t
 * @param[in] in input raft::device_matrix_view with layout raft::col_major of shape (M, N)
 * @param[out] S singular values raft::device_vector_view of shape (K)
 * @param[out] V right singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, n)
 * @param[out] U optional left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, n)
 */
template <typename ValueType, typename IndexType>
void svd_eig(
  const raft::handle_t& handle,
  raft::device_matrix_view<const ValueType, IndexType, raft::col_major> in,
  raft::device_vector_view<ValueType, IndexType> S,
  raft::device_matrix_view<ValueType, IndexType, raft::col_major> V,
  std::optional<raft::device_matrix_view<ValueType, IndexType, raft::col_major>> U = std::nullopt)
{
  if (U) {
    RAFT_EXPECTS(in.extent(0) == U.value().extent(0) && in.extent(1) == U.value().extent(1),
                 "U should have dimensions m * n");
  }
  RAFT_EXPECTS(in.extent(0) == V.extent(0) && in.extent(1) == V.extent(1),
               "V should have dimensions n * n");
  svdEig(handle,
         const_cast<ValueType*>(in.data_handle()),
         in.extent(0),
         in.extent(1),
         S.data_handle(),
         U.value().data_handle(),
         V.value().data_handle(),
         U.has_value(),
         handle.get_stream());
}

/**
 * @brief reconstruct a matrix use left and right singular vectors and
 * singular values
 * @param[in] handle raft::handle_t
 * @param[in] U left singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (m, k)
 * @param[in] S singular values raft::device_vector_view of shape (k, k)
 * @param[in] V right singular values of raft::device_matrix_view with layout
 * raft::col_major and dimensions (k, n)
 * @param[out] out output raft::device_matrix_view with layout raft::col_major of shape (m, n)
 */
template <typename ValueType, typename IndexType>
void svd_reconstruction(const raft::handle_t& handle,
                        raft::device_matrix_view<const ValueType, IndexType, raft::col_major> U,
                        raft::device_vector_view<const ValueType, IndexType> S,
                        raft::device_matrix_view<const ValueType, IndexType, raft::col_major> V,
                        raft::device_matrix_view<ValueType, IndexType, raft::col_major> out)
{
  RAFT_EXPECTS(S.extent(0) == S.extent(1), "S should be a square matrix");
  RAFT_EXPECTS(S.extent(0) == U.extent(1),
               "Number of rows of S should be equal to number of columns in U");
  RAFT_EXPECTS(S.extent(1) == V.extent(0),
               "Number of columns of S should be equal to number of rows in V");
  RAFT_EXPECTS(out.extent(0) == U.extent(0) && out.extent(1) == V.extent(1),
               "Number of rows should be equal in out and U and number of columns should be equal "
               "in out and V");

  svdReconstruction(handle,
                    const_cast<ValueType*>(U.data_handle()),
                    const_cast<ValueType*>(S.data_handle()),
                    const_cast<ValueType*>(V.data_handle()),
                    out.extent(0),
                    out.extent(1),
                    S.extent(0),
                    handle.get_stream());
}

/** @} */  // end of group svd

};  // end namespace linalg
};  // end namespace raft

#endif