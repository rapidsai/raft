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

#include "detail/svd.hpp"

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
// TODO: activate gen_left_vec and gen_right_vec options
// TODO: couldn't template this function due to cusolverDnSgesvd and
// cusolverSnSgesvd. Check if there is any other way.
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

};  // end namespace linalg
};  // end namespace raft
