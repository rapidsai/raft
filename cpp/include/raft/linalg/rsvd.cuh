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
#ifndef __RSVD_H
#define __RSVD_H

#pragma once

#include <raft/linalg/detail/rsvd.cuh>

namespace raft {
namespace linalg {

/**
 * @brief randomized singular value decomposition (RSVD)
 * @param handle:  raft handle
 * @param in:      input matrix
 *                 [dim = n_rows * n_cols] 
 * @param n_rows:  number rows of input matrix
 * @param n_cols:  number columns of input matrix
 * @param k:       Rank of the k-SVD decomposition of matrix in. Number of singular values to be computed.
 *                 The rank is less than min(m,n). 
 * @param p:       Oversampling. The size of the subspace will be (k + p). (k+p) is less than min(m,n).
 *                 (Recommanded to be at least 2*k)
 * @param niters:  Number of iteration of power method.
 * @param S:       array of singular values of input matrix.
 *                 [dim = min(n_rows, n_cols)] 
 * @param U:       left singular values of input matrix.
 *                 [dim = n_rows * n_rows] if gen_U
 *                 [dim = min(n_rows,n_cols) * n_rows] else
 * @param V:       right singular values of input matrix.
 *                 [dim = n_cols * n_cols] if gen_V
 *                 [dim = min(n_rows,n_cols) * n_cols] else
 * @param trans_V: Transpose V back ?
 * @param gen_U:   left vector needs to be generated or not?
 * @param gen_V:   right vector needs to be generated or not?
 */
template <typename math_t>
void randomizedSVD(const raft::handle_t& handle,
                   math_t* in,
                   std::size_t n_rows,
                   std::size_t n_cols,
                   std::size_t k,
                   std::size_t p,
                   std::size_t niters,
                   math_t* S,
                   math_t* U,
                   math_t* V,
                   bool trans_V,
                   bool gen_U,
                   bool gen_V)
{
  detail::randomizedSVD<math_t>(handle, in, n_rows, n_cols, k, p, niters, S, U,
    V, trans_V, gen_U, gen_V);
}


/**
 * @brief randomized singular value decomposition (RSVD)
 * @param handle:  raft handle
 * @param in:      input matrix
 *                 [dim = n_rows * n_cols] 
 * @param n_rows:  number rows of input matrix
 * @param n_cols:  number columns of input matrix
 * @param k:       Rank of the k-SVD decomposition of matrix in. Number of singular values to be computed.
 *                 The rank is less than min(m,n). 
 * @param p:       Oversampling. The size of the subspace will be (k + p). (k+p) is less than min(m,n).
 *                 (Recommanded to be at least 2*k)
 * @param niters:  Number of iteration of power method. (2 is recommanded)
 * @param S:       array of singular values of input matrix.
 *                 [dim = min(n_rows, n_cols)] 
 * @param U:       left singular values of input matrix.
 *                 [dim = n_rows * n_rows] if gen_U
 *                 [dim = min(n_rows,n_cols) * n_rows] else
 * @param V:       right singular values of input matrix.
 *                 [dim = n_cols * n_cols] if gen_V
 *                 [dim = min(n_rows,n_cols) * n_cols] else
 * @param trans_V: Transpose V back ?
 * @param gen_U:   left vector needs to be generated or not?
 * @param gen_V:   right vector needs to be generated or not?
 */
template <typename math_t>
void randomizedSVD(const raft::handle_t& handle,
                   const raft::device_matrix_view<math_t>& in,
                   std::size_t k,
                   std::size_t p,
                   std::size_t niters,
                   const raft::device_vector_view<math_t>& S,
                   const raft::device_matrix_view<math_t>& U,
                   const raft::device_matrix_view<math_t>& V,
                   bool trans_V,
                   bool gen_U,
                   bool gen_V)
{
  detail::randomizedSVD<math_t>(handle, in, in.extent(0), in.extent(1), k, p, niters, S.data(), U.data(),
    V.data(), trans_V, gen_U, gen_V);
}

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying no. of PCs and
 * upsamples directly
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param k: no. of singular values to be computed
 * @param p: no. of upsamples
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdFixedRank(const raft::handle_t& handle,
                   math_t* M,
                   int n_rows,
                   int n_cols,
                   math_t* S_vec,
                   math_t* U,
                   math_t* V,
                   int k,
                   int p,
                   bool use_bbt,
                   bool gen_left_vec,
                   bool gen_right_vec,
                   bool use_jacobi,
                   math_t tol,
                   int max_sweeps,
                   cudaStream_t stream)
{
  detail::rsvdFixedRank(handle,
                        M,
                        n_rows,
                        n_cols,
                        S_vec,
                        U,
                        V,
                        k,
                        p,
                        use_bbt,
                        gen_left_vec,
                        gen_right_vec,
                        use_jacobi,
                        tol,
                        max_sweeps,
                        stream);
}

/**
 * @brief randomized singular value decomposition (RSVD) on the column major
 * float type input matrix (Jacobi-based), by specifying the PC and upsampling
 * ratio
 * @param handle: raft handle
 * @param M: input matrix
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param S_vec: singular values of input matrix
 * @param U: left singular values of input matrix
 * @param V: right singular values of input matrix
 * @param PC_perc: percentage of singular values to be computed
 * @param UpS_perc: upsampling percentage
 * @param use_bbt: whether use eigen decomposition in computation or not
 * @param gen_left_vec: left vector needs to be generated or not?
 * @param gen_right_vec: right vector needs to be generated or not?
 * @param use_jacobi: whether to jacobi solver for decomposition
 * @param tol: tolerance for Jacobi-based solvers
 * @param max_sweeps: maximum number of sweeps for Jacobi-based solvers
 * @param stream cuda stream
 */
template <typename math_t>
void rsvdPerc(const raft::handle_t& handle,
              math_t* M,
              int n_rows,
              int n_cols,
              math_t* S_vec,
              math_t* U,
              math_t* V,
              math_t PC_perc,
              math_t UpS_perc,
              bool use_bbt,
              bool gen_left_vec,
              bool gen_right_vec,
              bool use_jacobi,
              math_t tol,
              int max_sweeps,
              cudaStream_t stream)
{
  detail::rsvdPerc(handle,
                   M,
                   n_rows,
                   n_cols,
                   S_vec,
                   U,
                   V,
                   PC_perc,
                   UpS_perc,
                   use_bbt,
                   gen_left_vec,
                   gen_right_vec,
                   use_jacobi,
                   tol,
                   max_sweeps,
                   stream);
}

};  // end namespace linalg
};  // end namespace raft

#endif