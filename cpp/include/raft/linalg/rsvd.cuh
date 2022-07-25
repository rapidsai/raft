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

template <typename math_t>
void randomizedSVD(const raft::handle_t& handle,
                    math_t* in,
                    std::size_t n_rows,
                    std::size_t n_cols,
                    std::size_t k, //Rank of the k-SVD decomposition of matrix A. rank is less than min(m,n). 
                    std::size_t p, //Oversampling. The size of the subspace will be (k + p). (k+p) is less than min(m,n). 
                    std::size_t niters, //Number of iteration of power method. 
                    math_t* sing_vals,
                    math_t* left_sing_vecs,
                    math_t* right_sing_vecs,
                    bool trans_right, // Transpose the right singular vector back
                    bool gen_left_vec, // left vector needs to be generated or not?
                    bool gen_right_vec) // right vector needs to be generated or not?
{
  detail::randomizedSVD<math_t>(handle, in, n_rows, n_cols, k, p, niters, sing_vals, left_sing_vecs,
    right_sing_vecs, trans_right, gen_left_vec, gen_right_vec);
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