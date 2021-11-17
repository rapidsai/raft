/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include "detail/eig.hpp"

namespace raft {
namespace linalg {

/**
 * @defgroup eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param handle raft handle
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param stream cuda stream
 * @{
 */
template <typename math_t>
void eigDC(const raft::handle_t &handle, const math_t *in, std::size_t n_rows,
           std::size_t n_cols, math_t *eig_vectors, math_t *eig_vals,
           cudaStream_t stream) {
  detail::eigDC(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream);
}

using detail::COPY_INPUT;
using detail::EigVecMemUsage;
using detail::OVERWRITE_INPUT;

#if CUDART_VERSION >= 10010

/**
 * @defgroup eig decomp with divide and conquer method for the column-major
 * symmetric matrices
 * @param handle raft handle
 * @param in the input buffer (symmetric matrix that has real eig values and
 * vectors.
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param n_eig_vals: number of eigenvectors to be generated
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param stream cuda stream
 * @{
 */
template <typename math_t>
void eigSelDC(const raft::handle_t &handle, math_t *in, int n_rows, int n_cols,
              int n_eig_vals, math_t *eig_vectors, math_t *eig_vals,
              EigVecMemUsage memUsage, cudaStream_t stream) {
  detail::eigSelDC(handle, in, n_rows, n_cols, n_eig_vals, eig_vectors,
                   eig_vals, memUsage, stream);
}

#endif

/**
 * @defgroup overloaded function for eig decomp with Jacobi method for the
 * column-major symmetric matrices (in parameter)
 * @param handle: raft handle
 * @param n_rows: number of rows of the input
 * @param n_cols: number of cols of the input
 * @param eig_vectors: eigenvectors
 * @param eig_vals: eigen values
 * @param tol: error tolerance for the jacobi method. Algorithm stops when the
 * error is below tol
 * @param sweeps: number of sweeps in the Jacobi algorithm. The more the better
 * accuracy.
 * @{
 */
template <typename math_t>
void eigJacobi(const raft::handle_t &handle, const math_t *in, int n_rows,
               int n_cols, math_t *eig_vectors, math_t *eig_vals,
               cudaStream_t stream, math_t tol = 1.e-7, int sweeps = 15) {
  detail::eigJacobi(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream,
                    tol, sweeps);
}

};  // end namespace linalg
};  // end namespace raft
