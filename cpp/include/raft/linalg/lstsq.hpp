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

#pragma once

#include <raft/handle.hpp>
#include <raft/linalg/detail/lstsq.hpp>
namespace raft {
namespace linalg {

/** Solves the linear ordinary least squares problem `Aw = b`
 *  Via SVD decomposition of `A = U S Vt` using default cuSOLVER routine.
 *
 *  @param A - input feature matrix; it's marked [in/out] in the used cuSOLVER routines,
 *             so it's not guaranteed to stay unmodified.
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
 *  @param A - input feature matrix; it's marked [in/out] in the used cuSOLVER routines,
 *             so it's not guaranteed to stay unmodified.
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
 * @param A[in/out] - input feature matrix.
 *            Warning: the content of this matrix is modified by the cuSOLVER routines.
 * @param b[in/out] - input target vector.
 *            Warning: the content of this vector is modified by the cuSOLVER routines.
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

};  // namespace linalg
};  // namespace raft
