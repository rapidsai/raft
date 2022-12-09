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
#ifndef __QR_H
#define __QR_H

#pragma once

#include "detail/qr.cuh"

namespace raft {
namespace linalg {

/**
 * @brief compute QR decomp and return only Q matrix
 * @param handle: raft handle
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void qrGetQ(const raft::handle_t& handle,
            const math_t* M,
            math_t* Q,
            int n_rows,
            int n_cols,
            cudaStream_t stream)
{
  detail::qrGetQ(handle, M, Q, n_rows, n_cols, stream);
}

/**
 * @brief compute QR decomp and return both Q and R matrices
 * @param handle: raft handle
 * @param M: input matrix
 * @param Q: Q matrix to be returned (on GPU)
 * @param R: R matrix to be returned (on GPU)
 * @param n_rows: number rows of input matrix
 * @param n_cols: number columns of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void qrGetQR(const raft::handle_t& handle,
             math_t* M,
             math_t* Q,
             math_t* R,
             int n_rows,
             int n_cols,
             cudaStream_t stream)
{
  detail::qrGetQR(handle, M, Q, R, n_rows, n_cols, stream);
}

/**
 * @defgroup qr QR Decomposition
 * @{
 */

/**
 * @brief Compute the QR decomposition of matrix M and return only the Q matrix.
 * @param[in] handle raft::handle_t
 * @param[in] M Input raft::device_matrix_view
 * @param[out] Q Output raft::device_matrix_view
 */
template <typename ElementType, typename IndexType>
void qr_get_q(const raft::handle_t& handle,
              raft::device_matrix_view<const ElementType, IndexType, raft::col_major> M,
              raft::device_matrix_view<ElementType, IndexType, raft::col_major> Q)
{
  RAFT_EXPECTS(Q.size() == M.size(), "Size mismatch between Output and Input");

  qrGetQ(handle, M.data_handle(), Q.data_handle(), M.extent(0), M.extent(1), handle.get_stream());
}

/**
 * @brief Compute the QR decomposition of matrix M and return both the Q and R matrices.
 * @param[in] handle raft::handle_t
 * @param[in] M Input raft::device_matrix_view
 * @param[in] Q Output raft::device_matrix_view
 * @param[out] R Output raft::device_matrix_view
 */
template <typename ElementType, typename IndexType>
void qr_get_qr(const raft::handle_t& handle,
               raft::device_matrix_view<const ElementType, IndexType, raft::col_major> M,
               raft::device_matrix_view<ElementType, IndexType, raft::col_major> Q,
               raft::device_matrix_view<ElementType, IndexType, raft::col_major> R)
{
  RAFT_EXPECTS(Q.size() == M.size(), "Size mismatch between Output and Input");

  qrGetQR(handle,
          M.data_handle(),
          Q.data_handle(),
          R.data_handle(),
          M.extent(0),
          M.extent(1),
          handle.get_stream());
}

/** @} */

};  // namespace linalg
};  // namespace raft

#endif