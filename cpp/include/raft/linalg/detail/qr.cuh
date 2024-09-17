/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "cublas_wrappers.hpp"
#include "cusolver_wrappers.hpp"

#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/triangular.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace raft {
namespace linalg {
namespace detail {

/**
 * @brief Calculate the QR decomposition and get matrix Q in place of the input.
 *
 * Subject to the algorithm constraint `n_rows >= n_cols`.
 *
 * @param handle
 * @param[inout] Q device pointer to input matrix and the output matrix Q,
 *                 both column-major and of size [n_rows, n_cols].
 * @param n_rows
 * @param n_cols
 * @param stream
 */
template <typename math_t>
void qrGetQ_inplace(
  raft::resources const& handle, math_t* Q, int n_rows, int n_cols, cudaStream_t stream)
{
  RAFT_EXPECTS(n_rows >= n_cols, "QR decomposition expects n_rows >= n_cols.");
  cusolverDnHandle_t cusolver = resource::get_cusolver_dn_handle(handle);

  rmm::device_uvector<math_t> tau(n_cols, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * n_cols, stream));

  rmm::device_scalar<int> dev_info(stream);
  int ws_size;

  RAFT_CUSOLVER_TRY(cusolverDngeqrf_bufferSize(cusolver, n_rows, n_cols, Q, n_rows, &ws_size));
  rmm::device_uvector<math_t> workspace(ws_size, stream);
  RAFT_CUSOLVER_TRY(cusolverDngeqrf(cusolver,
                                    n_rows,
                                    n_cols,
                                    Q,
                                    n_rows,
                                    tau.data(),
                                    workspace.data(),
                                    ws_size,
                                    dev_info.data(),
                                    stream));

  RAFT_CUSOLVER_TRY(
    cusolverDnorgqr_bufferSize(cusolver, n_rows, n_cols, n_cols, Q, n_rows, tau.data(), &ws_size));
  workspace.resize(ws_size, stream);
  RAFT_CUSOLVER_TRY(cusolverDnorgqr(cusolver,
                                    n_rows,
                                    n_cols,
                                    n_cols,
                                    Q,
                                    n_rows,
                                    tau.data(),
                                    workspace.data(),
                                    ws_size,
                                    dev_info.data(),
                                    stream));
}

template <typename math_t>
void qrGetQ(raft::resources const& handle,
            const math_t* M,
            math_t* Q,
            int n_rows,
            int n_cols,
            cudaStream_t stream)
{
  raft::copy(Q, M, n_rows * n_cols, stream);
  qrGetQ_inplace(handle, Q, n_rows, n_cols, stream);
}

template <typename math_t>
void qrGetQR(raft::resources const& handle,
             math_t* M,
             math_t* Q,
             math_t* R,
             int n_rows,
             int n_cols,
             cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = resource::get_cusolver_dn_handle(handle);

  int m = n_rows, n = n_cols;
  rmm::device_uvector<math_t> R_full(m * n, stream);
  rmm::device_uvector<math_t> tau(std::min(m, n), stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * std::min(m, n), stream));
  int R_full_nrows = m, R_full_ncols = n;
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(R_full.data(), M, sizeof(math_t) * m * n, cudaMemcpyDeviceToDevice, stream));

  int Lwork;
  rmm::device_scalar<int> devInfo(stream);

  RAFT_CUSOLVER_TRY(cusolverDngeqrf_bufferSize(
    cusolverH, R_full_nrows, R_full_ncols, R_full.data(), R_full_nrows, &Lwork));
  rmm::device_uvector<math_t> workspace(Lwork, stream);
  RAFT_CUSOLVER_TRY(cusolverDngeqrf(cusolverH,
                                    R_full_nrows,
                                    R_full_ncols,
                                    R_full.data(),
                                    R_full_nrows,
                                    tau.data(),
                                    workspace.data(),
                                    Lwork,
                                    devInfo.data(),
                                    stream));

  raft::matrix::upper_triangular<math_t, int>(
    handle,
    make_device_matrix_view<const math_t, int, col_major>(R_full.data(), m, n),
    make_device_matrix_view<math_t, int, col_major>(R, std::min(m, n), std::min(m, n)));

  RAFT_CUDA_TRY(
    cudaMemcpyAsync(Q, R_full.data(), sizeof(math_t) * m * n, cudaMemcpyDeviceToDevice, stream));
  int Q_nrows = m, Q_ncols = n;

  RAFT_CUSOLVER_TRY(cusolverDnorgqr_bufferSize(
    cusolverH, Q_nrows, Q_ncols, std::min(Q_ncols, Q_nrows), Q, Q_nrows, tau.data(), &Lwork));
  workspace.resize(Lwork, stream);
  RAFT_CUSOLVER_TRY(cusolverDnorgqr(cusolverH,
                                    Q_nrows,
                                    Q_ncols,
                                    std::min(Q_ncols, Q_nrows),
                                    Q,
                                    Q_nrows,
                                    tau.data(),
                                    workspace.data(),
                                    Lwork,
                                    devInfo.data(),
                                    stream));
}

};  // namespace detail
};  // namespace linalg
};  // namespace raft
