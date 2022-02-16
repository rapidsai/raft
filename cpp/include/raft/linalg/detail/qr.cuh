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

#include "cublas_wrappers.hpp"
#include "cusolver_wrappers.hpp"
#include <raft/matrix/matrix.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>

namespace raft {
namespace linalg {
namespace detail {

template <typename math_t>
void qrGetQ(const raft::handle_t& handle,
            const math_t* M,
            math_t* Q,
            int n_rows,
            int n_cols,
            cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  int m = n_rows, n = n_cols;
  int k = std::min(m, n);
  RAFT_CUDA_TRY(cudaMemcpyAsync(Q, M, sizeof(math_t) * m * n, cudaMemcpyDeviceToDevice, stream));

  rmm::device_uvector<math_t> tau(k, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(tau.data(), 0, sizeof(math_t) * k, stream));

  rmm::device_scalar<int> devInfo(stream);
  int Lwork;

  RAFT_CUSOLVER_TRY(cusolverDngeqrf_bufferSize(cusolverH, m, n, Q, m, &Lwork));
  rmm::device_uvector<math_t> workspace(Lwork, stream);
  RAFT_CUSOLVER_TRY(cusolverDngeqrf(
    cusolverH, m, n, Q, m, tau.data(), workspace.data(), Lwork, devInfo.data(), stream));

  RAFT_CUSOLVER_TRY(cusolverDnorgqr_bufferSize(cusolverH, m, n, k, Q, m, tau.data(), &Lwork));
  workspace.resize(Lwork, stream);
  RAFT_CUSOLVER_TRY(cusolverDnorgqr(
    cusolverH, m, n, k, Q, m, tau.data(), workspace.data(), Lwork, devInfo.data(), stream));
}

template <typename math_t>
void qrGetQR(const raft::handle_t& handle,
             math_t* M,
             math_t* Q,
             math_t* R,
             int n_rows,
             int n_cols,
             cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

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

  raft::matrix::copyUpperTriangular(R_full.data(), R, m, n, stream);

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
