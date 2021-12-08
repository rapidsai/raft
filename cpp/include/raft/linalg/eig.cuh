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

#include <cuda_runtime_api.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/cusolver_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/matrix/matrix.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

namespace raft {
namespace linalg {

template <typename math_t>
void eigDC_legacy(const raft::handle_t& handle,
                  const math_t* in,
                  std::size_t n_rows,
                  std::size_t n_cols,
                  math_t* eig_vectors,
                  math_t* eig_vals,
                  cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  int lwork;
  RAFT_CUSOLVER_TRY(cusolverDnsyevd_bufferSize(cusolverH,
                                               CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_UPPER,
                                               n_rows,
                                               in,
                                               n_cols,
                                               eig_vals,
                                               &lwork));

  rmm::device_uvector<math_t> d_work(lwork, stream);
  rmm::device_scalar<int> d_dev_info(stream);

  raft::matrix::copy(in, eig_vectors, n_rows, n_cols, stream);

  RAFT_CUSOLVER_TRY(cusolverDnsyevd(cusolverH,
                                    CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_UPPER,
                                    n_rows,
                                    eig_vectors,
                                    n_cols,
                                    eig_vals,
                                    d_work.data(),
                                    lwork,
                                    d_dev_info.data(),
                                    stream));
  RAFT_CHECK_CUDA(cudaGetLastError());

  auto dev_info = d_dev_info.value(stream);
  ASSERT(dev_info == 0,
         "eig.cuh: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");
}

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
void eigDC(const raft::handle_t& handle,
           const math_t* in,
           std::size_t n_rows,
           std::size_t n_cols,
           math_t* eig_vectors,
           math_t* eig_vals,
           cudaStream_t stream)
{
#if CUDART_VERSION < 11010
  eigDC_legacy(handle, in, n_rows, n_cols, eig_vectors, eig_vals, stream);
#else
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  cusolverDnParams_t dn_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateParams(&dn_params));

  size_t workspaceDevice = 0;
  size_t workspaceHost   = 0;
  RAFT_CUSOLVER_TRY(cusolverDnxsyevd_bufferSize(cusolverH,
                                                dn_params,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUBLAS_FILL_MODE_UPPER,
                                                static_cast<int64_t>(n_rows),
                                                eig_vectors,
                                                static_cast<int64_t>(n_cols),
                                                eig_vals,
                                                &workspaceDevice,
                                                &workspaceHost,
                                                stream));

  rmm::device_uvector<math_t> d_work(workspaceDevice / sizeof(math_t), stream);
  rmm::device_scalar<int> d_dev_info(stream);
  std::vector<math_t> h_work(workspaceHost / sizeof(math_t));

  raft::matrix::copy(in, eig_vectors, n_rows, n_cols, stream);

  RAFT_CUSOLVER_TRY(cusolverDnxsyevd(cusolverH,
                                     dn_params,
                                     CUSOLVER_EIG_MODE_VECTOR,
                                     CUBLAS_FILL_MODE_UPPER,
                                     static_cast<int64_t>(n_rows),
                                     eig_vectors,
                                     static_cast<int64_t>(n_cols),
                                     eig_vals,
                                     d_work.data(),
                                     workspaceDevice,
                                     h_work.data(),
                                     workspaceHost,
                                     d_dev_info.data(),
                                     stream));

  RAFT_CHECK_CUDA(cudaGetLastError());
  RAFT_CUSOLVER_TRY(cusolverDnDestroyParams(dn_params));
  int dev_info = d_dev_info.value(stream);
  ASSERT(dev_info == 0,
         "eig.cuh: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");
#endif
}

enum EigVecMemUsage { OVERWRITE_INPUT, COPY_INPUT };

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
void eigSelDC(const raft::handle_t& handle,
              math_t* in,
              int n_rows,
              int n_cols,
              int n_eig_vals,
              math_t* eig_vectors,
              math_t* eig_vals,
              EigVecMemUsage memUsage,
              cudaStream_t stream)
{
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  int lwork;
  int h_meig;

  RAFT_CUSOLVER_TRY(cusolverDnsyevdx_bufferSize(cusolverH,
                                                CUSOLVER_EIG_MODE_VECTOR,
                                                CUSOLVER_EIG_RANGE_I,
                                                CUBLAS_FILL_MODE_UPPER,
                                                n_rows,
                                                in,
                                                n_cols,
                                                math_t(0.0),
                                                math_t(0.0),
                                                n_cols - n_eig_vals + 1,
                                                n_cols,
                                                &h_meig,
                                                eig_vals,
                                                &lwork));

  rmm::device_uvector<math_t> d_work(lwork, stream);
  rmm::device_scalar<int> d_dev_info(stream);
  rmm::device_uvector<math_t> d_eig_vectors(0, stream);

  if (memUsage == OVERWRITE_INPUT) {
    RAFT_CUSOLVER_TRY(cusolverDnsyevdx(cusolverH,
                                       CUSOLVER_EIG_MODE_VECTOR,
                                       CUSOLVER_EIG_RANGE_I,
                                       CUBLAS_FILL_MODE_UPPER,
                                       n_rows,
                                       in,
                                       n_cols,
                                       math_t(0.0),
                                       math_t(0.0),
                                       n_cols - n_eig_vals + 1,
                                       n_cols,
                                       &h_meig,
                                       eig_vals,
                                       d_work.data(),
                                       lwork,
                                       d_dev_info.data(),
                                       stream));
  } else if (memUsage == COPY_INPUT) {
    d_eig_vectors.resize(n_rows * n_cols, stream);
    raft::matrix::copy(in, d_eig_vectors.data(), n_rows, n_cols, stream);

    RAFT_CUSOLVER_TRY(cusolverDnsyevdx(cusolverH,
                                       CUSOLVER_EIG_MODE_VECTOR,
                                       CUSOLVER_EIG_RANGE_I,
                                       CUBLAS_FILL_MODE_UPPER,
                                       n_rows,
                                       eig_vectors,
                                       n_cols,
                                       math_t(0.0),
                                       math_t(0.0),
                                       n_cols - n_eig_vals + 1,
                                       n_cols,
                                       &h_meig,
                                       eig_vals,
                                       d_work.data(),
                                       lwork,
                                       d_dev_info.data(),
                                       stream));
  }

  RAFT_CHECK_CUDA(cudaGetLastError());

  int dev_info = d_dev_info.value(stream);
  ASSERT(dev_info == 0,
         "eig.cuh: eigensolver couldn't converge to a solution. "
         "This usually occurs when some of the features do not vary enough.");

  if (memUsage == OVERWRITE_INPUT) {
    raft::matrix::truncZeroOrigin(in, n_rows, eig_vectors, n_rows, n_eig_vals, stream);
  } else if (memUsage == COPY_INPUT) {
    raft::matrix::truncZeroOrigin(
      d_eig_vectors.data(), n_rows, eig_vectors, n_rows, n_eig_vals, stream);
  }
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
void eigJacobi(const raft::handle_t& handle,
               const math_t* in,
               std::size_t n_rows,
               std::size_t n_cols,
               math_t* eig_vectors,
               math_t* eig_vals,
               cudaStream_t stream,
               math_t tol           = 1.e-7,
               std::uint32_t sweeps = 15)
{
  cusolverDnHandle_t cusolverH = handle.get_cusolver_dn_handle();

  syevjInfo_t syevj_params = nullptr;
  RAFT_CUSOLVER_TRY(cusolverDnCreateSyevjInfo(&syevj_params));
  RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetTolerance(syevj_params, tol));
  RAFT_CUSOLVER_TRY(cusolverDnXsyevjSetMaxSweeps(syevj_params, static_cast<int>(sweeps)));

  int lwork;
  RAFT_CUSOLVER_TRY(cusolverDnsyevj_bufferSize(cusolverH,
                                               CUSOLVER_EIG_MODE_VECTOR,
                                               CUBLAS_FILL_MODE_UPPER,
                                               n_rows,
                                               eig_vectors,
                                               n_cols,
                                               eig_vals,
                                               &lwork,
                                               syevj_params));

  rmm::device_uvector<math_t> d_work(lwork, stream);
  rmm::device_scalar<int> dev_info(stream);

  raft::matrix::copy(in, eig_vectors, n_rows, n_cols, stream);

  RAFT_CUSOLVER_TRY(cusolverDnsyevj(cusolverH,
                                    CUSOLVER_EIG_MODE_VECTOR,
                                    CUBLAS_FILL_MODE_UPPER,
                                    n_rows,
                                    eig_vectors,
                                    n_cols,
                                    eig_vals,
                                    d_work.data(),
                                    lwork,
                                    dev_info.data(),
                                    syevj_params,
                                    stream));

  int executed_sweeps;
  RAFT_CUSOLVER_TRY(cusolverDnXsyevjGetSweeps(cusolverH, syevj_params, &executed_sweeps));

  RAFT_CHECK_CUDA(cudaGetLastError());
  RAFT_CUSOLVER_TRY(cusolverDnDestroySyevjInfo(syevj_params));
}

};  // end namespace linalg
};  // end namespace raft
