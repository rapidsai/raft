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

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                    \
                  " is deprecated and will be removed in a future release." \
                  " Use cublaslt_wrappers.hpp if you really need this low-level api.")
#endif

#include "cublaslt_wrappers.hpp"

#include <raft/core/resources.hpp>

namespace raft::linalg::detail {

template <typename T, bool DevicePointerMode = false>
void legacy_gemm(raft::resources const& res,
                 const bool trans_a,
                 const bool trans_b,
                 const int m,
                 const int n,
                 const int k,
                 const T* alpha,
                 const T* A,
                 const int lda,
                 const T* B,
                 const int ldb,
                 const T* beta,
                 T* C,
                 const int ldc,
                 cudaStream_t stream)
{
  return legacy_matmul<DevicePointerMode, T, T, T, T>(res,
                                                      trans_a,
                                                      trans_b,
                                                      static_cast<uint64_t>(m),
                                                      static_cast<uint64_t>(n),
                                                      static_cast<uint64_t>(k),
                                                      alpha,
                                                      A,
                                                      static_cast<uint64_t>(lda),
                                                      B,
                                                      static_cast<uint64_t>(ldb),
                                                      beta,
                                                      C,
                                                      static_cast<uint64_t>(ldc),
                                                      stream);
}

template <typename T>
void legacy_gemm(raft::resources const& res,
                 const T* a,
                 int n_rows_a,
                 int n_cols_a,
                 const T* b,
                 T* c,
                 int n_rows_c,
                 int n_cols_c,
                 cublasOperation_t trans_a,
                 cublasOperation_t trans_b,
                 T alpha,
                 T beta,
                 cudaStream_t stream)
{
  int m  = n_rows_c;
  int n  = n_cols_c;
  auto k = trans_a == CUBLAS_OP_T ? n_rows_a : n_cols_a;
  return legacy_matmul<false, T, T, T, T>(res,
                                          trans_a == CUBLAS_OP_T,
                                          trans_b == CUBLAS_OP_T,
                                          static_cast<uint64_t>(n_rows_c),
                                          static_cast<uint64_t>(n_cols_c),
                                          static_cast<uint64_t>(k),
                                          &alpha,
                                          a,
                                          static_cast<uint64_t>(trans_a == CUBLAS_OP_T ? k : m),
                                          b,
                                          static_cast<uint64_t>(trans_b == CUBLAS_OP_T ? n : k),
                                          &beta,
                                          c,
                                          static_cast<uint64_t>(m),
                                          stream);
}

template <typename T>
void legacy_gemm(raft::resources const& res,
                 const T* a,
                 int n_rows_a,
                 int n_cols_a,
                 const T* b,
                 T* c,
                 int n_rows_c,
                 int n_cols_c,
                 cublasOperation_t trans_a,
                 cublasOperation_t trans_b,
                 cudaStream_t stream)
{
  return legacy_gemm(
    res, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, T{1}, T{0}, stream);
}

template <typename T, bool DevicePointerMode = false>
void legacy_gemm(raft::resources const& res,
                 T* z,
                 T* x,
                 T* y,
                 int _M,
                 int _N,
                 int _K,
                 bool isZColMajor,
                 bool isXColMajor,
                 bool isYColMajor,
                 cudaStream_t stream,
                 const T* alpha,
                 const T* beta)
{
  if (isZColMajor) {
    return legacy_matmul<DevicePointerMode, T, T, T, T>(
      res,
      !isXColMajor,
      !isYColMajor,
      static_cast<uint64_t>(_M),
      static_cast<uint64_t>(_N),
      static_cast<uint64_t>(_K),
      alpha,
      x,
      static_cast<uint64_t>(isXColMajor ? _M : _K),
      y,
      static_cast<uint64_t>(isYColMajor ? _K : _N),
      beta,
      z,
      static_cast<uint64_t>(_M),
      stream);
  } else {
    return legacy_gemm<T, DevicePointerMode>(
      res, z, y, x, _N, _M, _K, true, !isYColMajor, !isXColMajor, stream, alpha, beta);
  }
}

}  // namespace raft::linalg::detail
