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

#include "detail/gemm.hpp"

namespace raft {
namespace linalg {

/**
 * @brief the wrapper of cublas gemm function
 *  It computes the following equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam math_t the type of input/output matrices
 * @param handle raft handle
 * @param a input matrix
 * @param n_rows_a number of rows of A
 * @param n_cols_a number of columns of A
 * @param b input matrix
 * @param c output matrix
 * @param n_rows_c number of rows of C
 * @param n_cols_c number of columns of C
 * @param trans_a cublas transpose op for A
 * @param trans_b cublas transpose op for B
 * @param alpha scalar
 * @param beta scalar
 * @param stream cuda stream
 */
template <typename math_t>
void gemm(const raft::handle_t& handle,
          const math_t* a,
          int n_rows_a,
          int n_cols_a,
          const math_t* b,
          math_t* c,
          int n_rows_c,
          int n_cols_c,
          cublasOperation_t trans_a,
          cublasOperation_t trans_b,
          math_t alpha,
          math_t beta,
          cudaStream_t stream)
{
  detail::gemm(
    handle, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, alpha, beta, stream);
}

template <typename math_t>
void gemm(const raft::handle_t& handle,
          const math_t* a,
          int n_rows_a,
          int n_cols_a,
          const math_t* b,
          math_t* c,
          int n_rows_c,
          int n_cols_c,
          cublasOperation_t trans_a,
          cublasOperation_t trans_b,
          cudaStream_t stream)
{
  math_t alpha = math_t(1);
  math_t beta  = math_t(0);
  gemm(
    handle, a, n_rows_a, n_cols_a, b, c, n_rows_c, n_cols_c, trans_a, trans_b, alpha, beta, stream);
}

/**
 * @brief A wrapper for CUBLS GEMM function designed for handling all possible
 * combinations of operand layouts.
 * It computes the following equation: Z = alpha . X * Y + beta . Z
 * @tparam T Data type of input/output matrices (float/double)
 * @param handle raft handle
 * @param z output matrix of size M rows x N columns
 * @param x input matrix of size M rows x K columns
 * @param y input matrix of size K rows x N columns
 * @param _M number of rows of X and Z
 * @param _N number of rows of Y and columns of Z
 * @param _K number of columns of X and rows of Y
 * @param isZColMajor Storage layout of Z. true = col major, false = row major
 * @param isXColMajor Storage layout of X. true = col major, false = row major
 * @param isYColMajor Storage layout of Y. true = col major, false = row major
 * @param stream cuda stream
 * @param alpha scalar
 * @param beta scalar
 */
template <typename T>
void gemm(const raft::handle_t& handle,
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
          T alpha = T(1.0),
          T beta  = T(0.0))
{
  detail::gemm(
    handle, z, x, y, _M, _N, _K, isZColMajor, isXColMajor, isYColMajor, stream, alpha, beta);
}

}  // end namespace linalg
}  // end namespace raft
