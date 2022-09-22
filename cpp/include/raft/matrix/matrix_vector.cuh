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

#include "detail/matrix.cuh"
#include <raft/core/device_mdspan.hpp>

namespace raft::matrix {

/**
 * @brief multiply each row or column of matrix with vector, skipping zeros in vector
 * @param data input matrix, results are in-place
 * @param vec input vector
 * @param n_row number of rows of input matrix
 * @param n_col number of columns of input matrix
 * @param rowMajor whether matrix is row major
 * @param bcastAlongRows whether to broadcast vector along rows of matrix or columns
 * @param stream cuda stream
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void binary_mult_skip_zero(Type* data,
                           const Type* vec,
                           IdxType n_row,
                           IdxType n_col,
                           bool rowMajor,
                           bool bcastAlongRows,
                           cudaStream_t stream)
{
  detail::matrixVectorBinaryMultSkipZero<Type, IdxType, TPB>(
    data, vec, n_row, n_col, rowMajor, bcastAlongRows, stream);
}

/**
 * @brief divide each row or column of matrix with vector
 * @param data input matrix, results are in-place
 * @param vec input vector
 * @param n_row number of rows of input matrix
 * @param n_col number of columns of input matrix
 * @param rowMajor whether matrix is row major
 * @param bcastAlongRows whether to broadcast vector along rows of matrix or columns
 * @param stream cuda stream
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void binary_div(Type* data,
                const Type* vec,
                IdxType n_row,
                IdxType n_col,
                bool rowMajor,
                bool bcastAlongRows,
                cudaStream_t stream)
{
  detail::matrixVectorBinaryDiv<Type, IdxType, TPB>(
    data, vec, n_row, n_col, rowMajor, bcastAlongRows, stream);
}

/**
 * @brief divide each row or column of matrix with vector, skipping zeros in vector
 * @param data input matrix, results are in-place
 * @param vec input vector
 * @param n_row number of rows of input matrix
 * @param n_col number of columns of input matrix
 * @param rowMajor whether matrix is row major
 * @param bcastAlongRows whether to broadcast vector along rows of matrix or columns
 * @param stream cuda stream
 * @param return_zero result is zero if true and vector value is below threshold, original value if
 * false
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void binary_div_skip_zero(Type* data,
                          const Type* vec,
                          IdxType n_row,
                          IdxType n_col,
                          bool rowMajor,
                          bool bcastAlongRows,
                          cudaStream_t stream,
                          bool return_zero = false)
{
  detail::matrixVectorBinaryDivSkipZero<Type, IdxType, TPB>(
    data, vec, n_row, n_col, rowMajor, bcastAlongRows, stream, return_zero);
}

/**
 * @brief add each row or column of matrix with vector
 * @param data input matrix, results are in-place
 * @param vec input vector
 * @param n_row number of rows of input matrix
 * @param n_col number of columns of input matrix
 * @param rowMajor whether matrix is row major
 * @param bcastAlongRows whether to broadcast vector along rows of matrix or columns
 * @param stream cuda stream
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void binary_add(Type* data,
                const Type* vec,
                IdxType n_row,
                IdxType n_col,
                bool rowMajor,
                bool bcastAlongRows,
                cudaStream_t stream)
{
  detail::matrixVectorBinaryAdd<Type, IdxType, TPB>(
    data, vec, n_row, n_col, rowMajor, bcastAlongRows, stream);
}

/**
 * @brief subtract each row or column of matrix with vector
 * @param data input matrix, results are in-place
 * @param vec input vector
 * @param n_row number of rows of input matrix
 * @param n_col number of columns of input matrix
 * @param rowMajor whether matrix is row major
 * @param bcastAlongRows whether to broadcast vector along rows of matrix or columns
 * @param stream cuda stream
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void binary_sub(Type* data,
                const Type* vec,
                IdxType n_row,
                IdxType n_col,
                bool rowMajor,
                bool bcastAlongRows,
                cudaStream_t stream)
{
  detail::matrixVectorBinarySub<Type, IdxType, TPB>(
    data, vec, n_row, n_col, rowMajor, bcastAlongRows, stream);
}

}  // namespace raft::matrix