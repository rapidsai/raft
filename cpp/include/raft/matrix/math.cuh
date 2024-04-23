/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

/**
 * This file is deprecated and will be removed in a future release.
 * Please use versions in individual header files instead.
 */

#ifndef RAFT_HIDE_DEPRECATION_WARNINGS
#pragma message(__FILE__                                                    \
                  " is deprecated and will be removed in a future release." \
                  " Please use versions in individual header files instead.")
#endif

#ifndef __MATH_H
#define __MATH_H

#pragma once

#include "detail/math.cuh"

namespace raft {
namespace matrix {

/**
 * @defgroup MatrixMathOp math operation on the input matrix
 * @{
 */

/**
 * @brief Power of every element in the input matrix
 * @param in: input matrix
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar.
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void power(math_t* in, math_t* out, math_t scalar, int len, cudaStream_t stream)
{
  detail::power(in, out, scalar, len, stream);
}

/**
 * @brief Power of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar.
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void power(math_t* inout, math_t scalar, int len, cudaStream_t stream)
{
  detail::power(inout, scalar, len, stream);
}

/**
 * @brief Power of every element in the input matrix
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void power(math_t* inout, int len, cudaStream_t stream)
{
  detail::power(inout, len, stream);
}

/**
 * @brief Power of every element in the input matrix
 * @param in: input matrix
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @{
 */
template <typename math_t>
void power(math_t* in, math_t* out, int len, cudaStream_t stream)
{
  detail::power(in, out, len, stream);
}

/**
 * @brief Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param set_neg_zero whether to set negative numbers to zero
 */
template <typename math_t, typename IdxType = int>
void seqRoot(math_t* in,
             math_t* out,
             math_t scalar,
             IdxType len,
             cudaStream_t stream,
             bool set_neg_zero = false)
{
  detail::seqRoot(in, out, scalar, len, stream, set_neg_zero);
}

/**
 * @brief Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param set_neg_zero whether to set negative numbers to zero
 */
template <typename math_t, typename IdxType = int>
void seqRoot(
  math_t* inout, math_t scalar, IdxType len, cudaStream_t stream, bool set_neg_zero = false)
{
  detail::seqRoot(inout, scalar, len, stream, set_neg_zero);
}

/**
 * @brief Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t, typename IdxType = int>
void seqRoot(math_t* in, math_t* out, IdxType len, cudaStream_t stream)
{
  detail::seqRoot(in, out, len, stream);
}

/**
 * @brief Square root of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix with in-place results
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t, typename IdxType = int>
void seqRoot(math_t* inout, IdxType len, cudaStream_t stream)
{
  detail::seqRoot(inout, len, stream);
}

/**
 * @brief sets the small values to zero based on a defined threshold
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param out: output matrix. The result is stored in the out matrix
 * @param in: input matrix
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param thres threshold to set values to zero
 */
template <typename math_t, typename IdxType = int>
void setSmallValuesZero(
  math_t* out, const math_t* in, IdxType len, cudaStream_t stream, math_t thres = 1e-15)
{
  detail::setSmallValuesZero(out, in, len, stream, thres);
}

/**
 * @brief sets the small values to zero based on a defined threshold
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param thres: threshold
 */
template <typename math_t, typename IdxType = int>
void setSmallValuesZero(math_t* inout, IdxType len, cudaStream_t stream, math_t thres = 1e-15)
{
  detail::setSmallValuesZero(inout, len, stream, thres);
}

/**
 * @brief Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param setzero round down to zero if the input is less the threshold
 * @param thres the threshold used to forcibly set inputs to zero
 * @{
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t* in,
                math_t* out,
                math_t scalar,
                int len,
                cudaStream_t stream,
                bool setzero = false,
                math_t thres = 1e-15)
{
  detail::reciprocal(in, out, scalar, len, stream, setzero, thres);
}

/**
 * @brief Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix with in-place results
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param setzero round down to zero if the input is less the threshold
 * @param thres the threshold used to forcibly set inputs to zero
 * @{
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t* inout,
                math_t scalar,
                IdxType len,
                cudaStream_t stream,
                bool setzero = false,
                math_t thres = 1e-15)
{
  detail::reciprocal(inout, scalar, len, stream, setzero, thres);
}

/**
 * @brief Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t* inout, IdxType len, cudaStream_t stream)
{
  detail::reciprocal(inout, len, stream);
}

/**
 * @brief Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param in: input matrix and also the result is stored
 * @param out: output matrix. The result is stored in the out matrix
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t* in, math_t* out, IdxType len, cudaStream_t stream)
{
  detail::reciprocal(in, out, len, stream);
}

/**
 * @brief set values to scalar in matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @param out output matrix. The result is stored in the out matrix
 * @param in input matrix
 * @param scalar svalar value
 * @param len number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void setValue(math_t* out, const math_t* in, math_t scalar, int len, cudaStream_t stream = 0)
{
  detail::setValue(out, in, scalar, len, stream);
}

/**
 * @brief ratio of every element over sum of input vector is calculated
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param handle
 * @param src: input matrix
 * @param dest: output matrix. The result is stored in the dest matrix
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t, typename IdxType = int>
void ratio(
  raft::resources const& handle, math_t* src, math_t* dest, IdxType len, cudaStream_t stream)
{
  detail::ratio(handle, src, dest, len, stream);
}

/** @} */

/**
 * @brief Argmin: find the row idx with minimum value for each column
 * @param in: input matrix (column-major)
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param out: output vector of size n_cols
 * @param stream: cuda stream
 */
template <typename math_t, typename out_t, typename idx_t = int>
void argmin(const math_t* in, idx_t n_rows, idx_t n_cols, out_t* out, cudaStream_t stream)
{
  detail::argmin(in, n_rows, n_cols, out, stream);
}

/**
 * @brief Argmax: find the row idx with maximum value for each column
 * @param in: input matrix (column-major)
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param out: output vector of size n_cols
 * @param stream: cuda stream
 */
template <typename math_t, typename out_t, typename idx_t = int>
void argmax(const math_t* in, idx_t n_rows, idx_t n_cols, out_t* out, cudaStream_t stream)
{
  detail::argmax(in, n_rows, n_cols, out, stream);
}

/**
 * @brief sign flip for PCA. This is used to stabilize the sign of column
 * major eigen vectors. Flips the sign if the column has negative |max|.
 * @param inout: input matrix. Result also stored in this parameter
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param stream cuda stream
 */
template <typename math_t>
void signFlip(math_t* inout, int n_rows, int n_cols, cudaStream_t stream)
{
  detail::signFlip(inout, n_rows, n_cols, stream);
}

/**
 * @brief multiply each row or column of matrix with vector
 * @param data input matrix, results are in-place
 * @param vec input vector
 * @param n_row number of rows of input matrix
 * @param n_col number of columns of input matrix
 * @param rowMajor whether matrix is row major
 * @param bcastAlongRows whether to broadcast vector along rows of matrix or columns
 * @param stream cuda stream
 */
template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryMult(Type* data,
                            const Type* vec,
                            IdxType n_row,
                            IdxType n_col,
                            bool rowMajor,
                            bool bcastAlongRows,
                            cudaStream_t stream)
{
  detail::matrixVectorBinaryMult<Type, IdxType, TPB>(
    data, vec, n_row, n_col, rowMajor, bcastAlongRows, stream);
}

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
void matrixVectorBinaryMultSkipZero(Type* data,
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
void matrixVectorBinaryDiv(Type* data,
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
void matrixVectorBinaryDivSkipZero(Type* data,
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
void matrixVectorBinaryAdd(Type* data,
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
void matrixVectorBinarySub(Type* data,
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

};  // end namespace matrix
};  // end namespace raft

#endif
