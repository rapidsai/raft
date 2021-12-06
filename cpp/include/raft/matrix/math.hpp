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

#include "detail/math.cuh"

#include <raft/handle.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/map_then_reduce.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/unary_op.cuh>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

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
  auto d_src  = in;
  auto d_dest = out;

  raft::linalg::binaryOp(
    d_dest,
    d_src,
    d_src,
    len,
    [=] __device__(math_t a, math_t b) { return scalar * a * b; },
    stream);
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
  power(inout, inout, scalar, len, stream);
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
  math_t scalar = 1.0;
  power(inout, scalar, len, stream);
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
  math_t scalar = 1.0;
  power(in, out, scalar, len, stream);
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
  auto d_src  = in;
  auto d_dest = out;

  raft::linalg::unaryOp(
    d_dest,
    d_src,
    len,
    [=] __device__(math_t a) {
      if (set_neg_zero) {
        if (a < math_t(0)) {
          return math_t(0);
        } else {
          return sqrt(a * scalar);
        }
      } else {
        return sqrt(a * scalar);
      }
    },
    stream);
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
  seqRoot(inout, inout, scalar, len, stream, set_neg_zero);
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
  math_t scalar = 1.0;
  seqRoot(in, out, scalar, len, stream);
}

template <typename math_t, typename IdxType = int>
void seqRoot(math_t* inout, IdxType len, cudaStream_t stream)
{
  math_t scalar = 1.0;
  seqRoot(inout, inout, scalar, len, stream);
}

template <typename math_t, typename IdxType = int>
void setSmallValuesZero(
  math_t* out, const math_t* in, IdxType len, cudaStream_t stream, math_t thres = 1e-15)
{
  raft::linalg::unaryOp(
    out,
    in,
    len,
    [=] __device__(math_t a) {
      if (a <= thres && -a <= thres) {
        return math_t(0);
      } else {
        return a;
      }
    },
    stream);
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
  setSmallValuesZero(inout, inout, len, stream, thres);
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
  auto d_src  = in;
  auto d_dest = out;

  raft::linalg::unaryOp(
    d_dest,
    d_src,
    len,
    [=] __device__(math_t a) {
      if (setzero) {
        if (abs(a) <= thres) {
          return math_t(0);
        } else {
          return scalar / a;
        }
      } else {
        return scalar / a;
      }
    },
    stream);
}

/**
 * @brief Reciprocal of every element in the input matrix
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param inout: input matrix and also the result is stored
 * @param scalar: every element is multiplied with scalar
 * @param len: number elements of input matrix
 * @param stream cuda stream
 * @param setzero: (default false) when true and |value|<thres, avoid dividing by (almost) zero
 * @param thres: Threshold to avoid dividing by zero (|value| < thres -> result = 0)
 */
template <typename math_t, typename IdxType = int>
void reciprocal(math_t* inout,
                math_t scalar,
                IdxType len,
                cudaStream_t stream,
                bool setzero = false,
                math_t thres = 1e-15)
{
  reciprocal(inout, inout, scalar, len, stream, setzero, thres);
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
  math_t scalar = 1.0;
  reciprocal(inout, scalar, len, stream);
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
  math_t scalar = 1.0;
  reciprocal(in, out, scalar, len, stream);
}

template <typename math_t>
void setValue(math_t* out, const math_t* in, math_t scalar, int len, cudaStream_t stream = 0)
{
  raft::linalg::unaryOp(
    out, in, len, [scalar] __device__(math_t in) { return scalar; }, stream);
}

/**
 * @brief ratio of every element over sum of input vector is calculated
 * @tparam math_t data-type upon which the math operation will be performed
 * @tparam IdxType Integer type used to for addressing
 * @param src: input matrix
 * @param dest: output matrix. The result is stored in the dest matrix
 * @param len: number elements of input matrix
 * @param stream cuda stream
 */
template <typename math_t, typename IdxType = int>
void ratio(
  const raft::handle_t& handle, math_t* src, math_t* dest, IdxType len, cudaStream_t stream)
{
  auto d_src  = src;
  auto d_dest = dest;

  rmm::device_scalar<math_t> d_sum(stream);
  auto* d_sum_ptr = d_sum.data();
  auto no_op      = [] __device__(math_t in) { return in; };
  raft::linalg::mapThenSumReduce(d_sum_ptr, len, no_op, stream, src);
  raft::linalg::unaryOp(
    d_dest, d_src, len, [=] __device__(math_t a) { return a / (*d_sum_ptr); }, stream);
}

/** @} */

/**
 * @brief Argmax: find the row idx with maximum value for each column
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param out: output vector of size n_cols
 * @param stream: cuda stream
 */
template <typename math_t>
void argmax(const math_t* in, int n_rows, int n_cols, math_t* out, cudaStream_t stream)
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

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryMult(Type* data,
                            const Type* vec,
                            IdxType n_row,
                            IdxType n_col,
                            bool rowMajor,
                            bool bcastAlongRows,
                            cudaStream_t stream)
{
  raft::linalg::matrixVectorOp(
    data,
    data,
    vec,
    n_col,
    n_row,
    rowMajor,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a * b; },
    stream);
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryMultSkipZero(Type* data,
                                    const Type* vec,
                                    IdxType n_row,
                                    IdxType n_col,
                                    bool rowMajor,
                                    bool bcastAlongRows,
                                    cudaStream_t stream)
{
  raft::linalg::matrixVectorOp(
    data,
    data,
    vec,
    n_col,
    n_row,
    rowMajor,
    bcastAlongRows,
    [] __device__(Type a, Type b) {
      if (b == Type(0))
        return a;
      else
        return a * b;
    },
    stream);
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryDiv(Type* data,
                           const Type* vec,
                           IdxType n_row,
                           IdxType n_col,
                           bool rowMajor,
                           bool bcastAlongRows,
                           cudaStream_t stream)
{
  raft::linalg::matrixVectorOp(
    data,
    data,
    vec,
    n_col,
    n_row,
    rowMajor,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a / b; },
    stream);
}

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
  if (return_zero) {
    raft::linalg::matrixVectorOp(
      data,
      data,
      vec,
      n_col,
      n_row,
      rowMajor,
      bcastAlongRows,
      [] __device__(Type a, Type b) {
        if (raft::myAbs(b) < Type(1e-10))
          return Type(0);
        else
          return a / b;
      },
      stream);
  } else {
    raft::linalg::matrixVectorOp(
      data,
      data,
      vec,
      n_col,
      n_row,
      rowMajor,
      bcastAlongRows,
      [] __device__(Type a, Type b) {
        if (raft::myAbs(b) < Type(1e-10))
          return a;
        else
          return a / b;
      },
      stream);
  }
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinaryAdd(Type* data,
                           const Type* vec,
                           IdxType n_row,
                           IdxType n_col,
                           bool rowMajor,
                           bool bcastAlongRows,
                           cudaStream_t stream)
{
  raft::linalg::matrixVectorOp(
    data,
    data,
    vec,
    n_col,
    n_row,
    rowMajor,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a + b; },
    stream);
}

template <typename Type, typename IdxType = int, int TPB = 256>
void matrixVectorBinarySub(Type* data,
                           const Type* vec,
                           IdxType n_row,
                           IdxType n_col,
                           bool rowMajor,
                           bool bcastAlongRows,
                           cudaStream_t stream)
{
  raft::linalg::matrixVectorOp(
    data,
    data,
    vec,
    n_col,
    n_row,
    rowMajor,
    bcastAlongRows,
    [] __device__(Type a, Type b) { return a - b; },
    stream);
}

};  // end namespace matrix
};  // end namespace raft
