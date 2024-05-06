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

#ifndef __MATRIX_H
#define __MATRIX_H

#pragma once

#include "detail/linewise_op.cuh"
#include "detail/matrix.cuh"

#include <raft/common/nvtx.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

namespace raft {
namespace matrix {

using namespace std;

/**
 * @brief Copy selected rows of the input matrix into contiguous space.
 *
 * On exit out[i + k*n_rows] = in[indices[i] + k*n_rows],
 * where i = 0..n_rows_indices-1, and k = 0..n_cols-1.
 *
 * @param in input matrix
 * @param n_rows number of rows of output matrix
 * @param n_cols number of columns of output matrix
 * @param out output matrix
 * @param indices of the rows to be copied
 * @param n_rows_indices number of rows to copy
 * @param stream cuda stream
 * @param rowMajor whether the matrix has row major layout
 */
template <typename m_t, typename idx_array_t = int, typename idx_t = size_t>
void copyRows(const m_t* in,
              idx_t n_rows,
              idx_t n_cols,
              m_t* out,
              const idx_array_t* indices,
              idx_t n_rows_indices,
              cudaStream_t stream,
              bool rowMajor = false)
{
  detail::copyRows(in, n_rows, n_cols, out, indices, n_rows_indices, stream, rowMajor);
}

/**
 * @brief copy matrix operation for column major matrices.
 * @param in: input matrix
 * @param out: output matrix
 * @param n_rows: number of rows of output matrix
 * @param n_cols: number of columns of output matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void copy(const m_t* in, m_t* out, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  raft::copy_async(out, in, n_rows * n_cols, stream);
}

/**
 * @brief copy matrix operation for column major matrices.
 * @param[in] handle: raft handle
 * @param[in] in: input matrix
 * @param[out] out: output matrix
 */
template <typename m_t, typename idx_t = int, typename matrix_idx_t>
void copy(raft::resources const& handle,
          raft::device_matrix_view<const m_t, matrix_idx_t, col_major> in,
          raft::device_matrix_view<m_t, matrix_idx_t, col_major> out)
{
  RAFT_EXPECTS(in.extent(0) == out.extent(0) && in.extent(1) == out.extent(1),
               "Input and output matrix shapes must match.");

  raft::copy_async(out.data_handle(),
                   in.data_handle(),
                   in.extent(0) * out.extent(1),
                   resource::get_cuda_stream(handle));
}

/**
 * @brief copy matrix operation for column major matrices. First n_rows and
 * n_cols of input matrix "in" is copied to "out" matrix.
 * @param in: input matrix
 * @param in_n_rows: number of rows of input matrix
 * @param out: output matrix
 * @param out_n_rows: number of rows of output matrix
 * @param out_n_cols: number of columns of output matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void truncZeroOrigin(
  m_t* in, idx_t in_n_rows, m_t* out, idx_t out_n_rows, idx_t out_n_cols, cudaStream_t stream)
{
  detail::truncZeroOrigin(in, in_n_rows, out, out_n_rows, out_n_cols, stream);
}

/**
 * @brief Columns of a column major matrix is reversed (i.e. first column and
 * last column are swapped)
 * @param inout: input and output matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void colReverse(m_t* inout, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  detail::colReverse(inout, n_rows, n_cols, stream);
}

/**
 * @brief Rows of a column major matrix is reversed (i.e. first row and last
 * row are swapped)
 * @param inout: input and output matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void rowReverse(m_t* inout, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  detail::rowReverse(inout, n_rows, n_cols, stream);
}

/**
 * @brief Prints the data stored in GPU memory
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param h_separator: horizontal separator character
 * @param v_separator: vertical separator character
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void print(const m_t* in,
           idx_t n_rows,
           idx_t n_cols,
           char h_separator    = ' ',
           char v_separator    = '\n',
           cudaStream_t stream = rmm::cuda_stream_default)
{
  detail::print(in, n_rows, n_cols, h_separator, v_separator, stream);
}

/**
 * @brief Prints the data stored in CPU memory
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 */
template <typename m_t, typename idx_t = int>
void printHost(const m_t* in, idx_t n_rows, idx_t n_cols)
{
  detail::printHost(in, n_rows, n_cols);
}

/**
 * @brief Slice a matrix (in-place)
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param out: output matrix
 * @param x1, y1: coordinate of the top-left point of the wanted area (0-based)
 * @param x2, y2: coordinate of the bottom-right point of the wanted area
 * (1-based)
 * example: Slice the 2nd and 3rd columns of a 4x3 matrix: slice_matrix(M_d, 4,
 * 3, 0, 1, 4, 3);
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void sliceMatrix(m_t* in,
                 idx_t n_rows,
                 idx_t n_cols,
                 m_t* out,
                 idx_t x1,
                 idx_t y1,
                 idx_t x2,
                 idx_t y2,
                 cudaStream_t stream)
{
  detail::sliceMatrix(in, n_rows, n_cols, out, x1, y1, x2, y2, false, stream);
}

/**
 * @brief Copy the upper triangular part of a matrix to another
 * @param src: input matrix with a size of n_rows x n_cols
 * @param dst: output matrix with a size of kxk, k = min(n_rows, n_cols)
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void copyUpperTriangular(m_t* src, m_t* dst, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  detail::copyUpperTriangular(src, dst, n_rows, n_cols, stream);
}

/**
 * @brief Initialize a diagonal col-major matrix with a vector
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols (col-major)
 * @param n_rows: number of rows of the matrix
 * @param n_cols: number of columns of the matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void initializeDiagonalMatrix(
  m_t* vec, m_t* matrix, idx_t n_rows, idx_t n_cols, cudaStream_t stream)
{
  detail::initializeDiagonalMatrix(vec, matrix, n_rows, n_cols, false, stream);
}

/**
 * @brief Get a square matrix with elements on diagonal reversed (in-place)
 * @param in: square input matrix with size len x len
 * @param len: size of one side of the matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void getDiagonalInverseMatrix(m_t* in, idx_t len, cudaStream_t stream)
{
  detail::getDiagonalInverseMatrix(in, len, stream);
}

/**
 * @brief Get the L2/F-norm of a matrix/vector
 * @param handle
 * @param in: input matrix/vector with totally size elements
 * @param size: size of the matrix/vector
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
m_t getL2Norm(raft::resources const& handle, m_t* in, idx_t size, cudaStream_t stream)
{
  return detail::getL2Norm(handle, in, size, stream);
}

/**
 * Run a function over matrix lines (rows or columns) with a variable number
 * row-vectors or column-vectors.
 * The term `line` here signifies that the lines can be either columns or rows,
 * depending on the matrix layout.
 * What matters is if the vectors are applied along lines (indices of vectors correspond to
 * indices within lines), or across lines (indices of vectors correspond to line numbers).
 *
 * @param [out] out result of the operation; can be same as `in`; should be aligned the same
 *        as `in` to allow faster vectorized memory transfers.
 * @param [in] in input matrix consisting of `nLines` lines, each `lineLen`-long.
 * @param [in] lineLen length of matrix line in elements (`=nCols` in row-major or `=nRows` in
 * col-major)
 * @param [in] nLines number of matrix lines (`=nRows` in row-major or `=nCols` in col-major)
 * @param [in] alongLines whether vectors are indices along or across lines.
 * @param [in] op the operation applied on each line:
 *    for i in [0..lineLen) and j in [0..nLines):
 *      out[i, j] = op(in[i, j], vec1[i], vec2[i], ... veck[i])   if alongLines = true
 *      out[i, j] = op(in[i, j], vec1[j], vec2[j], ... veck[j])   if alongLines = false
 *    where matrix indexing is row-major ([i, j] = [i + lineLen * j]).
 * @param [in] stream a cuda stream for the kernels
 * @param [in] vecs zero or more vectors to be passed as arguments,
 *    size of each vector is `alongLines ? lineLen : nLines`.
 */
template <typename m_t, typename idx_t = int, typename Lambda, typename... Vecs>
void linewiseOp(m_t* out,
                const m_t* in,
                const idx_t lineLen,
                const idx_t nLines,
                const bool alongLines,
                Lambda op,
                cudaStream_t stream,
                const Vecs*... vecs)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope("linewiseOp-%c-%zu (%zu, %zu)",
                                                            alongLines ? 'l' : 'x',
                                                            sizeof...(Vecs),
                                                            size_t(lineLen),
                                                            size_t(nLines));
  detail::MatrixLinewiseOp<16, 256>::run<m_t, idx_t, Lambda, Vecs...>(
    out, in, lineLen, nLines, alongLines, op, stream, vecs...);
}

};  // end namespace matrix
};  // end namespace raft

#endif
