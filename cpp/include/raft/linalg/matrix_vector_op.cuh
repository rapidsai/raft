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
#ifndef __MATRIX_VECTOR_OP_H
#define __MATRIX_VECTOR_OP_H

#pragma once

#include "apply.hpp"
#include "detail/matrix_vector_op.cuh"

#include <raft/core/mdarray.hpp>

namespace raft {
namespace linalg {

/**
 * @brief Operations for all the columns or rows with a given vector.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when adresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam Type the matrix/vector type
 * @tparam Lambda a device function which represents a binary operator
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output matrix (passing out = matrix makes it in-place)
 * @param matrix the input matrix
 * @param vec the vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns
 * @param op the mathematical operation
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda, typename IdxType = int, int TPB = 256>
void matrixVectorOp(Type* out,
                    const Type* matrix,
                    const Type* vec,
                    IdxType D,
                    IdxType N,
                    bool rowMajor,
                    bool bcastAlongRows,
                    Lambda op,
                    cudaStream_t stream)
{
  detail::matrixVectorOp(out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
}

/**
 * @brief Operations for all the columns or rows with the given vectors.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when adresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam Type the matrix/vector type
 * @tparam Lambda a device function which represents a binary operator
 * @tparam IdxType Integer type used to for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param out the output matrix (passing out = matrix makes it in-place)
 * @param matrix the input matrix
 * @param vec1 the first vector
 * @param vec2 the second vector
 * @param D number of columns of matrix
 * @param N number of rows of matrix
 * @param rowMajor whether input is row or col major
 * @param bcastAlongRows whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns
 * @param op the mathematical operation
 * @param stream cuda stream where to launch work
 */
template <typename Type, typename Lambda, typename IdxType = int, int TPB = 256>
void matrixVectorOp(Type* out,
                    const Type* matrix,
                    const Type* vec1,
                    const Type* vec2,
                    IdxType D,
                    IdxType N,
                    bool rowMajor,
                    bool bcastAlongRows,
                    Lambda op,
                    cudaStream_t stream)
{
  detail::matrixVectorOp(out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
}

/**
 * @defgroup matrix_vector_op Matrix Vector Operations
 * @{
 */

/**
 * @brief Operations for all the columns or rows with a given vector.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when adresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam InValueType the data-type of the input matrices and vectors
 * @tparam LayoutPolicy the layout of input and output (raft::row_major or raft::col_major)
 * @tparam Lambda a device function which represents a binary operator
 * @tparam OutElementType the data-type of the output raft::matrix_view
 * @tparam IndexType Integer used for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param[in] handle raft::handle_t
 * @param[in] matrix input raft::matrix_view
 * @param[in] vec vector raft::vector_view
 * @param[out] out output raft::matrix_view
 * @param[in] apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 * @param[in] op the mathematical operation
 */
template <typename InValueType,
          typename LayoutPolicy,
          typename Lambda,
          typename OutValueType,
          typename IndexType,
          int TPB = 256>
void matrix_vector_op(const raft::handle_t& handle,
                      raft::device_matrix_view<const InValueType, IndexType, LayoutPolicy> matrix,
                      raft::device_vector_view<const InValueType, IndexType> vec,
                      raft::device_matrix_view<OutValueType, IndexType, LayoutPolicy> out,
                      Apply apply,
                      Lambda op)
{
  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(matrix.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == matrix.size(), "Size mismatch between Output and Input");

  auto constexpr rowMajor = std::is_same_v<typename decltype(out)::layout_type, raft::row_major>;
  auto bcastAlongRows     = apply == Apply::ALONG_ROWS;

  if (bcastAlongRows) {
    RAFT_EXPECTS(out.extent(1) == static_cast<IndexType>(vec.size()),
                 "Size mismatch between matrix and vector");
  } else {
    RAFT_EXPECTS(out.extent(0) == static_cast<IndexType>(vec.size()),
                 "Size mismatch between matrix and vector");
  }

  matrixVectorOp(out.data_handle(),
                 matrix.data_handle(),
                 vec.data_handle(),
                 out.extent(1),
                 out.extent(0),
                 rowMajor,
                 bcastAlongRows,
                 op,
                 handle.get_stream());
}

/**
 * @brief Operations for all the columns or rows with the given vectors.
 * Caution : Threads process multiple elements to speed up processing. These
 * are loaded in a single read thanks to type promotion. Faster processing
 * would thus only be enabled when adresses are optimally aligned for it.
 * Note : the function will also check that the size of the window of accesses
 * is a multiple of the number of elements processed by a thread in order to
 * enable faster processing
 * @tparam InValueType the data-type of the input matrices and vectors
 * @tparam LayoutPolicy the layout of input and output (raft::row_major or raft::col_major)
 * @tparam Lambda a device function which represents a binary operator
 * @tparam OutElementType the data-type of the output raft::matrix_view
 * @tparam IndexType Integer used for addressing
 * @tparam TPB threads per block of the cuda kernel launched
 * @param handle raft::handle_t
 * @param matrix input raft::matrix_view
 * @param vec1 the first vector raft::vector_view
 * @param vec2 the second vector raft::vector_view
 * @param out output raft::matrix_view
 * @param apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 * @param op the mathematical operation
 */
template <typename InValueType,
          typename LayoutPolicy,
          typename Lambda,
          typename OutValueType,
          typename IndexType,
          int TPB = 256>
void matrix_vector_op(const raft::handle_t& handle,
                      raft::device_matrix_view<const InValueType, IndexType, LayoutPolicy> matrix,
                      raft::device_vector_view<const InValueType, IndexType> vec1,
                      raft::device_vector_view<const InValueType, IndexType> vec2,
                      raft::device_matrix_view<OutValueType, IndexType, LayoutPolicy> out,
                      Apply apply,
                      Lambda op)
{
  RAFT_EXPECTS(out.is_exhaustive(), "Output must be contiguous");
  RAFT_EXPECTS(matrix.is_exhaustive(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == matrix.size(), "Size mismatch between Output and Input");

  auto constexpr rowMajor = std::is_same_v<typename decltype(out)::layout_type, raft::row_major>;
  auto bcastAlongRows     = apply == Apply::ALONG_ROWS;

  if (bcastAlongRows) {
    RAFT_EXPECTS(out.extent(1) == static_cast<IndexType>(vec1.size()),
                 "Size mismatch between matrix and vector");
    RAFT_EXPECTS(out.extent(1) == static_cast<IndexType>(vec2.size()),
                 "Size mismatch between matrix and vector");
  } else {
    RAFT_EXPECTS(out.extent(0) == static_cast<IndexType>(vec1.size()),
                 "Size mismatch between matrix and vector");
    RAFT_EXPECTS(out.extent(0) == static_cast<IndexType>(vec2.size()),
                 "Size mismatch between matrix and vector");
  }

  matrixVectorOp(out.data_handle(),
                 matrix.data_handle(),
                 vec1.data_handle(),
                 vec2.data_handle(),
                 out.extent(1),
                 out.extent(0),
                 rowMajor,
                 bcastAlongRows,
                 op,
                 handle.get_stream());
}

/** @} */  // end of group matrix_vector_op

};  // end namespace linalg
};  // end namespace raft

#endif