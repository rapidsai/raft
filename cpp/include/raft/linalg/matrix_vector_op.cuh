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
 * @tparam InElementType the data-type of the input matrices and vectors
 * @tparam LayoutPolicy the layout of input and output (raft::row_major or raft::col_major)
 * @tparam Lambda a device function which represents a binary operator
 * @tparam OutElementType the data-type of the output raft::matrix_view
 * @tparam TPB threads per block of the cuda kernel launched
 * @param handle raft::handle_t
 * @param out output raft::matrix_view
 * @param matrix input raft::matrix_view
 * @param vec vector raft::vector_view
 * @param apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 * @param op the mathematical operation
 */
template <typename InElementType,
          typename LayoutPolicy,
          typename Lambda,
          typename OutElementType = InElementType,
          int TPB                 = 256>
void matrix_vector_op(const raft::handle_t& handle,
                      raft::matrix_view<OutElementType, LayoutPolicy> out,
                      const raft::matrix_view<InElementType, LayoutPolicy> matrix,
                      const raft::vector_view<InElementType> vec,
                      Apply apply,
                      Lambda op)
{
  static_assert(
    std::is_same_v<typename decltype(out)::layout_type, typename decltype(matrix)::layout_type>,
    "Layout mismatch between Input and Output");
  RAFT_EXPECTS(out.is_contiguous(), "Output must be contiguous");
  RAFT_EXPECTS(matrix.is_contiguous(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == matrix.size(), "Size mismatch between Output and Input");

  auto constexpr rowMajor = std::is_same_v<typename decltype(out)::layout_type, raft::row_major>;
  auto bcastAlongRows     = apply == Apply::ALONG_ROWS;

  if (bcastAlongRows) {
    RAFT_EXPECTS(out.extent(1) == vec.size(), "Size mismatch between matrix and vector");
  } else {
    RAFT_EXPECTS(out.extent(0) == vec.size(), "Size mismatch between matrix and vector");
  }

  matrixVectorOp(out.data(),
                 matrix.data(),
                 vec.data(),
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
 * @tparam InElementType the data-type of the input matrices and vectors
 * @tparam LayoutPolicy the layout of input and output (raft::row_major or raft::col_major)
 * @tparam Lambda a device function which represents a binary operator
 * @tparam OutElementType the data-type of the output raft::matrix_view
 * @tparam TPB threads per block of the cuda kernel launched
 * @param handle raft::handle_t
 * @param out output raft::matrix_view
 * @param matrix input raft::matrix_view
 * @param vec1 the first vector raft::vector_view
 * @param vec2 the second vector raft::vector_view
 * @param apply whether the broadcast of vector needs to happen along
 * the rows of the matrix or columns using enum class raft::linalg::Apply
 * @param op the mathematical operation
 */
template <typename InElementType,
          typename LayoutPolicy,
          typename Lambda,
          typename OutElementType = InElementType,
          int TPB                 = 256>
void matrix_vector_op(const raft::handle_t& handle,
                      raft::matrix_view<OutElementType, LayoutPolicy> out,
                      const raft::matrix_view<InElementType, LayoutPolicy> matrix,
                      const raft::vector_view<InElementType> vec1,
                      const raft::vector_view<InElementType> vec2,
                      Apply apply,
                      Lambda op)
{
  static_assert(
    std::is_same_v<typename decltype(out)::layout_type, typename decltype(matrix)::layout_type>,
    "Layout mismatch between Input and Output");
  RAFT_EXPECTS(out.is_contiguous(), "Output must be contiguous");
  RAFT_EXPECTS(matrix.is_contiguous(), "Input must be contiguous");
  RAFT_EXPECTS(out.size() == matrix.size(), "Size mismatch between Output and Input");

  auto constexpr rowMajor = std::is_same_v<typename decltype(out)::layout_type, raft::row_major>;
  auto bcastAlongRows     = apply == Apply::ALONG_ROWS;

  if (bcastAlongRows) {
    RAFT_EXPECTS(out.extent(1) == vec1.size(), "Size mismatch between matrix and vector");
    RAFT_EXPECTS(out.extent(1) == vec2.size(), "Size mismatch between matrix and vector");
  } else {
    RAFT_EXPECTS(out.extent(0) == vec1.size(), "Size mismatch between matrix and vector");
    RAFT_EXPECTS(out.extent(0) == vec2.size(), "Size mismatch between matrix and vector");
  }

  matrixVectorOp(out.data(),
                 matrix.data(),
                 vec1.data(),
                 vec2.data(),
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