/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/matrix/matrix.hpp>

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
  IdxType stride = rowMajor ? D : N;
  IdxType nLines = rowMajor ? N : D;
  return matrix::linewiseOp(
    out, matrix, stride, nLines, rowMajor == bcastAlongRows, op, stream, vec);
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
  IdxType stride = rowMajor ? D : N;
  IdxType nLines = rowMajor ? N : D;
  return matrix::linewiseOp(
    out, matrix, stride, nLines, rowMajor == bcastAlongRows, op, stream, vec1, vec2);
}

};  // end namespace linalg
};  // end namespace raft
