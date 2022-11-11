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

#include <raft/matrix/matrix.cuh>

namespace raft {
namespace linalg {
namespace detail {

template <typename MatT, typename Lambda, typename VecT, typename IdxType = int, int TPB = 256>
void matrixVectorOp(MatT* out,
                    const MatT* matrix,
                    const VecT* vec,
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

template <typename MatT,
          typename Lambda,
          typename Vec1T,
          typename Vec2T,
          typename IdxType = int,
          int TPB          = 256>
void matrixVectorOp(MatT* out,
                    const MatT* matrix,
                    const Vec1T* vec1,
                    const Vec2T* vec2,
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

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
