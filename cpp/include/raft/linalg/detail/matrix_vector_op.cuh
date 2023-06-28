/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/matrix/linewise_op.cuh>

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
  raft::resources handle;
  resource::set_cuda_stream(handle, stream);
  bool along_lines = rowMajor == bcastAlongRows;
  if (rowMajor) {
    matrix::linewise_op<MatT, IdxType, row_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, row_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, row_major>(out, N, D),
      along_lines,
      op,
      make_device_vector_view<const VecT, IdxType>(vec, bcastAlongRows ? N : D));
  } else {
    matrix::linewise_op<MatT, IdxType, col_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, col_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, col_major>(out, N, D),
      along_lines,
      op,
      make_device_vector_view<const VecT, IdxType>(vec, bcastAlongRows ? N : D));
  }
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
  raft::resources handle;
  resource::set_cuda_stream(handle, stream);
  bool along_lines = rowMajor == bcastAlongRows;
  if (rowMajor) {
    matrix::linewise_op<MatT, IdxType, row_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, row_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, row_major>(out, N, D),
      along_lines,
      op,
      make_device_vector_view<const Vec1T, IdxType>(vec1, bcastAlongRows ? N : D),
      make_device_vector_view<const Vec2T, IdxType>(vec2, bcastAlongRows ? N : D));
  } else {
    matrix::linewise_op<MatT, IdxType, col_major, Lambda>(
      handle,
      make_device_matrix_view<const MatT, IdxType, col_major>(matrix, N, D),
      make_device_matrix_view<MatT, IdxType, col_major>(out, N, D),
      along_lines,
      op,
      make_device_vector_view<const Vec1T, IdxType>(vec1, bcastAlongRows ? N : D),
      make_device_vector_view<const Vec2T, IdxType>(vec2, bcastAlongRows ? N : D));
  }
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
