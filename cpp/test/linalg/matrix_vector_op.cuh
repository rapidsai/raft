/*
 * Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft {
namespace linalg {

template <typename OutT, typename MatT, typename VecT, typename Lambda, typename IdxType = int>
RAFT_KERNEL naiveMatVecKernel(OutT* out,
                              const MatT* mat,
                              const VecT* vec,
                              IdxType D,
                              IdxType N,
                              bool rowMajor,
                              bool bcastAlongRows,
                              Lambda op)
{
  IdxType idx = threadIdx.x + blockIdx.x * blockDim.x;
  IdxType len = N * D;
  IdxType col;
  if (rowMajor && bcastAlongRows) {
    col = idx % D;
  } else if (!rowMajor && !bcastAlongRows) {
    col = idx % N;
  } else if (rowMajor && !bcastAlongRows) {
    col = idx / D;
  } else {
    col = idx / N;
  }
  if (idx < len) { out[idx] = op(mat[idx], vec[col]); }
}

template <typename OutT, typename MatT, typename VecT, typename Lambda, typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const VecT* vec,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 Lambda op,
                 cudaStream_t stream)
{
  static const IdxType TPB = 64;
  IdxType len              = N * D;
  IdxType nblks            = raft::ceildiv(len, TPB);
  naiveMatVecKernel<<<nblks, TPB, 0, stream>>>(out, mat, vec, D, N, rowMajor, bcastAlongRows, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename OutT, typename MatT, typename VecT, typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const VecT* vec,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 OutT scalar,
                 cudaStream_t stream)
{
  naiveMatVec(
    out,
    mat,
    vec,
    D,
    N,
    rowMajor,
    bcastAlongRows,
    [scalar] __device__(MatT a, VecT b) { return (OutT)(a + scalar * b); },
    stream);
}

template <typename OutT,
          typename MatT,
          typename Vec1T,
          typename Vec2T,
          typename Lambda,
          typename IdxType = int>
RAFT_KERNEL naiveMatVecKernel(OutT* out,
                              const MatT* mat,
                              const Vec1T* vec1,
                              const Vec2T* vec2,
                              IdxType D,
                              IdxType N,
                              bool rowMajor,
                              bool bcastAlongRows,
                              Lambda op)
{
  IdxType idx = threadIdx.x + blockIdx.x * blockDim.x;
  IdxType len = N * D;
  IdxType col;
  if (rowMajor && bcastAlongRows) {
    col = idx % D;
  } else if (!rowMajor && !bcastAlongRows) {
    col = idx % N;
  } else if (rowMajor && !bcastAlongRows) {
    col = idx / D;
  } else {
    col = idx / N;
  }
  if (idx < len) { out[idx] = op(mat[idx], vec1[col], vec2[col]); }
}

template <typename OutT,
          typename MatT,
          typename Vec1T,
          typename Vec2T,
          typename Lambda,
          typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const Vec1T* vec1,
                 const Vec2T* vec2,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 Lambda op,
                 cudaStream_t stream)
{
  static const IdxType TPB = 64;
  IdxType len              = N * D;
  IdxType nblks            = raft::ceildiv(len, TPB);
  naiveMatVecKernel<<<nblks, TPB, 0, stream>>>(
    out, mat, vec1, vec2, D, N, rowMajor, bcastAlongRows, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename OutT, typename MatT, typename Vec1T, typename Vec2T, typename IdxType = int>
void naiveMatVec(OutT* out,
                 const MatT* mat,
                 const Vec1T* vec1,
                 const Vec2T* vec2,
                 IdxType D,
                 IdxType N,
                 bool rowMajor,
                 bool bcastAlongRows,
                 OutT scalar,
                 cudaStream_t stream)
{
  naiveMatVec(
    out,
    mat,
    vec1,
    vec2,
    D,
    N,
    rowMajor,
    bcastAlongRows,
    [scalar] __device__(MatT a, Vec1T b, Vec2T c) { return (OutT)(a + scalar * b + c); },
    stream);
}

}  // end namespace linalg
}  // end namespace raft
