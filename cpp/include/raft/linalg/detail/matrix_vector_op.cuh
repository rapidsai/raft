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

namespace {
template <size_t VecBytes>
struct AlignedAccess {
  template <typename T>
  static inline bool test(const T* matrix, size_t strideBytes)
  {
    return Pow2<VecBytes>::isAligned(matrix) && Pow2<VecBytes>::isAligned(strideBytes) &&
           Pow2<sizeof(T)>::isAligned(VecBytes);
  }
};
};  // namespace

template <typename Type, int veclen_, typename Lambda, typename IdxType>
__global__ void matrixVectorOpKernel(Type* out,
                                     const Type* matrix,
                                     const Type* vector,
                                     IdxType D,
                                     IdxType N,
                                     bool rowMajor,
                                     bool bcastAlongRows,
                                     Lambda op)
{
  typedef TxN_t<Type, veclen_> VecType;
  IdxType len = N * D;
  IdxType idx = threadIdx.x;
  idx += (IdxType)blockIdx.x * (IdxType)blockDim.x;
  idx *= VecType::Ratio;
  if (idx >= len) return;
  IdxType vIdx;
  VecType mat, vec;
  ///@todo: yikes! use fast-int-div here.
  ///@todo: shared mem for vector could help with perf
  if (rowMajor && bcastAlongRows) {
    vIdx = idx % D;
    vec.load(vector, vIdx);
  } else if (!rowMajor && !bcastAlongRows) {
    vIdx = idx % N;
    vec.load(vector, vIdx);
  } else if (rowMajor && !bcastAlongRows) {
    vIdx = idx / D;
    vec.fill(vector[vIdx]);
  } else {
    vIdx = idx / N;
    vec.fill(vector[vIdx]);
  }
  mat.load(matrix, idx);
#pragma unroll
  for (int i = 0; i < VecType::Ratio; ++i)
    mat.data[i] = op(mat.data[i], vec.data[i]);
  mat.store(out, idx);
}

template <typename Type, int veclen_, typename Lambda, typename IdxType, int TPB>
void matrixVectorOpImpl(Type* out,
                        const Type* matrix,
                        const Type* vec,
                        IdxType D,
                        IdxType N,
                        bool rowMajor,
                        bool bcastAlongRows,
                        Lambda op,
                        cudaStream_t stream)
{
  IdxType len   = N * D;
  IdxType nblks = raft::ceildiv(veclen_ ? len / veclen_ : veclen_, (IdxType)TPB);
  matrixVectorOpKernel<Type, veclen_, Lambda, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, matrix, vec, D, N, rowMajor, bcastAlongRows, op);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

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

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
