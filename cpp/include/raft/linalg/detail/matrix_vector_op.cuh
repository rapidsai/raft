/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

 #include <raft/cuda_utils.cuh>
 #include <raft/vectorized.cuh>
 
 namespace raft {
 namespace linalg {
 namespace detail {
 
 template <typename Type, int veclen_, typename Lambda, typename IdxType>
 __global__ void matrixVectorOpKernel(Type *out, const Type *matrix,
                                      const Type *vector, IdxType D, IdxType N,
                                      bool rowMajor, bool bcastAlongRows,
                                      Lambda op) {
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
     mat.val.data[i] = op(mat.val.data[i], vec.val.data[i]);
   mat.store(out, idx);
 }
 
 template <typename Type, int veclen_, typename Lambda, typename IdxType,
           int TPB>
 void matrixVectorOpImpl(Type *out, const Type *matrix, const Type *vec,
                         IdxType D, IdxType N, bool rowMajor,
                         bool bcastAlongRows, Lambda op, cudaStream_t stream) {
   IdxType len = N * D;
   IdxType nblks =
     raft::ceildiv(veclen_ ? len / veclen_ : veclen_, (IdxType)TPB);
   matrixVectorOpKernel<Type, veclen_, Lambda, IdxType>
     <<<nblks, TPB, 0, stream>>>(out, matrix, vec, D, N, rowMajor,
                                 bcastAlongRows, op);
   CUDA_CHECK(cudaPeekAtLastError());
 }
 
 template <typename Type, typename Lambda, typename IdxType = int, int TPB = 256>
 void matrixVectorOp(Type *out, const Type *matrix, const Type *vec, IdxType D,
                     IdxType N, bool rowMajor, bool bcastAlongRows, Lambda op,
                     cudaStream_t stream) {
   IdxType stride = rowMajor ? D : N;
   size_t stride_bytes = stride * sizeof(Type);
 
   auto test_aligned_access = [stride_bytes, matrix](const int n_bytes) {
     return n_bytes / sizeof(Type) && stride_bytes % n_bytes == 0 &&
            reinterpret_cast<uintptr_t>(matrix) % sizeof(Type);
   };
 
   if (test_aligned_access(16)) {
     matrixVectorOpImpl<Type, 16 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (test_aligned_access(8)) {
     matrixVectorOpImpl<Type, 8 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (test_aligned_access(4)) {
     matrixVectorOpImpl<Type, 4 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (test_aligned_access(2)) {
     matrixVectorOpImpl<Type, 2 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (1 / sizeof(Type)) {
     matrixVectorOpImpl<Type, 1 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
   } else {
     matrixVectorOpImpl<Type, 1, Lambda, IdxType, TPB>(
       out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
   }
 }
 
 ///@todo: come up with a cleaner interface to support these cases in future!
 
 template <typename Type, int veclen_, typename Lambda, typename IdxType>
 __global__ void matrixVectorOpKernel(Type *out, const Type *matrix,
                                      const Type *vector1, const Type *vector2,
                                      IdxType D, IdxType N, bool rowMajor,
                                      bool bcastAlongRows, Lambda op) {
   typedef TxN_t<Type, veclen_> VecType;
   IdxType len = N * D;
   IdxType idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * VecType::Ratio;
   if (idx >= len) return;
   IdxType vIdx;
   VecType mat, vec1, vec2;
   ///@todo: yikes! use fast-int-div here.
   ///@todo: shared mem for vector could help with perf
   if (rowMajor && bcastAlongRows) {
     vIdx = idx % D;
     vec1.load(vector1, vIdx);
     vec2.load(vector2, vIdx);
   } else if (!rowMajor && !bcastAlongRows) {
     vIdx = idx % N;
     vec1.load(vector1, vIdx);
     vec2.load(vector2, vIdx);
   } else if (rowMajor && !bcastAlongRows) {
     vIdx = idx / D;
     vec1.fill(vector1[vIdx]);
     vec2.fill(vector2[vIdx]);
   } else {
     vIdx = idx / N;
     vec1.fill(vector1[vIdx]);
     vec2.fill(vector2[vIdx]);
   }
   mat.load(matrix, idx);
 #pragma unroll
   for (int i = 0; i < VecType::Ratio; ++i)
     mat.val.data[i] = op(mat.val.data[i], vec1.val.data[i], vec2.val.data[i]);
   mat.store(out, idx);
 }
 
 template <typename Type, int veclen_, typename Lambda, typename IdxType,
           int TPB>
 void matrixVectorOpImpl(Type *out, const Type *matrix, const Type *vec1,
                         const Type *vec2, IdxType D, IdxType N, bool rowMajor,
                         bool bcastAlongRows, Lambda op, cudaStream_t stream) {
   IdxType nblks = raft::ceildiv(N * D, (IdxType)TPB);
   matrixVectorOpKernel<Type, veclen_, Lambda, IdxType>
     <<<nblks, TPB, 0, stream>>>(out, matrix, vec1, vec2, D, N, rowMajor,
                                 bcastAlongRows, op);
   CUDA_CHECK(cudaPeekAtLastError());
 }
 
 template <typename Type, typename Lambda, typename IdxType = int, int TPB = 256>
 void matrixVectorOp(Type *out, const Type *matrix, const Type *vec1,
                     const Type *vec2, IdxType D, IdxType N, bool rowMajor,
                     bool bcastAlongRows, Lambda op, cudaStream_t stream) {
   IdxType stride = rowMajor ? D : N;
   size_t stride_bytes = stride * sizeof(Type);
 
   auto test_aligned_access = [stride_bytes, matrix](const int n_bytes) {
     return n_bytes / sizeof(Type) && stride_bytes % n_bytes == 0 &&
            reinterpret_cast<uintptr_t>(matrix) % sizeof(Type);
   };
 
   if (test_aligned_access(16)) {
     matrixVectorOpImpl<Type, 16 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (test_aligned_access(8)) {
     matrixVectorOpImpl<Type, 8 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (test_aligned_access(4)) {
     matrixVectorOpImpl<Type, 4 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (test_aligned_access(2)) {
     matrixVectorOpImpl<Type, 2 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
   } else if (1 / sizeof(Type)) {
     matrixVectorOpImpl<Type, 1 / sizeof(Type), Lambda, IdxType, TPB>(
       out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
   } else {
     matrixVectorOpImpl<Type, 1, Lambda, IdxType, TPB>(
       out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
   }
 }
 
 };  // end namespace detail 
 };  // end namespace linalg
 };  // end namespace raft
 