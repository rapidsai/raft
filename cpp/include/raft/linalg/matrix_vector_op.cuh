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

#include <raft/cuda_utils.cuh>
#include <raft/vectorized.cuh>

namespace raft {
namespace linalg {

template <typename Type, int VecLen, typename Lambda, typename IdxType>
__global__ void matrix_vector_op_kernel(Type *out, const Type *matrix,
                                     const Type *vector, IdxType D, IdxType N,
                                     bool rowMajor, bool bcastAlongRows,
                                     Lambda op) {
  using vec_t = TxN_t<Type, VecLen>;
  IdxType len = N * D;
  IdxType idx = threadIdx.x;
  idx += static_cast<IdxType>(blockIdx.x) * static_cast<IdxType>(blockDim.x);
  idx *= vec_t::Ratio;
  if (idx >= len) return;
  IdxType v_idx;
  vec_t mat, vec;
  ///@todo: yikes! use fast-int-div here.
  ///@todo: shared mem for vector could help with perf
  if (rowMajor && bcastAlongRows) {
    v_idx = idx % D;
    vec.load(vector, v_idx);
  } else if (!rowMajor && !bcastAlongRows) {
    v_idx = idx % N;
    vec.load(vector, v_idx);
  } else if (rowMajor && !bcastAlongRows) {
    v_idx = idx / D;
    vec.fill(vector[v_idx]);
  } else {
    v_idx = idx / N;
    vec.fill(vector[v_idx]);
  }
  mat.load(matrix, idx);
#pragma unroll
  for (int i = 0; i < vec_t::Ratio; ++i) {
    mat.val.data[i] = op(mat.val.data[i], vec.val.data[i]);
  }
  mat.store(out, idx);
}

template <typename Type, int VecLen, typename Lambda, typename IdxType,
          int TPB>
void matrix_vector_op_impl(Type *out, const Type *matrix, const Type *vec,
                        IdxType D, IdxType N, bool rowMajor,
                        bool bcastAlongRows, Lambda op, cudaStream_t stream) {
  auto len = N * D;
  IdxType nblks =
    raft::ceildiv(VecLen ? len / VecLen : VecLen, static_cast<IdxType>(TPB));
  matrix_vector_op_kernel<Type, VecLen, Lambda, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, matrix, vec, D, N, rowMajor,
                                bcastAlongRows, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Operations for all the columns or rows with a given vector.
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
void matrixVectorOp(Type *out, const Type *matrix, const Type *vec,  // NOLINT
                    IdxType D,
                    IdxType N, bool rowMajor, bool bcastAlongRows, Lambda op,
                    cudaStream_t stream) {
  IdxType stride = rowMajor ? D : N;
  size_t bytes = stride * sizeof(Type);
  if (16 / sizeof(Type) && bytes % 16 == 0) {
    matrix_vector_op_impl<Type, 16 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (8 / sizeof(Type) && bytes % 8 == 0) {
    matrix_vector_op_impl<Type, 8 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (4 / sizeof(Type) && bytes % 4 == 0) {
    matrix_vector_op_impl<Type, 4 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (2 / sizeof(Type) && bytes % 2 == 0) {
    matrix_vector_op_impl<Type, 2 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (1 / sizeof(Type)) {
    matrix_vector_op_impl<Type, 1 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  } else {
    matrix_vector_op_impl<Type, 1, Lambda, IdxType, TPB>(
      out, matrix, vec, D, N, rowMajor, bcastAlongRows, op, stream);
  }
}

///@todo: come up with a cleaner interface to support these cases in future!

template <typename Type, int VecLen, typename Lambda, typename IdxType>
__global__ void matrix_vector_op_kernel(Type *out, const Type *matrix,
                                     const Type *vector1, const Type *vector2,
                                     IdxType D, IdxType N, bool rowMajor,
                                     bool bcastAlongRows, Lambda op) {
  using vec_t = TxN_t<Type, VecLen>;
  auto len = N * D;
  IdxType idx = (threadIdx.x + (blockIdx.x * blockDim.x)) * vec_t::Ratio;
  if (idx >= len) return;
  IdxType v_idx;
  vec_t mat, vec1, vec2;
  ///@todo: yikes! use fast-int-div here.
  ///@todo: shared mem for vector could help with perf
  if (rowMajor && bcastAlongRows) {
    v_idx = idx % D;
    vec1.load(vector1, v_idx);
    vec2.load(vector2, v_idx);
  } else if (!rowMajor && !bcastAlongRows) {
    v_idx = idx % N;
    vec1.load(vector1, v_idx);
    vec2.load(vector2, v_idx);
  } else if (rowMajor && !bcastAlongRows) {
    v_idx = idx / D;
    vec1.fill(vector1[v_idx]);
    vec2.fill(vector2[v_idx]);
  } else {
    v_idx = idx / N;
    vec1.fill(vector1[v_idx]);
    vec2.fill(vector2[v_idx]);
  }
  mat.load(matrix, idx);
#pragma unroll
  for (int i = 0; i < vec_t::Ratio; ++i) {
    mat.val.data[i] = op(mat.val.data[i], vec1.val.data[i], vec2.val.data[i]);
  }
  mat.store(out, idx);
}

template <typename Type, int VecLen, typename Lambda, typename IdxType,
          int TPB>
void matrix_vector_op_impl(Type *out, const Type *matrix, const Type *vec1,
                        const Type *vec2, IdxType D, IdxType N, bool rowMajor,
                        bool bcastAlongRows, Lambda op, cudaStream_t stream) {
  auto nblks = raft::ceildiv(N * D, static_cast<IdxType>(TPB));
  matrix_vector_op_kernel<Type, VecLen, Lambda, IdxType>
    <<<nblks, TPB, 0, stream>>>(out, matrix, vec1, vec2, D, N, rowMajor,
                                bcastAlongRows, op);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Operations for all the columns or rows with the given vectors.
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
void matrixVectorOp(Type *out, const Type *matrix, const Type *vec1,  // NOLINT
                    const Type *vec2, IdxType D, IdxType N, bool rowMajor,
                    bool bcastAlongRows, Lambda op, cudaStream_t stream) {
  auto stride = rowMajor ? D : N;
  size_t bytes = stride * sizeof(Type);
  if (16 / sizeof(Type) && bytes % 16 == 0) {
    matrix_vector_op_impl<Type, 16 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (8 / sizeof(Type) && bytes % 8 == 0) {
    matrix_vector_op_impl<Type, 8 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (4 / sizeof(Type) && bytes % 4 == 0) {
    matrix_vector_op_impl<Type, 4 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (2 / sizeof(Type) && bytes % 2 == 0) {
    matrix_vector_op_impl<Type, 2 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else if (1 / sizeof(Type)) {
    matrix_vector_op_impl<Type, 1 / sizeof(Type), Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  } else {
    matrix_vector_op_impl<Type, 1, Lambda, IdxType, TPB>(
      out, matrix, vec1, vec2, D, N, rowMajor, bcastAlongRows, op, stream);
  }
}

};  // end namespace linalg
};  // end namespace raft
