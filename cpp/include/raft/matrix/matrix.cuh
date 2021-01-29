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

#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include <cstddef>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>

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
void copyRows(const m_t *in, idx_t n_rows, idx_t n_cols, m_t *out,
              const idx_array_t *indices, idx_t n_rows_indices,
              cudaStream_t stream, bool rowMajor = false) {
  if (rowMajor) {
    ASSERT(false, "matrix.h: row major is not supported yet!");
  }

  idx_t size = n_rows_indices * n_cols;
  auto counting = thrust::make_counting_iterator<idx_t>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + size,
                   [=] __device__(idx_t idx) {
                     idx_t row = idx % n_rows_indices;
                     idx_t col = idx / n_rows_indices;

                     out[col * n_rows_indices + row] =
                       in[col * n_rows + indices[row]];
                   });
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
void copy(const m_t *in, m_t *out, idx_t n_rows, idx_t n_cols,
          cudaStream_t stream) {
  raft::copy_async(out, in, n_rows * n_cols, stream);
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
void truncZeroOrigin(m_t *in, idx_t in_n_rows, m_t *out, idx_t out_n_rows,
                     idx_t out_n_cols, cudaStream_t stream) {
  auto m = out_n_rows;
  auto k = in_n_rows;
  idx_t size = out_n_rows * out_n_cols;
  auto d_q = in;
  auto d_q_trunc = out;
  auto counting = thrust::make_counting_iterator<idx_t>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting, counting + size,
                   [=] __device__(idx_t idx) {
                     idx_t row = idx % m;
                     idx_t col = idx / m;
                     d_q_trunc[col * m + row] = d_q[col * k + row];
                   });
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
void colReverse(m_t *inout, idx_t n_rows, idx_t n_cols, cudaStream_t stream) {
  auto n = n_cols;
  auto m = n_rows;
  idx_t size = n_rows * n_cols;
  auto d_q = inout;
  auto d_q_reversed = inout;
  auto counting = thrust::make_counting_iterator<idx_t>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + (size / 2), [=] __device__(idx_t idx) {
                     idx_t dest_row = idx % m;
                     idx_t dest_col = idx / m;
                     idx_t src_row = dest_row;
                     idx_t src_col = (n - dest_col) - 1;
                     m_t temp = (m_t)d_q_reversed[idx];
                     d_q_reversed[idx] = d_q[src_col * m + src_row];
                     d_q[src_col * m + src_row] = temp;
                   });
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
void rowReverse(m_t *inout, idx_t n_rows, idx_t n_cols, cudaStream_t stream) {
  auto m = n_rows;
  idx_t size = n_rows * n_cols;
  auto d_q = inout;
  auto d_q_reversed = inout;
  auto counting = thrust::make_counting_iterator<idx_t>(0);

  thrust::for_each(thrust::cuda::par.on(stream), counting,
                   counting + (size / 2), [=] __device__(idx_t idx) {
                     idx_t dest_row = idx % m;
                     idx_t dest_col = idx / m;
                     idx_t src_row = (m - dest_row) - 1;
                     ;
                     idx_t src_col = dest_col;

                     m_t temp = (m_t)d_q_reversed[idx];
                     d_q_reversed[idx] = d_q[src_col * m + src_row];
                     d_q[src_col * m + src_row] = temp;
                   });
}

/**
 * @brief Prints the data stored in GPU memory
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param h_separator: horizontal separator character
 * @param v_separator: vertical separator character
 */
template <typename m_t, typename idx_t = int>
void print(const m_t *in, idx_t n_rows, idx_t n_cols, char h_separator = ' ',
           char v_separator = '\n') {
  std::vector<m_t> h_matrix = std::vector<m_t>(n_cols * n_rows);
  CUDA_CHECK(cudaMemcpy(h_matrix.data(), in, n_cols * n_rows * sizeof(m_t),
                        cudaMemcpyDeviceToHost));

  for (idx_t i = 0; i < n_rows; i++) {
    for (idx_t j = 0; j < n_cols; j++) {
      printf("%1.4f%c", h_matrix[j * n_rows + i],
             j < n_cols - 1 ? h_separator : v_separator);
    }
  }
}

/**
 * @brief Prints the data stored in CPU memory
 * @param in: input matrix
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 */
template <typename m_t, typename idx_t = int>
void printHost(const m_t *in, idx_t n_rows, idx_t n_cols) {
  for (idx_t i = 0; i < n_rows; i++) {
    for (idx_t j = 0; j < n_cols; j++) {
      printf("%1.4f ", in[j * n_rows + i]);
    }
    printf("\n");
  }
}

/**
 * @brief Kernel for copying a slice of a big matrix to a small matrix with a
 * size matches that slice
 * @param src_d: input matrix
 * @param m: number of rows of input matrix
 * @param n: number of columns of input matrix
 * @param dst_d: output matrix
 * @param x1, y1: coordinate of the top-left point of the wanted area (0-based)
 * @param x2, y2: coordinate of the bottom-right point of the wanted area
 * (1-based)
 */
template <typename m_t, typename idx_t = int>
__global__ void slice(m_t *src_d, idx_t m, idx_t n, m_t *dst_d, idx_t x1,
                      idx_t y1, idx_t x2, idx_t y2) {
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx_t dm = x2 - x1, dn = y2 - y1;
  if (idx < dm * dn) {
    idx_t i = idx % dm, j = idx / dm;
    idx_t is = i + x1, js = j + y1;
    dst_d[idx] = src_d[is + js * m];
  }
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
void sliceMatrix(m_t *in, idx_t n_rows, idx_t n_cols, m_t *out, idx_t x1,
                 idx_t y1, idx_t x2, idx_t y2, cudaStream_t stream) {
  // Slicing
  dim3 block(64);
  dim3 grid(((x2 - x1) * (y2 - y1) + block.x - 1) / block.x);
  slice<<<grid, block, 0, stream>>>(in, n_rows, n_cols, out, x1, y1, x2, y2);
}

/**
 * @brief Kernel for copying the upper triangular part of a matrix to another
 * @param src: input matrix with a size of mxn
 * @param dst: output matrix with a size of kxk
 * @param n_rows: number of rows of input matrix
 * @param n_cols: number of columns of input matrix
 * @param k: min(n_rows, n_cols)
 */
template <typename m_t, typename idx_t = int>
__global__ void getUpperTriangular(m_t *src, m_t *dst, idx_t n_rows,
                                   idx_t n_cols, idx_t k) {
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  idx_t m = n_rows, n = n_cols;
  if (idx < m * n) {
    idx_t i = idx % m, j = idx / m;
    if (i < k && j < k && j >= i) {
      dst[i + j * k] = src[idx];
    }
  }
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
void copyUpperTriangular(m_t *src, m_t *dst, idx_t n_rows, idx_t n_cols,
                         cudaStream_t stream) {
  idx_t m = n_rows, n = n_cols;
  idx_t k = min(m, n);
  dim3 block(64);
  dim3 grid((m * n + block.x - 1) / block.x);
  getUpperTriangular<<<grid, block, 0, stream>>>(src, dst, m, n, k);
}

/**
 * @brief Copy a vector to the diagonal of a matrix
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols
 * @param m: number of rows of the matrix
 * @param n: number of columns of the matrix
 * @param k: dimensionality
 */
template <typename m_t, typename idx_t = int>
__global__ void copyVectorToMatrixDiagonal(m_t *vec, m_t *matrix, idx_t m,
                                           idx_t n, idx_t k) {
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;

  if (idx < k) {
    matrix[idx + idx * m] = vec[idx];
  }
}

/**
 * @brief Initialize a diagonal matrix with a vector
 * @param vec: vector of length k = min(n_rows, n_cols)
 * @param matrix: matrix of size n_rows x n_cols
 * @param n_rows: number of rows of the matrix
 * @param n_cols: number of columns of the matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void initializeDiagonalMatrix(m_t *vec, m_t *matrix, idx_t n_rows, idx_t n_cols,
                              cudaStream_t stream) {
  idx_t k = min(n_rows, n_cols);
  dim3 block(64);
  dim3 grid((k + block.x - 1) / block.x);
  copyVectorToMatrixDiagonal<<<grid, block, 0, stream>>>(vec, matrix, n_rows,
                                                         n_cols, k);
}

/**
 * @brief Calculate the inverse of the diagonal of a square matrix
 * element-wise and in place
 * @param in: square input matrix with size len x len
 * @param len: size of one side of the matrix
 */
template <typename m_t, typename idx_t = int>
__global__ void matrixDiagonalInverse(m_t *in, idx_t len) {
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < len) {
    in[idx + idx * len] = 1.0 / in[idx + idx * len];
  }
}

/**
 * @brief Get a square matrix with elements on diagonal reversed (in-place)
 * @param in: square input matrix with size len x len
 * @param len: size of one side of the matrix
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
void getDiagonalInverseMatrix(m_t *in, idx_t len, cudaStream_t stream) {
  dim3 block(64);
  dim3 grid((len + block.x - 1) / block.x);
  matrixDiagonalInverse<m_t><<<grid, block, 0, stream>>>(in, len);
}

/**
 * @brief Get the L2/F-norm of a matrix/vector
 * @param in: input matrix/vector with totally size elements
 * @param size: size of the matrix/vector
 * @param cublasH cublas handle
 * @param stream: cuda stream
 */
template <typename m_t, typename idx_t = int>
m_t getL2Norm(const raft::handle_t &handle, m_t *in, idx_t size,
              cudaStream_t stream) {
  cublasHandle_t cublasH = handle.get_cublas_handle();
  m_t normval = 0;
  CUBLAS_CHECK(
    raft::linalg::cublasnrm2(cublasH, size, in, 1, &normval, stream));
  return normval;
}

};  // end namespace matrix
};  // end namespace raft
