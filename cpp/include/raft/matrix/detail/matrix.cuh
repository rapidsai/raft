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

namespace raft {
namespace matrix {
namespace detail {

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

}  // end namespace detail
}  // end namespace matrix
}  // end namespace raft