/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

namespace raft::matrix::detail {

/**
 * @brief Create a diagonal identity matrix
 * @param matrix: matrix of size total_size
 * @param lda: Leading dimension
 * @param total_size: number of elements of the matrix
 */
template <typename m_t, typename idx_t = int>
__global__ void createEyeKernel(m_t* matrix, idx_t lda, idx_t total_size)
{
  idx_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx < total_size) {
    idx_t i = idx % lda, j = idx / lda;
    matrix[idx] = m_t(j == i);
  }
}

template <typename m_t, typename idx_t = int>
void createEye(m_t* matrix, idx_t n_rows, idx_t n_cols, bool rowMajor, cudaStream_t stream)
{
  dim3 block(64);
  dim3 grid((n_rows * n_cols + block.x - 1) / block.x);
  idx_t lda = rowMajor ? n_cols : n_rows;
  createEyeKernel<<<grid, block, 0, stream>>>(matrix, lda, n_rows * n_cols);
}

}  // end namespace raft::matrix::detail