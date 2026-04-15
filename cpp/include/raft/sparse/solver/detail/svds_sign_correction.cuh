/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_utils.cuh>

#include <cstdint>

namespace raft::sparse::solver::detail {

/**
 * @brief CUDA kernel for SVD sign correction.
 *
 * For each component i, finds the element with largest absolute value in column i of U.
 * If that element is negative, flips the sign of both U[:, i] and Vt[i, :].
 * This ensures deterministic output regardless of the SVD algorithm's internal sign choices.
 *
 * One thread block per component. Uses shared memory reduction to find argmax(|U[:, i]|).
 *
 * @param U left singular vectors, col-major (m x k), U[:, i] is at U + i*m
 * @param Vt right singular vectors, col-major (k x n), Vt[i, :] is row i
 * @param m number of rows of U
 * @param n number of columns of Vt
 * @param k number of components
 */
template <typename ValueTypeT>
RAFT_KERNEL svd_sign_correction_kernel(ValueTypeT* U, ValueTypeT* Vt, int m, int n, int k)
{
  int comp = blockIdx.x;  // one block per component
  if (comp >= k) return;

  // Find max |U[:, comp]| via block reduction
  // U column comp starts at U + comp * m (col-major)
  ValueTypeT* u_col = U + static_cast<int64_t>(comp) * m;

  extern __shared__ char smem[];
  auto* s_max_val = reinterpret_cast<ValueTypeT*>(smem);
  auto* s_max_idx = reinterpret_cast<int*>(smem + blockDim.x * sizeof(ValueTypeT));

  ValueTypeT local_max_val = 0;
  int local_max_idx        = 0;

  for (int row = threadIdx.x; row < m; row += blockDim.x) {
    ValueTypeT abs_val = raft::abs(u_col[row]);
    if (abs_val > local_max_val) {
      local_max_val = abs_val;
      local_max_idx = row;
    }
  }

  s_max_val[threadIdx.x] = local_max_val;
  s_max_idx[threadIdx.x] = local_max_idx;
  __syncthreads();

  // Tree reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      if (s_max_val[threadIdx.x + stride] > s_max_val[threadIdx.x]) {
        s_max_val[threadIdx.x] = s_max_val[threadIdx.x + stride];
        s_max_idx[threadIdx.x] = s_max_idx[threadIdx.x + stride];
      }
    }
    __syncthreads();
  }

  // Thread 0 has the result
  bool flip = (s_max_val[0] > 0) && (u_col[s_max_idx[0]] < 0);
  __syncthreads();

  if (!flip) return;

  // Flip U column
  for (int row = threadIdx.x; row < m; row += blockDim.x) {
    u_col[row] = -u_col[row];
  }

  // Flip Vt row: Vt is col-major (k x n), so row i has elements at Vt[i + j*k] for j=0..n-1
  for (int col = threadIdx.x; col < n; col += blockDim.x) {
    Vt[comp + static_cast<int64_t>(col) * k] = -Vt[comp + static_cast<int64_t>(col) * k];
  }
}

/**
 * @brief Apply deterministic sign correction to SVD output.
 *
 * For each component, ensures the element with largest absolute value in U is positive.
 * Both U and Vt are modified in-place to maintain A ≈ U @ diag(S) @ Vt.
 *
 * @param handle raft resources handle
 * @param U left singular vectors of shape (m, k), col-major, modified in-place
 * @param Vt right singular vectors of shape (k, n), col-major, modified in-place
 */
template <typename ValueTypeT>
void svd_sign_correction(raft::resources const& handle,
                         raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> U,
                         raft::device_matrix_view<ValueTypeT, uint32_t, raft::col_major> Vt)
{
  int m = U.extent(0);
  int k = U.extent(1);
  int n = Vt.extent(1);

  auto stream = raft::resource::get_cuda_stream(handle);

  // threads_per_block must be a power of 2 for the tree reduction in the kernel
  constexpr int threads_per_block = 256;
  int smem_size                   = threads_per_block * (sizeof(ValueTypeT) + sizeof(int));

  svd_sign_correction_kernel<<<k, threads_per_block, smem_size, stream>>>(
    U.data_handle(), Vt.data_handle(), m, n, k);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace raft::sparse::solver::detail
