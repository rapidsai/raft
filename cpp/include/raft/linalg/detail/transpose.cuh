/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "cublas_wrappers.hpp"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include <cmath>

namespace raft {
namespace linalg {
namespace detail {

template <typename IndexType, int TILE_DIM, int BLOCK_ROWS>
RAFT_KERNEL transpose_half_kernel(IndexType n_rows,
                                  IndexType n_cols,
                                  const half* __restrict__ in,
                                  half* __restrict__ out)
{
  __shared__ half tile[TILE_DIM][TILE_DIM + 1];

  for (int block_offset_y = 0; block_offset_y < n_rows; block_offset_y += gridDim.y * TILE_DIM) {
    for (int block_offset_x = 0; block_offset_x < n_cols; block_offset_x += gridDim.x * TILE_DIM) {
      auto x = block_offset_x + blockIdx.x * TILE_DIM + threadIdx.x;
      auto y = block_offset_y + blockIdx.y * TILE_DIM + threadIdx.y;

      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n_cols && (y + j) < n_rows) {
          tile[threadIdx.y + j][threadIdx.x] = __ldg(&in[(y + j) * n_cols + x]);
        }
      }
      __syncthreads();

      x = block_offset_y + blockIdx.y * TILE_DIM + threadIdx.x;
      y = block_offset_x + blockIdx.x * TILE_DIM + threadIdx.y;

      for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < n_rows && (y + j) < n_cols) {
          out[(y + j) * n_rows + x] = tile[threadIdx.x][threadIdx.y + j];
        }
      }
      __syncthreads();
    }
  }
}

template <typename IndexType>
void transpose_half(
  raft::resources const& handle, IndexType n_rows, IndexType n_cols, const half* in, half* out)
{
  if (n_cols == 0 || n_rows == 0) return;
  auto stream = resource::get_cuda_stream(handle);

  int dev_id, sm_count;

  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);

  constexpr int tpb         = 256;
  constexpr int block_dim_x = 128 / sizeof(half);
  constexpr int block_dim_y = tpb / block_dim_x;

  dim3 blocks(block_dim_x, block_dim_y);

  int max_active_blocks = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_active_blocks, transpose_half_kernel<IndexType, block_dim_x, block_dim_y>, tpb, 0));
  int num_blocks = max_active_blocks * sm_count;

  int grid_x = (n_cols + block_dim_x - 1) / block_dim_x;
  int grid_y = (n_rows + block_dim_x - 1) / block_dim_x;

  float ratio = static_cast<float>(grid_y) / static_cast<float>(grid_x);
  int adjusted_grid_y =
    std::max(std::min(grid_y, static_cast<int>(std::sqrt(num_blocks * ratio))), 1);
  int adjusted_grid_x = std::max(std::min(grid_x, num_blocks / adjusted_grid_y), 1);

  dim3 grids(adjusted_grid_x, adjusted_grid_y);

  transpose_half_kernel<IndexType, block_dim_x, block_dim_y>
    <<<grids, blocks, 0, stream>>>(n_rows, n_cols, in, out);

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename math_t>
void transpose(raft::resources const& handle,
               math_t* in,
               math_t* out,
               int n_rows,
               int n_cols,
               cudaStream_t stream)
{
  int out_n_rows = n_cols;
  int out_n_cols = n_rows;

  if constexpr (std::is_same_v<math_t, half>) {
    transpose_half(handle, out_n_rows, out_n_cols, in, out);
  } else {
    cublasHandle_t cublas_h = resource::get_cublas_handle(handle);
    RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, stream));
    const math_t alpha = 1.0;
    const math_t beta  = 0.0;
    RAFT_CUBLAS_TRY(cublasgeam(cublas_h,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               out_n_rows,
                               out_n_cols,
                               &alpha,
                               in,
                               n_rows,
                               &beta,
                               out,
                               out_n_rows,
                               out,
                               out_n_rows,
                               stream));
  }
}

template <typename math_t>
void transpose(math_t* inout, int n, cudaStream_t stream)
{
  auto m        = n;
  auto size     = n * n;
  auto d_inout  = inout;
  auto counting = thrust::make_counting_iterator<int>(0);

  thrust::for_each(rmm::exec_policy(stream), counting, counting + size, [=] __device__(int idx) {
    int s_row = idx % m;
    int s_col = idx / m;
    int d_row = s_col;
    int d_col = s_row;
    if (s_row < s_col) {
      auto temp                  = d_inout[d_col * m + d_row];
      d_inout[d_col * m + d_row] = d_inout[s_col * m + s_row];
      d_inout[s_col * m + s_row] = temp;
    }
  });
}

template <typename T, typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_row_major_impl(
  raft::resources const& handle,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  auto out_n_rows   = in.extent(1);
  auto out_n_cols   = in.extent(0);
  T constexpr kOne  = 1;
  T constexpr kZero = 0;

  CUBLAS_TRY(cublasgeam(resource::get_cublas_handle(handle),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        out_n_cols,
                        out_n_rows,
                        &kOne,
                        in.data_handle(),
                        in.stride(0),
                        &kZero,
                        static_cast<T*>(nullptr),
                        out.stride(0),
                        out.data_handle(),
                        out.stride(0),
                        resource::get_cuda_stream(handle)));
}

template <typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_row_major_impl(
  raft::resources const& handle,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  auto out_n_rows = in.extent(1);
  auto out_n_cols = in.extent(0);
  transpose_half<IndexType>(handle, out_n_cols, out_n_rows, in.data_handle(), out.data_handle());
}

template <typename T, typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_col_major_impl(
  raft::resources const& handle,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<T, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  auto out_n_rows   = in.extent(1);
  auto out_n_cols   = in.extent(0);
  T constexpr kOne  = 1;
  T constexpr kZero = 0;

  CUBLAS_TRY(cublasgeam(resource::get_cublas_handle(handle),
                        CUBLAS_OP_T,
                        CUBLAS_OP_N,
                        out_n_rows,
                        out_n_cols,
                        &kOne,
                        in.data_handle(),
                        in.stride(1),
                        &kZero,
                        static_cast<T*>(nullptr),
                        out.stride(1),
                        out.data_handle(),
                        out.stride(1),
                        resource::get_cuda_stream(handle)));
}

template <typename IndexType, typename LayoutPolicy, typename AccessorPolicy>
void transpose_col_major_impl(
  raft::resources const& handle,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> in,
  raft::mdspan<half, raft::matrix_extent<IndexType>, LayoutPolicy, AccessorPolicy> out)
{
  auto out_n_rows = in.extent(1);
  auto out_n_cols = in.extent(0);
  transpose_half<IndexType>(handle, out_n_rows, out_n_cols, in.data_handle(), out.data_handle());
}

};  // end namespace detail
};  // end namespace linalg
};  // end namespace raft
