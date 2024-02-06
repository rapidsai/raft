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

#include <raft/core/handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/device_atomics.cuh>

namespace raft::distance::detail {

/**
 * @brief Compress 2D boolean matrix to bitfield
 *
 * Utility kernel for masked_l2_nn.
 *
 * @tparam T
 *
 * @parameter[in]  in       An `m x n` boolean matrix. Row major.
 * @parameter[out] out      An `(m / bits_per_elem) x n` matrix with elements of
 *                          type T, where T is of size `bits_per_elem` bits.
 *                          Note: the division (`/`) is a ceilDiv.
 */
template <typename T = uint64_t, typename = std::enable_if_t<std::is_integral<T>::value>>
RAFT_KERNEL compress_to_bits_kernel(
  raft::device_matrix_view<const bool, int, raft::layout_c_contiguous> in,
  raft::device_matrix_view<T, int, raft::layout_c_contiguous> out)
{
  constexpr int bits_per_element = 8 * sizeof(T);
  constexpr int tile_dim_m       = bits_per_element;
  constexpr int nthreads         = 128;
  constexpr int tile_dim_n       = nthreads;  // read 128 bools at once = 1 sector

  // Tile in shared memory is transposed
  __shared__ bool smem[tile_dim_n][tile_dim_m];

  const int num_tiles_per_m = raft::ceildiv(in.extent(0), tile_dim_m);
  const int num_tiles_per_n = raft::ceildiv(in.extent(1), tile_dim_n);

  for (int lin_tile_idx = blockIdx.x; true; lin_tile_idx += gridDim.x) {
    const int tile_idx_n = tile_dim_n * (lin_tile_idx % num_tiles_per_n);
    const int tile_idx_m = tile_dim_m * (lin_tile_idx / num_tiles_per_n);

    if (in.extent(0) <= tile_idx_m) { break; }
    // Fill shared memory tile
    bool reg_buf[tile_dim_m];
#pragma unroll
    for (int i = 0; i < tile_dim_m; ++i) {
      const int in_m       = tile_idx_m + i;
      const int in_n       = tile_idx_n + threadIdx.x;
      bool in_bounds       = in_m < in.extent(0) && in_n < in.extent(1);
      reg_buf[i]           = in_bounds ? in(in_m, in_n) : false;
      smem[threadIdx.x][i] = reg_buf[i];
    }
    __syncthreads();

    // Drain memory tile into single output element out_elem.
    T out_elem{0};
#pragma unroll
    for (int j = 0; j < tile_dim_n; ++j) {
      if (smem[threadIdx.x][j]) { out_elem |= T(1) << j; }
    }
    __syncthreads();

    // Write output.
    int out_m = tile_idx_m / bits_per_element;
    int out_n = tile_idx_n + threadIdx.x;

    if (out_m < out.extent(0) && out_n < out.extent(1)) { out(out_m, out_n) = out_elem; }
  }
}

/**
 * @brief Compress 2D boolean matrix to bitfield
 *
 * Utility kernel for masked_l2_nn.
 *
 * @tparam T
 *
 * @parameter[in]  in       An `m x n` boolean matrix. Row major.
 * @parameter[out] out      An `(m / bits_per_elem) x n` matrix with elements of
 *                          type T, where T is of size `bits_per_elem` bits.
 *                          Note: the division (`/`) is a ceilDiv.
 */
template <typename T = uint64_t, typename = std::enable_if_t<std::is_integral<T>::value>>
void compress_to_bits(raft::resources const& handle,
                      raft::device_matrix_view<const bool, int, raft::layout_c_contiguous> in,
                      raft::device_matrix_view<T, int, raft::layout_c_contiguous> out)
{
  auto stream                    = resource::get_cuda_stream(handle);
  constexpr int bits_per_element = 8 * sizeof(T);

  RAFT_EXPECTS(raft::ceildiv(in.extent(0), bits_per_element) == out.extent(0),
               "Number of output rows must be ceildiv(input rows, bits_per_elem)");
  RAFT_EXPECTS(in.extent(1) == out.extent(1), "Number of output columns must equal input columns.");

  const int num_SMs           = raft::getMultiProcessorCount();
  int blocks_per_sm           = 0;
  constexpr int num_threads   = 128;
  constexpr int dyn_smem_size = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, compress_to_bits_kernel<T>, num_threads, dyn_smem_size));

  dim3 grid(num_SMs * blocks_per_sm);
  dim3 block(128);
  compress_to_bits_kernel<<<grid, block, 0, stream>>>(in, out);
  RAFT_CUDA_TRY(cudaGetLastError());
}

};  // namespace raft::distance::detail
