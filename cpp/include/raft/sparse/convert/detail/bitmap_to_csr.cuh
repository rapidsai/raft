/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <raft/core/detail/mdspan_util.cuh>  // detail::popc
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/detail/adj_to_csr.cuh>

#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <assert.h>

namespace cg = cooperative_groups;

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

// Threads per block in calc_nnz_by_rows_kernel.
static const constexpr int calc_nnz_by_rows_tpb = 32;

template <typename bitmap_t, typename index_t, typename nnz_t>
RAFT_KERNEL __launch_bounds__(calc_nnz_by_rows_tpb) calc_nnz_by_rows_kernel(const bitmap_t* bitmap,
                                                                            index_t num_rows,
                                                                            index_t num_cols,
                                                                            index_t bitmap_num,
                                                                            nnz_t* nnz_per_row)
{
  constexpr bitmap_t FULL_MASK      = ~bitmap_t(0u);
  constexpr bitmap_t ONE            = bitmap_t(1u);
  constexpr index_t BITS_PER_BITMAP = sizeof(bitmap_t) * 8;

  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<32>(block);

  int lane_id = threadIdx.x & 0x1f;

  for (index_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    index_t offset = 0;
    index_t s_bit  = row * num_cols;
    index_t e_bit  = s_bit + num_cols;
    index_t l_sum  = 0;

    while (offset < num_cols) {
      index_t bitmap_idx                     = lane_id + (s_bit + offset) / BITS_PER_BITMAP;
      std::remove_const_t<bitmap_t> l_bitmap = 0;

      if (bitmap_idx * BITS_PER_BITMAP < e_bit) { l_bitmap = bitmap[bitmap_idx]; }

      if (s_bit > bitmap_idx * BITS_PER_BITMAP) {
        l_bitmap >>= (s_bit - bitmap_idx * BITS_PER_BITMAP);
        l_bitmap <<= (s_bit - bitmap_idx * BITS_PER_BITMAP);
      }

      if ((bitmap_idx + 1) * BITS_PER_BITMAP > e_bit) {
        l_bitmap <<= ((bitmap_idx + 1) * BITS_PER_BITMAP - e_bit);
        l_bitmap >>= ((bitmap_idx + 1) * BITS_PER_BITMAP - e_bit);
      }

      l_sum += static_cast<index_t>(raft::detail::popc(l_bitmap));
      offset += BITS_PER_BITMAP * warpSize;
    }

    l_sum = cg::reduce(tile, l_sum, cg::plus<index_t>());

    if (lane_id == 0) { *(nnz_per_row + row) += static_cast<nnz_t>(l_sum); }
  }
}

template <typename bitmap_t, typename index_t, typename nnz_t>
void calc_nnz_by_rows(raft::resources const& handle,
                      const bitmap_t* bitmap,
                      index_t num_rows,
                      index_t num_cols,
                      nnz_t* nnz_per_row)
{
  auto stream              = resource::get_cuda_stream(handle);
  const index_t total      = num_rows * num_cols;
  const index_t bitmap_num = raft::ceildiv(total, index_t(sizeof(bitmap_t) * 8));

  int dev_id, sm_count, blocks_per_sm;

  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, calc_nnz_by_rows_kernel<bitmap_t, index_t, nnz_t>, calc_nnz_by_rows_tpb, 0);

  index_t max_active_blocks = sm_count * blocks_per_sm;
  auto grid = std::min(max_active_blocks, raft::ceildiv(bitmap_num, index_t(calc_nnz_by_rows_tpb)));
  auto block = calc_nnz_by_rows_tpb;

  calc_nnz_by_rows_kernel<bitmap_t, index_t, nnz_t>
    <<<grid, block, 0, stream>>>(bitmap, num_rows, num_cols, bitmap_num, nnz_per_row);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/*
  Execute the exclusive_scan within one warp with no inter-warp communication.
  This function calculates the exclusive prefix sum of `value` across threads within the same warp.
  Each thread in the warp will end up with the sum of all the values of the threads with lower IDs
  in the same warp, with the first thread always getting a sum of 0.
*/
template <typename value_t>
RAFT_DEVICE_INLINE_FUNCTION value_t warp_exclusive_scan(value_t value)
{
  int lane_id           = threadIdx.x & 0x1f;
  value_t shifted_value = __shfl_up_sync(0xffffffff, value, 1, warpSize);
  if (lane_id == 0) shifted_value = 0;

  value_t sum = shifted_value;

  for (int i = 1; i < warpSize; i *= 2) {
    value_t n = __shfl_up_sync(0xffffffff, sum, i, warpSize);
    if (lane_id >= i) { sum += n; }
  }
  return sum;
}

// Threads per block in fill_indices_by_rows_kernel.
static const constexpr int fill_indices_by_rows_tpb = 32;

template <typename bitmap_t, typename index_t, typename nnz_t, bool check_nnz>
RAFT_KERNEL __launch_bounds__(fill_indices_by_rows_tpb)
  fill_indices_by_rows_kernel(const bitmap_t* bitmap,
                              const index_t* indptr,
                              index_t num_rows,
                              index_t num_cols,
                              nnz_t nnz,
                              index_t bitmap_num,
                              index_t* indices)
{
  constexpr bitmap_t FULL_MASK      = ~bitmap_t(0u);
  constexpr bitmap_t ONE            = bitmap_t(1u);
  constexpr index_t BITS_PER_BITMAP = sizeof(bitmap_t) * 8;

  int lane_id = threadIdx.x & 0x1f;

  // Ensure the HBM allocated for CSR values is sufficient to handle all non-zero bitmap bits.
  // An assert will trigger if the allocated HBM is insufficient when `NDEBUG` isn't defined.
  // Note: Assertion is active only if `NDEBUG` is undefined.
  if constexpr (check_nnz) {
    if (lane_id == 0) { assert(nnz < indptr[num_rows]); }
  }

#pragma unroll
  for (index_t row = blockIdx.x; row < num_rows; row += gridDim.x) {
    index_t g_sum      = 0;
    index_t s_bit      = row * num_cols;
    index_t e_bit      = s_bit + num_cols;
    index_t indptr_row = indptr[row];

#pragma unroll
    for (index_t offset = 0; offset < num_cols; offset += BITS_PER_BITMAP * warpSize) {
      index_t bitmap_idx                     = lane_id + (s_bit + offset) / BITS_PER_BITMAP;
      std::remove_const_t<bitmap_t> l_bitmap = 0;
      index_t l_offset = offset + lane_id * BITS_PER_BITMAP - (s_bit % BITS_PER_BITMAP);

      if (bitmap_idx * BITS_PER_BITMAP < e_bit) { l_bitmap = bitmap[bitmap_idx]; }

      if (s_bit > bitmap_idx * BITS_PER_BITMAP) {
        l_bitmap >>= (s_bit - bitmap_idx * BITS_PER_BITMAP);
        l_bitmap <<= (s_bit - bitmap_idx * BITS_PER_BITMAP);
      }

      if ((bitmap_idx + 1) * BITS_PER_BITMAP > e_bit) {
        l_bitmap <<= ((bitmap_idx + 1) * BITS_PER_BITMAP - e_bit);
        l_bitmap >>= ((bitmap_idx + 1) * BITS_PER_BITMAP - e_bit);
      }

      index_t l_sum =
        g_sum + warp_exclusive_scan(static_cast<index_t>(raft::detail::popc(l_bitmap)));

      for (int i = 0; i < BITS_PER_BITMAP; i++) {
        if (l_bitmap & (ONE << i)) {
          indices[indptr_row + l_sum] = l_offset + i;
          l_sum++;
        }
      }
      g_sum = __shfl_sync(0xffffffff, l_sum, warpSize - 1);
    }
  }
}

template <typename bitmap_t, typename index_t, typename nnz_t, bool check_nnz = false>
void fill_indices_by_rows(raft::resources const& handle,
                          const bitmap_t* bitmap,
                          const index_t* indptr,
                          index_t num_rows,
                          index_t num_cols,
                          nnz_t nnz,
                          index_t* indices)
{
  auto stream              = resource::get_cuda_stream(handle);
  const index_t total      = num_rows * num_cols;
  const index_t bitmap_num = raft::ceildiv(total, index_t(sizeof(bitmap_t) * 8));

  int dev_id, sm_count, blocks_per_sm;

  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm,
    fill_indices_by_rows_kernel<bitmap_t, index_t, nnz_t, check_nnz>,
    fill_indices_by_rows_tpb,
    0);

  index_t max_active_blocks = sm_count * blocks_per_sm;
  auto grid                 = std::min(max_active_blocks, num_rows);
  auto block                = fill_indices_by_rows_tpb;

  fill_indices_by_rows_kernel<bitmap_t, index_t, nnz_t, check_nnz>
    <<<grid, block, 0, stream>>>(bitmap, indptr, num_rows, num_cols, nnz, bitmap_num, indices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename bitmap_t,
          typename index_t,
          typename csr_matrix_t,
          typename = std::enable_if_t<raft::is_device_csr_matrix_v<csr_matrix_t>>>
void bitmap_to_csr(raft::resources const& handle,
                   raft::core::bitmap_view<bitmap_t, index_t> bitmap,
                   csr_matrix_t& csr)
{
  auto csr_view = csr.structure_view();

  if (csr_view.get_n_rows() == 0 || csr_view.get_n_cols() == 0 || csr_view.get_nnz() == 0) {
    return;
  }

  RAFT_EXPECTS(bitmap.get_n_rows() == csr_view.get_n_rows(),
               "Number of rows in bitmap must be equal to "
               "number of rows in csr");

  RAFT_EXPECTS(bitmap.get_n_cols() == csr_view.get_n_cols(),
               "Number of columns in bitmap must be equal to "
               "number of columns in csr");

  auto thrust_policy = resource::get_thrust_policy(handle);
  auto stream        = resource::get_cuda_stream(handle);

  index_t* indptr  = csr_view.get_indptr().data();
  index_t* indices = csr_view.get_indices().data();

  RAFT_CUDA_TRY(cudaMemsetAsync(indptr, 0, (csr_view.get_n_rows() + 1) * sizeof(index_t), stream));

  calc_nnz_by_rows(handle, bitmap.data(), csr_view.get_n_rows(), csr_view.get_n_cols(), indptr);
  thrust::exclusive_scan(thrust_policy, indptr, indptr + csr_view.get_n_rows() + 1, indptr);

  if constexpr (is_device_csr_sparsity_owning_v<csr_matrix_t>) {
    index_t nnz = 0;
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &nnz, indptr + csr_view.get_n_rows(), sizeof(index_t), cudaMemcpyDeviceToHost, stream));
    resource::sync_stream(handle);
    csr.initialize_sparsity(nnz);
  }
  constexpr bool check_nnz = is_device_csr_sparsity_preserving_v<csr_matrix_t>;
  fill_indices_by_rows<bitmap_t, index_t, typename csr_matrix_t::nnz_type, check_nnz>(
    handle,
    bitmap.data(),
    indptr,
    csr_view.get_n_rows(),
    csr_view.get_n_cols(),
    csr_view.get_nnz(),
    indices);

  thrust::fill_n(thrust_policy,
                 csr.get_elements().data(),
                 csr_view.get_nnz(),
                 typename csr_matrix_t::element_type(1));
}

};  // end NAMESPACE detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
