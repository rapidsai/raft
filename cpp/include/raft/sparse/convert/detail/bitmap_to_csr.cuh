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

#include <cooperative_groups.h>

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/sparse/convert/detail/adj_to_csr.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

// Threads per block in bitmap_to_any_kernel.
static const constexpr int bitmap_to_any_tpb = 512;

template <typename bitmap_t, typename index_t, typename any_t>
RAFT_KERNEL __launch_bounds__(bitmap_to_any_tpb)
  bitmap_to_any_kernel(const bitmap_t* bitmap, const index_t num_bits, any_t* index_array)
{
  index_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (index_t idx = thread_idx; idx < num_bits; idx += blockDim.x * gridDim.x) {
    bitmap_t element     = bitmap[idx / (8 * sizeof(bitmap_t))];
    index_t bit_position = idx % (8 * sizeof(bitmap_t));
    index_array[idx]     = static_cast<any_t>((element >> bit_position) & 1);
  }
}

template <typename bitmap_t, typename index_t, typename any_t>
void bitmap_to_any(raft::resources const& handle,
                   const bitmap_t* bitmap,
                   const index_t num_bits,
                   any_t* any_array)
{
  auto stream = resource::get_cuda_stream(handle);

  int dev_id, sm_count, blocks_per_sm;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, bitmap_to_any_kernel<bitmap_t, index_t, any_t>, bitmap_to_any_tpb, 0);

  auto grid  = sm_count * blocks_per_sm;
  auto block = bitmap_to_any_tpb;

  bitmap_to_any_kernel<bitmap_t, index_t, any_t>
    <<<grid, block, 0, stream>>>(bitmap, num_bits, any_array);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

// Threads per block in init_row_indicator_kernel.
static const constexpr int init_row_indicator_tpb = 512;

template <typename index_t>
RAFT_KERNEL __launch_bounds__(init_row_indicator_tpb)
  init_row_indicator_kernel(index_t num_cols, index_t total, index_t* row_indicator)
{
  index_t thread_idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (index_t idx = thread_idx; idx < total; idx += blockDim.x * gridDim.x) {
    row_indicator[idx] = idx / num_cols;
  }
}

template <typename index_t>
void init_row_indicator(raft::resources const& handle,
                        index_t num_cols,
                        index_t total,
                        index_t* row_indicator)
{
  auto stream = resource::get_cuda_stream(handle);

  int dev_id, sm_count, blocks_per_sm;
  cudaGetDevice(&dev_id);
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, init_row_indicator_kernel<index_t>, init_row_indicator_tpb, 0);

  auto grid  = sm_count * blocks_per_sm;
  auto block = init_row_indicator_tpb;

  init_row_indicator_kernel<index_t><<<grid, block, 0, stream>>>(num_cols, total, row_indicator);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename bitmap_t, typename index_t>
void bitmap_to_csr(raft::resources const& handle,
                   const bitmap_t* bitmap,
                   index_t num_rows,
                   index_t num_cols,
                   index_t* indptr,
                   index_t* indices)
{
  const index_t total = num_rows * num_cols;
  if (total == 0) { return; }

  auto thrust_policy = resource::get_thrust_policy(handle);
  auto stream        = resource::get_cuda_stream(handle);

  rmm::device_uvector<bool> bool_matrix(total, resource::get_cuda_stream(handle));
  rmm::device_uvector<index_t> int_matrix(total, resource::get_cuda_stream(handle));
  rmm::device_uvector<index_t> row_indicator(total, resource::get_cuda_stream(handle));

  bitmap_to_any(handle, bitmap, total, bool_matrix.data());
  bitmap_to_any(handle, bitmap, total, int_matrix.data());

  init_row_indicator(handle, num_cols, total, row_indicator.data());

  thrust::reduce_by_key(thrust_policy,
                        row_indicator.data(),
                        row_indicator.data() + total,
                        int_matrix.data(),
                        thrust::make_discard_iterator(),
                        indptr + 1,
                        thrust::equal_to<index_t>(),
                        thrust::plus<index_t>());
  // compute indptr
  thrust::inclusive_scan(thrust_policy, indptr, indptr + num_rows + 1, indptr);

  // compute indices
  adj_to_csr(handle, bool_matrix.data(), indptr, num_rows, num_cols, row_indicator.data(), indices);
}

};  // end NAMESPACE detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
