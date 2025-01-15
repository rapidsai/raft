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
#include <raft/sparse/convert/detail/bitmap_to_csr.cuh>

#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

#include <assert.h>

namespace raft {
namespace sparse {
namespace convert {
namespace detail {

template <typename index_t, typename nnz_t>
RAFT_KERNEL repeat_csr_kernel(const index_t* indptr,
                              const index_t* indices,
                              index_t* repeated_indptr,
                              index_t* repeated_indices,
                              nnz_t nnz,
                              index_t repeat_count)
{
  int global_id                  = blockIdx.x * blockDim.x + threadIdx.x;
  bool guard                     = global_id < nnz;
  index_t* repeated_indices_addr = repeated_indices + global_id;

  for (index_t i = global_id; i < repeat_count; i += gridDim.x * blockDim.x) {
    repeated_indptr[i] = (i + 2) * nnz;
  }

  __syncthreads();

  index_t item;
  item = (global_id < nnz) ? indices[global_id] : -1;

  __syncthreads();

  for (index_t row = 0; row < repeat_count; ++row) {
    index_t start_offset = row * nnz;
    if (guard) { repeated_indices_addr[start_offset] = item; }
  }
}

template <typename index_t, typename nnz_t>
void gpu_repeat_csr(raft::resources const& handle,
                    const index_t* d_indptr,
                    const index_t* d_indices,
                    nnz_t nnz,
                    index_t repeat_count,
                    index_t* d_repeated_indptr,
                    index_t* d_repeated_indices)
{
  if (nnz == 0) return;

  auto stream            = resource::get_cuda_stream(handle);
  index_t repeat_csr_tpb = 256;
  index_t grid           = (nnz + repeat_csr_tpb - 1) / (repeat_csr_tpb);

  repeat_csr_kernel<<<grid, repeat_csr_tpb, 0, stream>>>(
    d_indptr, d_indices, d_repeated_indptr, d_repeated_indices, nnz, repeat_count);
}

template <typename bitset_t,
          typename index_t,
          typename csr_matrix_t,
          typename = std::enable_if_t<raft::is_device_csr_matrix_v<csr_matrix_t>>>
void bitset_to_csr(raft::resources const& handle,
                   raft::core::bitset_view<bitset_t, index_t> bitset,
                   csr_matrix_t& csr)
{
  using row_t = typename csr_matrix_t::row_type;
  using nnz_t = typename csr_matrix_t::nnz_type;

  auto csr_view = csr.structure_view();

  RAFT_EXPECTS(bitset.size() == csr_view.get_n_cols(),
               "Number of size in bitset must be equal to "
               "number of columns in csr");
  if (csr_view.get_n_rows() == 0 || csr_view.get_n_cols() == 0) { return; }

  auto thrust_policy = resource::get_thrust_policy(handle);
  auto stream        = resource::get_cuda_stream(handle);

  index_t* indptr  = csr_view.get_indptr().data();
  index_t* indices = csr_view.get_indices().data();

  RAFT_CUDA_TRY(cudaMemsetAsync(indptr, 0, (csr_view.get_n_rows() + 1) * sizeof(index_t), stream));

  size_t sub_nnz_size      = 0;
  index_t bits_per_sub_col = 0;

  // Get buffer size and number of bits per each sub-columns
  calc_nnz_by_rows(handle,
                   bitset.data(),
                   row_t(1),
                   csr_view.get_n_cols(),
                   static_cast<nnz_t*>(nullptr),
                   sub_nnz_size,
                   bits_per_sub_col);

  rmm::device_async_resource_ref device_memory = resource::get_workspace_resource(handle);
  rmm::device_uvector<nnz_t> sub_nnz(sub_nnz_size + 1, stream, device_memory);

  calc_nnz_by_rows(handle,
                   bitset.data(),
                   row_t(1),
                   csr_view.get_n_cols(),
                   sub_nnz.data(),
                   sub_nnz_size,
                   bits_per_sub_col);

  thrust::exclusive_scan(
    thrust_policy, sub_nnz.data(), sub_nnz.data() + sub_nnz_size + 1, sub_nnz.data());

  nnz_t bitset_nnz = 0;
  if constexpr (is_device_csr_sparsity_owning_v<csr_matrix_t>) {
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      &bitset_nnz, sub_nnz.data() + sub_nnz_size, sizeof(nnz_t), cudaMemcpyDeviceToHost, stream));
    resource::sync_stream(handle);
    csr.initialize_sparsity(bitset_nnz * csr_view.get_n_rows());
    if (bitset_nnz == 0) return;
  } else {
    bitset_nnz = csr_view.get_nnz() / csr_view.get_n_rows();
  }

  constexpr bool check_nnz = is_device_csr_sparsity_preserving_v<csr_matrix_t>;
  fill_indices_by_rows<bitset_t, index_t, nnz_t, check_nnz>(handle,
                                                            bitset.data(),
                                                            indptr,
                                                            1,
                                                            csr_view.get_n_cols(),
                                                            csr_view.get_nnz(),
                                                            indices,
                                                            sub_nnz.data(),
                                                            bits_per_sub_col,
                                                            sub_nnz_size);
  if (csr_view.get_n_rows() > 1) {
    gpu_repeat_csr<index_t, nnz_t>(handle,
                                   indptr,
                                   indices,
                                   bitset_nnz,
                                   csr_view.get_n_rows() - 1,
                                   indptr + 2,
                                   indices + bitset_nnz);
  }

  thrust::fill_n(thrust_policy,
                 csr.get_elements().data(),
                 csr_view.get_nnz(),
                 typename csr_matrix_t::element_type(1));
}

};  // end NAMESPACE detail
};  // end NAMESPACE convert
};  // end NAMESPACE sparse
};  // end NAMESPACE raft
