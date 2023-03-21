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

#include "kernel.cuh"
#include <raft/distance/detail/pairwise_matrix/kernel_sm60.cuh>  // pairwise_matrix_sm60_wrapper
#include <raft/linalg/contractions.cuh>                          // raft::linalg::Policy4x4
#include <raft/util/arch.cuh>  // raft::util::arch::SM_compute_arch

namespace raft::bench::distance::tune {

// Distance op
using OpT                  = raft::distance::detail::ops::lp_unexp_distance_op<DataT, AccT, IdxT>;
constexpr float metric_arg = 2.0;
OpT distance_op{metric_arg};

// Kernel policy
constexpr int vec_len = 1;
using Policy          = typename raft::linalg::Policy4x4<DataT, vec_len>::Policy;

// Architecture
namespace arch                 = raft::util::arch;
constexpr auto sm_compat_range = arch::SM_range(arch::SM_min(), arch::SM_future());

void launch_kernel(pairwise_matrix_params params, dim3 grid, cudaStream_t stream)
{
  dim3 block(Policy::Nthreads);
  int smem_size = OpT::shared_mem_size<Policy>();

  // Obtain function pointer to kernel
  auto kernel = raft::distance::detail::pairwise_matrix_kernel<Policy,
                                                               row_major,
                                                               decltype(sm_compat_range),
                                                               OpT,
                                                               IdxT,
                                                               DataT,
                                                               OutT,
                                                               FinOpT>;

  kernel<<<grid, block, smem_size, stream>>>(distance_op, params);
  RAFT_CUDA_TRY(cudaGetLastError());
}

void get_block_size(int& m, int& n, int& k)
{
  m = Policy::Mblk;
  n = Policy::Nblk;
  k = Policy::Kblk;
}

void* get_kernel_ptr()
{
  auto kernel = raft::distance::detail::pairwise_matrix_kernel<Policy,
                                                               row_major,
                                                               decltype(sm_compat_range),
                                                               OpT,
                                                               IdxT,
                                                               DataT,
                                                               OutT,
                                                               FinOpT>;
  return reinterpret_cast<void*>(kernel);
}

int get_max_occupancy()
{
  void* kernel_ptr = get_kernel_ptr();
  int max_occupancy;
  int smem_size = OpT::shared_mem_size<Policy>();

  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_occupancy, kernel_ptr, Policy::Nthreads, smem_size));

  return max_occupancy;
}

}  // namespace raft::bench::distance::tune
