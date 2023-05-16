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

#include "kernel_cutlass.cuh"
#include <raft/distance/detail/distance_ops/all_ops.cuh>  // distance_op
#include <raft/distance/detail/pairwise_matrix/dispatch_sm80.cuh>
#include <raft/distance/detail/pairwise_matrix/params.cuh>
#include <raft/distance/distance_types.hpp>  // Compute_options
#include <raft/util/arch.cuh>                // raft::util::arch::SM_compute_arch

namespace raft::bench::distance::tune_cutlass {

// Distance op
using OpT = raft::distance::detail::ops::l2_exp_distance_op<DataT, AccT, IdxT>;

constexpr bool perform_sqrt = false;
OpT distance_op{perform_sqrt};

// Architecture
namespace arch                 = raft::util::arch;
constexpr auto sm_compat_range = arch::SM_range(arch::SM_80(), arch::SM_future());

void launch_kernel(pairwise_matrix_params params, bool use_1x_tfloat, cudaStream_t stream)
{
  raft::distance::detail::pairwise_matrix_sm80_dispatch(
    distance_op,
    use_1x_tfloat ? raft::distance::Compute_options::Fast_Reduced_Precision
                  : raft::distance::Compute_options::Fast_Similar_Precision,
    params,
    sm_compat_range,
    stream);
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace raft::bench::distance::tune_cutlass
