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

#include <raft/distance/detail/distance_ops/all_ops.cuh>    // lp_unexp_distance_op
#include <raft/distance/detail/pairwise_matrix/params.cuh>  // pairwise_matrix_params

namespace raft::bench::distance::tune {

// Launch one specific kernel with the following template parameters
constexpr bool row_major = true;
using DataT              = float;
using AccT               = float;
using OutT               = DataT;
using IdxT               = int;

// Distance op
// C++17 inline variable. Used by both tuned_kernel.cu and tune_pairwise.cu
// See: https://open-std.org/JTC1/SC22/WG21/docs/papers/2016/p0386r0.pdf
using OpT                  = raft::distance::detail::ops::lp_unexp_distance_op<DataT, AccT, IdxT>;
constexpr float metric_arg = 2.0;
inline const OpT distance_op{metric_arg};
using FinOpT = raft::identity_op;

using pairwise_matrix_params =
  raft::distance::detail::pairwise_matrix_params<IdxT, DataT, OutT, FinOpT>;

// Launches kernel
void launch_kernel(OpT, pairwise_matrix_params, dim3, cudaStream_t);

// Describes the block size that is decided by the policy
void get_block_size(int& m, int& n, int& k);

void* get_kernel_ptr();
int get_max_occupancy(OpT);

}  // namespace raft::bench::distance::tune
