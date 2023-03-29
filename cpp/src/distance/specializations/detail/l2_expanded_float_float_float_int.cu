/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/core/operators.hpp>                            // raft::identity_op
#include <raft/distance/detail/distance_ops/all_ops.cuh>      // ops::*
#include <raft/distance/detail/pairwise_matrix/dispatch.cuh>  // pairwise_matrix_instantiation_point
#include <raft/distance/detail/pairwise_matrix/dispatch_sm60.cuh>
#include <raft/distance/detail/pairwise_matrix/dispatch_sm80.cuh>

namespace raft::distance::detail {

template void pairwise_matrix_instantiation_point<ops::l2_exp_distance_op<float, float, int>,
                                                  int,
                                                  float,
                                                  float,
                                                  decltype(raft::identity_op())>(
  ops::l2_exp_distance_op<float, float, int>,
  pairwise_matrix_params<int, float, float, decltype(raft::identity_op())>,
  cudaStream_t);

}  // namespace raft::distance::detail
