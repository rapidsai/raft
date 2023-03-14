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

#pragma once

#include <raft/distance/detail/distance.cuh>

namespace raft::distance::detail {

extern template void pairwise_matrix_instantiation_point<ops::l1_distance_op<float, float, int>,
                                                         int,
                                                         float,
                                                         float,
                                                         decltype(raft::identity_op())>(
  ops::l1_distance_op<float, float, int>,
  pairwise_matrix_params<int, float, float, decltype(raft::identity_op())>,
  cudaStream_t);

extern template void pairwise_matrix_instantiation_point<ops::l1_distance_op<double, double, int>,
                                                         int,
                                                         double,
                                                         double,
                                                         decltype(raft::identity_op())>(
  ops::l1_distance_op<double, double, int>,
  pairwise_matrix_params<int, double, double, decltype(raft::identity_op())>,
  cudaStream_t);
}  // namespace raft::distance::detail
