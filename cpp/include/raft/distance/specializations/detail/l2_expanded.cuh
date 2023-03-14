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

extern template void
pairwise_matrix_dispatch<ops::l2_exp_distance_op<float, float, int>,
                         float,
                         float,
                         float,
                         decltype(raft::identity_op()),
                         int,
                         raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_80>>(
  ops::l2_exp_distance_op<float, float, int>,
  int,
  int,
  int,
  const float*,
  const float*,
  const float*,
  const float*,
  float*,
  decltype(raft::identity_op()),
  cudaStream_t,
  bool);
extern template void
pairwise_matrix_dispatch<ops::l2_exp_distance_op<double, double, int>,
                         double,
                         double,
                         double,
                         decltype(raft::identity_op()),
                         int,
                         raft::arch::SM_range<raft::arch::SM_min, raft::arch::SM_80>>(
  ops::l2_exp_distance_op<double, double, int>,
  int,
  int,
  int,
  const double*,
  const double*,
  const double*,
  const double*,
  double*,
  decltype(raft::identity_op()),
  cudaStream_t,
  bool);
}  // namespace raft::distance::detail
