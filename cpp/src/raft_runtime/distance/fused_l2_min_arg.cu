/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "fused_distance_min_arg.hpp"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/kvp.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/linalg/norm.cuh>

#include <thrust/for_each.h>
#include <thrust/tuple.h>

namespace raft::runtime::distance {

[[deprecated("use fused_distance_nn_min_arg instead")]] void fused_l2_nn_min_arg(
  raft::resources const& handle,
  int* min,
  const float* x,
  const float* y,
  int m,
  int n,
  int k,
  bool sqrt)
{
  compute_fused_l2_nn_min_arg<float, int>(handle, min, x, y, m, n, k, sqrt);
}

[[deprecated("use fused_distance_nn_min_arg instead")]] void fused_l2_nn_min_arg(
  raft::resources const& handle,
  int* min,
  const double* x,
  const double* y,
  int m,
  int n,
  int k,
  bool sqrt)
{
  compute_fused_l2_nn_min_arg<double, int>(handle, min, x, y, m, n, k, sqrt);
}

}  // end namespace raft::runtime::distance
