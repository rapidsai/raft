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

#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::runtime::distance {

/**
 * @defgroup fused_l2_nn_min_arg_runtime Fused L2 1NN Runtime API
 * @{
 */

/**
 * @brief Wrapper around fusedL2NN with minimum reduction operators.
 *
 * fusedL2NN cannot be compiled in the distance library due to the lambda
 * operators, so this wrapper covers the most common case (minimum).
 *
 * @param[in] handle         raft handle
 * @param[out] min           will contain the reduced output (Length = `m`)
 *                           (on device)
 * @param[in]  x             first matrix. Row major. Dim = `m x k`.
 *                           (on device).
 * @param[in]  y             second matrix. Row major. Dim = `n x k`.
 *                           (on device).
 * @param[in]  m             gemm m
 * @param[in]  n             gemm n
 * @param[in]  k             gemm k
 * @param[in]  sqrt          Whether the output `minDist` should contain L2-sqrt
 */
[[deprecated("use fused_distance_nn_min_arg instead")]] void fused_l2_nn_min_arg(
  raft::resources const& handle,
  int* min,
  const float* x,
  const float* y,
  int m,
  int n,
  int k,
  bool sqrt);

[[deprecated("use fused_distance_nn_min_arg instead")]] void fused_l2_nn_min_arg(
  raft::resources const& handle,
  int* min,
  const double* x,
  const double* y,
  int m,
  int n,
  int k,
  bool sqrt);

/** @} */  // end group fused_l2_nn_min_arg_runtime

}  // end namespace raft::runtime::distance
