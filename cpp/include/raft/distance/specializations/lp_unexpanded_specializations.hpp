/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <raft/distance/distance.hpp>

namespace raft {
namespace distance {
extern template void
distance<raft::distance::DistanceType::LpUnexpanded, float, float, float, int>(
  const float *x, const float *y, float *dist, int m, int n, int k,
  void *workspace, size_t worksize, cudaStream_t stream, bool isRowMajor,
  float metric_arg);

extern template void distance<raft::distance::DistanceType::LpUnexpanded,
                              double, double, double, int>(
  const double *x, const double *y, double *dist, int m, int n, int k,
  void *workspace, size_t worksize, cudaStream_t stream, bool isRowMajor,
  double metric_arg);

extern template void
distance<raft::distance::DistanceType::LpUnexpanded, float, float, float, int>(
  const float *x, const float *y, float *dist, int m, int n, int k,
  cudaStream_t stream, bool isRowMajor, float metric_arg);

extern template void distance<raft::distance::DistanceType::LpUnexpanded,
                              double, double, double, int>(
  const double *x, const double *y, double *dist, int m, int n, int k,
  cudaStream_t stream, bool isRowMajor, double metric_arg);

}  // namespace distance
}  // namespace raft
