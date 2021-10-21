//
// Created by cjnolet on 10/21/21.
//

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <raft/linalg/distance_type.h>
#include <raft/distance/distance.hpp>

namespace raft {
namespace distance {
template void distance<raft::distance::DistanceType::CorrelationExpanded, float,
                       float, float, int>(const float *x, const float *y,
                                          float *dist, int m, int n, int k,
                                          void *workspace, size_t worksize,
                                          cudaStream_t stream, bool isRowMajor,
                                          float metric_arg);

template void distance<raft::distance::DistanceType::CorrelationExpanded,
                       double, double, double, int>(
  const double *x, const double *y, double *dist, int m, int n, int k,
  void *workspace, size_t worksize, cudaStream_t stream, bool isRowMajor,
  double metric_arg);

template void distance<raft::distance::DistanceType::CorrelationExpanded, float,
                       float, float, uint32_t>(
  const float *x, const float *y, float *dist, uint32_t m, uint32_t n,
  uint32_t k, void *workspace, size_t worksize, cudaStream_t stream,
  bool isRowMajor, float metric_arg);

template void distance<raft::distance::DistanceType::CorrelationExpanded,
                       double, double, double, uint32_t>(
  const double *x, const double *y, double *dist, uint32_t m, uint32_t n,
  uint32_t k, void *workspace, size_t worksize, cudaStream_t stream,
  bool isRowMajor, double metric_arg);
}  // namespace distance
}  // namespace raft
