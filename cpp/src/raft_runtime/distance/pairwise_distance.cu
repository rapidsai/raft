/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>

namespace raft::runtime::distance {

void pairwise_distance(raft::resources const& handle,
                       float* x,
                       float* y,
                       float* dists,
                       int m,
                       int n,
                       int k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor,
                       float metric_arg)
{
  raft::distance::pairwise_distance<float, int>(
    handle, x, y, dists, m, n, k, metric, isRowMajor, metric_arg);
}

void pairwise_distance(raft::resources const& handle,
                       double* x,
                       double* y,
                       double* dists,
                       int m,
                       int n,
                       int k,
                       raft::distance::DistanceType metric,
                       bool isRowMajor,
                       float metric_arg)
{
  raft::distance::pairwise_distance<double, int>(
    handle, x, y, dists, m, n, k, metric, isRowMajor, metric_arg);
}
}  // namespace raft::runtime::distance