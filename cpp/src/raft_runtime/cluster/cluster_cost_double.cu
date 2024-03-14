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

#include "cluster_cost.cuh"

#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::runtime::cluster::kmeans {

void cluster_cost(raft::resources const& handle,
                  const double* X,
                  int n_samples,
                  int n_features,
                  int n_clusters,
                  const double* centroids,
                  double* cost)
{
  cluster_cost<double, int>(handle, X, n_samples, n_features, n_clusters, centroids, cost);
}
}  // namespace raft::runtime::cluster::kmeans
