/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::cluster::kmeans::runtime {

void update_centroids(raft::handle_t const& handle,
                      const float* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const float* sample_weights,
                      const float* centroids,
                      const int* labels,
                      float* new_centroids,
                      float* weight_per_cluster);

void update_centroids(raft::handle_t const& handle,
                      const double* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const double* sample_weights,
                      const double* centroids,
                      const int* labels,
                      double* new_centroids,
                      double* weight_per_cluster);

}  // namespace raft::cluster::kmeans::runtime