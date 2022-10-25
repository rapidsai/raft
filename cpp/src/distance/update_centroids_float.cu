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

#include "update_centroids.cuh"
#include <raft/distance/distance_types.hpp>
#include <raft/distance/specializations.cuh>
#include <raft/handle.hpp>

namespace raft::cluster::kmeans::runtime {

void update_centroids(raft::handle_t const& handle,
                      const float* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const float* centroids,
                      const float* weight,
                      const float* l2norm_x,
                      float* new_centroids,
                      float* new_weight,
                      raft::distance::DistanceType metric,
                      int batch_samples,
                      int batch_centroids)
{
  update_centroids<float, int>(handle,
                               X,
                               n_samples,
                               n_features,
                               n_clusters,
                               centroids,
                               weight,
                               l2norm_x,
                               new_centroids,
                               new_weight,
                               metric,
                               batch_samples,
                               batch_centroids);
}

}  // namespace raft::cluster::kmeans::runtime