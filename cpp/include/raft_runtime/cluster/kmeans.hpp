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

#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::runtime::cluster::kmeans {

/**
 * @defgroup kmeans_runtime Kmeans Runtime API
 * @{
 */

void update_centroids(raft::resources const& handle,
                      const float* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const float* sample_weights,
                      const float* centroids,
                      const int* labels,
                      float* new_centroids,
                      float* weight_per_cluster);

void update_centroids(raft::resources const& handle,
                      const double* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const double* sample_weights,
                      const double* centroids,
                      const int* labels,
                      double* new_centroids,
                      double* weight_per_cluster);

void fit(raft::resources const& handle,
         const raft::cluster::kmeans::KMeansParams& params,
         raft::device_matrix_view<const float, int, row_major> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int, row_major> centroids,
         raft::host_scalar_view<float, int> inertia,
         raft::host_scalar_view<int, int> n_iter);

void fit(raft::resources const& handle,
         const raft::cluster::kmeans::KMeansParams& params,
         raft::device_matrix_view<const double, int, row_major> X,
         std::optional<raft::device_vector_view<const double, int>> sample_weight,
         raft::device_matrix_view<double, int, row_major> centroids,
         raft::host_scalar_view<double, int> inertia,
         raft::host_scalar_view<int, int> n_iter);

void init_plus_plus(raft::resources const& handle,
                    const raft::cluster::kmeans::KMeansParams& params,
                    raft::device_matrix_view<const float, int, row_major> X,
                    raft::device_matrix_view<float, int, row_major> centroids);

void init_plus_plus(raft::resources const& handle,
                    const raft::cluster::kmeans::KMeansParams& params,
                    raft::device_matrix_view<const double, int, row_major> X,
                    raft::device_matrix_view<double, int, row_major> centroids);

void cluster_cost(raft::resources const& handle,
                  const float* X,
                  int n_samples,
                  int n_features,
                  int n_clusters,
                  const float* centroids,
                  float* cost);

void cluster_cost(raft::resources const& handle,
                  const double* X,
                  int n_samples,
                  int n_features,
                  int n_clusters,
                  const double* centroids,
                  double* cost);

/** @} */  // end group kmeans_runtime

}  // namespace raft::runtime::cluster::kmeans
