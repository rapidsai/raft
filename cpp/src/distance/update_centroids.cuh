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

#include <raft/cluster/kmeans.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/specializations.cuh>
#include <raft/handle.hpp>

namespace raft::cluster::kmeans::runtime {

template <typename DataT, typename IndexT>
void update_centroids(raft::handle_t const& handle,
                      const DataT* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const DataT* centroids,
                      const DataT* weight,
                      const DataT* l2norm_x,
                      DataT* new_centroids,
                      DataT* new_weight,
                      raft::distance::DistanceType metric,
                      int batch_samples,
                      int batch_centroids)
{
  auto min_cluster_and_dist =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
  auto X_view = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<const DataT, IndexT>(centroids, n_clusters, n_features);
  auto weight_view   = raft::make_device_vector_view<const DataT, IndexT>(weight, n_clusters);
  auto l2norm_x_view = raft::make_device_vector_view<const DataT, IndexT>(l2norm_x, n_samples);
  auto new_centroids_view =
    raft::make_device_matrix_view<DataT, IndexT>(new_centroids, n_clusters, n_features);
  auto new_weight_view = raft::make_device_vector_view<DataT, IndexT>(new_weight, n_clusters);

  raft::cluster::kmeans::update_centroids<DataT, IndexT>(handle,
                                                         X_view,
                                                         centroids_view,
                                                         weight_view,
                                                         l2norm_x_view,
                                                         min_cluster_and_dist.view(),
                                                         new_centroids_view,
                                                         new_weight_view,
                                                         metric,
                                                         batch_samples,
                                                         batch_centroids);
}
}  // namespace raft::cluster::kmeans::runtime