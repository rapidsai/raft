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
#include <raft/linalg/norm.cuh>

namespace raft::cluster::kmeans::runtime {

template <typename DataT, typename IndexT>
void update_centroids(raft::handle_t const& handle,
                      const DataT* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const DataT* sample_weights,
                      const DataT* l2norm_x,
                      const DataT* centroids,
                      DataT* new_centroids,
                      DataT* weight_per_cluster,
                      raft::distance::DistanceType metric,
                      int batch_samples,
                      int batch_centroids)
{
  auto min_cluster_and_dist =
    raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);
  auto X_view = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<const DataT, IndexT>(centroids, n_clusters, n_features);

  rmm::device_uvector<DataT> sample_weights_uvec(0, handle.get_stream());
  if (sample_weights == nullptr) {
    sample_weights_uvec.resize(n_samples, handle.get_stream());
    DataT weight = 1.0 / n_samples;
    thrust::fill(handle.get_thrust_policy(),
                 sample_weights_uvec.data(),
                 sample_weights_uvec.data() + n_samples,
                 weight);
  }
  auto sample_weights_view = raft::make_device_vector_view<const DataT, IndexT>(
    sample_weights == nullptr ? sample_weights_uvec.data() : sample_weights, n_samples);

  rmm::device_uvector<DataT> l2norm_x_uvec(0, handle.get_stream());
  if (l2norm_x == nullptr) {
    l2norm_x_uvec.resize(n_samples, handle.get_stream());
    raft::linalg::rowNorm(l2norm_x_uvec.data(),
                          X,
                          n_samples,
                          n_features,
                          raft::linalg::L2Norm,
                          true,
                          handle.get_stream());
  }
  auto l2norm_x_view = raft::make_device_vector_view<const DataT, IndexT>(
    l2norm_x == nullptr ? l2norm_x_uvec.data() : l2norm_x, n_samples);

  auto new_centroids_view =
    raft::make_device_matrix_view<DataT, IndexT>(new_centroids, n_clusters, n_features);
  rmm::device_uvector<DataT> weight_per_cluster_uvec(0, handle.get_stream());
  if (weight_per_cluster == nullptr) {
    weight_per_cluster_uvec.resize(n_clusters, handle.get_stream());
  }
  auto weight_per_cluster_view = raft::make_device_vector_view<DataT, IndexT>(
    weight_per_cluster == nullptr ? weight_per_cluster_uvec.data() : weight_per_cluster,
    n_clusters);

  raft::cluster::kmeans::update_centroids<DataT, IndexT>(handle,
                                                         X_view,
                                                         sample_weights_view,
                                                         l2norm_x_view,
                                                         centroids_view,
                                                         min_cluster_and_dist.view(),
                                                         weight_per_cluster_view,
                                                         new_centroids_view,
                                                         metric,
                                                         batch_samples,
                                                         batch_centroids);
}
}  // namespace raft::cluster::kmeans::runtime