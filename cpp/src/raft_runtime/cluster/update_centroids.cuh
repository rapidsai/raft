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

#include <raft/cluster/kmeans.cuh>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/linalg/norm.cuh>

namespace raft::runtime::cluster::kmeans {

template <typename DataT, typename IndexT>
void update_centroids(raft::resources const& handle,
                      const DataT* X,
                      int n_samples,
                      int n_features,
                      int n_clusters,
                      const DataT* sample_weights,
                      const DataT* centroids,
                      const IndexT* labels,
                      DataT* new_centroids,
                      DataT* weight_per_cluster)
{
  auto X_view = raft::make_device_matrix_view<const DataT, IndexT>(X, n_samples, n_features);
  auto centroids_view =
    raft::make_device_matrix_view<const DataT, IndexT>(centroids, n_clusters, n_features);

  rmm::device_uvector<DataT> sample_weights_uvec(0, resource::get_cuda_stream(handle));
  if (sample_weights == nullptr) {
    sample_weights_uvec.resize(n_samples, resource::get_cuda_stream(handle));
    DataT weight = 1.0 / n_samples;
    thrust::fill(resource::get_thrust_policy(handle),
                 sample_weights_uvec.data(),
                 sample_weights_uvec.data() + n_samples,
                 weight);
  }
  auto sample_weights_view = raft::make_device_vector_view<const DataT, IndexT>(
    sample_weights == nullptr ? sample_weights_uvec.data() : sample_weights, n_samples);

  auto new_centroids_view =
    raft::make_device_matrix_view<DataT, IndexT>(new_centroids, n_clusters, n_features);
  rmm::device_uvector<DataT> weight_per_cluster_uvec(0, resource::get_cuda_stream(handle));
  if (weight_per_cluster == nullptr) {
    weight_per_cluster_uvec.resize(n_clusters, resource::get_cuda_stream(handle));
  }
  auto weight_per_cluster_view = raft::make_device_vector_view<DataT, IndexT>(
    weight_per_cluster == nullptr ? weight_per_cluster_uvec.data() : weight_per_cluster,
    n_clusters);

  raft::cluster::kmeans::update_centroids<DataT, IndexT>(handle,
                                                         X_view,
                                                         sample_weights_view,
                                                         centroids_view,
                                                         labels,
                                                         weight_per_cluster_view,
                                                         new_centroids_view);
}
}  // namespace raft::runtime::cluster::kmeans