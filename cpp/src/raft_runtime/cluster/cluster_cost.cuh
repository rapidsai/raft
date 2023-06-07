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
#include <raft/core/operators.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/distance/fused_l2_nn.cuh>
#include <raft/util/cuda_utils.cuh>

namespace raft::runtime::cluster::kmeans {
template <typename ElementType, typename IndexType>
void cluster_cost(raft::resources const& handle,
                  const ElementType* X,
                  IndexType n_samples,
                  IndexType n_features,
                  IndexType n_clusters,
                  const ElementType* centroids,
                  ElementType* cost)
{
  rmm::device_uvector<char> workspace(n_samples * sizeof(IndexType),
                                      resource::get_cuda_stream(handle));

  rmm::device_uvector<ElementType> x_norms(n_samples, resource::get_cuda_stream(handle));
  rmm::device_uvector<ElementType> centroid_norms(n_clusters, resource::get_cuda_stream(handle));
  raft::linalg::rowNorm(x_norms.data(),
                        X,
                        n_features,
                        n_samples,
                        raft::linalg::L2Norm,
                        true,
                        resource::get_cuda_stream(handle));
  raft::linalg::rowNorm(centroid_norms.data(),
                        centroids,
                        n_features,
                        n_clusters,
                        raft::linalg::L2Norm,
                        true,
                        resource::get_cuda_stream(handle));

  auto min_cluster_distance =
    raft::make_device_vector<raft::KeyValuePair<IndexType, ElementType>>(handle, n_samples);
  raft::distance::fusedL2NNMinReduce(min_cluster_distance.data_handle(),
                                     X,
                                     centroids,
                                     x_norms.data(),
                                     centroid_norms.data(),
                                     n_samples,
                                     n_clusters,
                                     n_features,
                                     (void*)workspace.data(),
                                     false,
                                     true,
                                     resource::get_cuda_stream(handle));

  auto distances = raft::make_device_vector<ElementType, IndexType>(handle, n_samples);
  thrust::transform(resource::get_thrust_policy(handle),
                    min_cluster_distance.data_handle(),
                    min_cluster_distance.data_handle() + n_samples,
                    distances.data_handle(),
                    raft::value_op{});

  rmm::device_scalar<ElementType> device_cost(0, resource::get_cuda_stream(handle));
  raft::cluster::kmeans::cluster_cost(handle,
                                      distances.view(),
                                      workspace,
                                      make_device_scalar_view<ElementType>(device_cost.data()),
                                      raft::add_op{});

  raft::update_host(cost, device_cost.data(), 1, resource::get_cuda_stream(handle));
}
}  // namespace raft::runtime::cluster::kmeans
