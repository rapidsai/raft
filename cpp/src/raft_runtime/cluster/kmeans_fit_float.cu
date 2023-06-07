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
#include <raft/core/resources.hpp>

namespace raft::runtime::cluster::kmeans {

void fit(raft::resources const& handle,
         const raft::cluster::kmeans::KMeansParams& params,
         raft::device_matrix_view<const float, int> X,
         std::optional<raft::device_vector_view<const float, int>> sample_weight,
         raft::device_matrix_view<float, int> centroids,
         raft::host_scalar_view<float, int> inertia,
         raft::host_scalar_view<int, int> n_iter)
{
  raft::cluster::kmeans::fit<float, int>(
    handle, params, X, sample_weight, centroids, inertia, n_iter);
}
}  // namespace raft::runtime::cluster::kmeans
