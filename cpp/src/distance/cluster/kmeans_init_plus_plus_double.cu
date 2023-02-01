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
#include <raft/core/device_resources.hpp>
#include <raft/distance/specializations.cuh>

namespace raft::runtime::cluster::kmeans {

void init_plus_plus(raft::device_resources const& handle,
                    const raft::cluster::kmeans::KMeansParams& params,
                    raft::device_matrix_view<const double, int> X,
                    raft::device_matrix_view<double, int> centroids)
{
  rmm::device_uvector<char> workspace(0, handle.get_stream());
  raft::cluster::kmeans::init_plus_plus<double, int>(handle, params, X, centroids, workspace);
}
}  // namespace raft::runtime::cluster::kmeans
