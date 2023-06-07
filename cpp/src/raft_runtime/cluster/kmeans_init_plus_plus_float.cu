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
#include <raft/core/resources.hpp>

namespace raft::runtime::cluster::kmeans {

void init_plus_plus(raft::resources const& handle,
                    const raft::cluster::kmeans::KMeansParams& params,
                    raft::device_matrix_view<const float, int> X,
                    raft::device_matrix_view<float, int> centroids)
{
  rmm::device_uvector<char> workspace(0, resource::get_cuda_stream(handle));
  raft::cluster::kmeans::init_plus_plus<float, int>(handle, params, X, centroids, workspace);
}
}  // namespace raft::runtime::cluster::kmeans
