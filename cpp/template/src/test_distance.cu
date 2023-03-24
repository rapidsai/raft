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

#include <cstdint>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/distance/distance.cuh>
#include <raft/random/make_blobs.cuh>

#ifdef RAFT_COMPILED
#include <raft/distance/specializations.cuh>
#endif

int main()
{
  raft::device_resources handle;

  int n_samples  = 5000;
  int n_features = 50;

  auto input  = raft::make_device_matrix<float, int>(handle, n_samples, n_features);
  auto labels = raft::make_device_vector<int, int>(handle, n_samples);
  auto output = raft::make_device_matrix<float, int>(handle, n_samples, n_samples);

  raft::random::make_blobs(handle, input.view(), labels.view());

  auto metric = raft::distance::DistanceType::L2SqrtExpanded;
  raft::distance::pairwise_distance(handle, input.view(), input.view(), output.view(), metric);
}
