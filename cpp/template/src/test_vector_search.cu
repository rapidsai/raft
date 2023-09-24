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
#include <raft/neighbors/cagra.cuh>
#include <raft/random/make_blobs.cuh>

int main()
{
  using namespace raft::neighbors;
  raft::device_resources dev_resources;
  // Use 5 GB of pool memory
  raft::resource::set_workspace_to_pool_resource(
    dev_resources, std::make_optional<std::size_t>(5 * 1024 * 1024 * 1024ull));

  int64_t n_samples = 50000;
  int64_t n_dim     = 90;
  int64_t topk      = 12;
  int64_t n_queries = 1;

  // create input and output arrays
  auto input     = raft::make_device_matrix<float>(dev_resources, n_samples, n_dim);
  auto labels    = raft::make_device_vector<int64_t>(dev_resources, n_samples);
  auto queries   = raft::make_device_matrix<float>(dev_resources, n_queries, n_dim);
  auto neighbors = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
  auto distances = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  raft::random::make_blobs(dev_resources, input.view(), labels.view());

  // use default index parameters
  cagra::index_params index_params;
  // create and fill the index from a [n_samples, n_dim] input
  auto index = cagra::build<float, int64_t>(
    dev_resources, index_params, raft::make_const_mdspan(input.view()));
  // use default search parameters
  cagra::search_params search_params;
  // search K nearest neighbors
  cagra::search<float, int64_t>(dev_resources,
                                search_params,
                                index,
                                raft::make_const_mdspan(queries.view()),
                                neighbors.view(),
                                distances.view());
}
