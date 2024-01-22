/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include "common.cuh"
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/thrust_policy.hpp>
#include <raft/neighbors/ivf_flat.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#include <cstdint>
#include <optional>

void ivf_flat_build_search_simple(raft::device_resources const& dev_resources,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace raft::neighbors;

  ivf_flat::index_params index_params;
  index_params.n_lists                  = 1024;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.metric                   = raft::distance::DistanceType::L2Expanded;

  std::cout << "Building IVF-Flat index" << std::endl;
  auto index = ivf_flat::build(dev_resources, index_params, dataset);

  std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;

  // Create output arrays.
  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
  auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Set search parameters.
  ivf_flat::search_params search_params;
  search_params.n_probes = 50;

  // Search K nearest neighbors for each of the queries.
  ivf_flat::search(
    dev_resources, search_params, index, queries, neighbors.view(), distances.view());

  // The call to ivf_flat::search is asynchronous. Before accessing the data, sync by calling
  // raft::resource::sync_stream(dev_resources);

  print_results(dev_resources, neighbors.view(), distances.view());
}

void ivf_flat_build_extend_search(raft::device_resources const& dev_resources,
                                  raft::device_matrix_view<const float, int64_t> dataset,
                                  raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace raft::neighbors;

  // Define dataset indices.
  auto data_indices = raft::make_device_vector<int64_t, int64_t>(dev_resources, dataset.extent(0));
  thrust::counting_iterator<int64_t> first(0);
  thrust::device_ptr<int64_t> ptr(data_indices.data_handle());
  thrust::copy(
    raft::resource::get_thrust_policy(dev_resources), first, first + dataset.extent(0), ptr);

  // Sub-sample the dataset to create a training set.
  auto trainset =
    subsample(dev_resources, dataset, raft::make_const_mdspan(data_indices.view()), 0.1);

  ivf_flat::index_params index_params;
  index_params.n_lists           = 100;
  index_params.metric            = raft::distance::DistanceType::L2Expanded;
  index_params.add_data_on_build = false;

  std::cout << "\nRun k-means clustering using the training set" << std::endl;
  auto index =
    ivf_flat::build(dev_resources, index_params, raft::make_const_mdspan(trainset.view()));

  std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;

  std::cout << "Filling index with the dataset vectors" << std::endl;
  index = ivf_flat::extend(dev_resources,
                           dataset,
                           std::make_optional(raft::make_const_mdspan(data_indices.view())),
                           index);

  std::cout << "Index size after addin dataset vectors " << index.size() << std::endl;

  // Set search parameters.
  ivf_flat::search_params search_params;
  search_params.n_probes = 10;

  // Create output arrays.
  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t, int64_t>(dev_resources, n_queries, topk);
  auto distances    = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, topk);

  // Search K nearest neighbors for each queries.
  ivf_flat::search(
    dev_resources, search_params, index, queries, neighbors.view(), distances.view());

  // The call to ivf_flat::search is asynchronous. Before accessing the data, sync using:
  // raft::resource::sync_stream(dev_resources);

  print_results(dev_resources, neighbors.view(), distances.view());
}

int main()
{
  raft::device_resources dev_resources;

  // Set pool memory resource with 1 GiB initial pool size. All allocations use the same pool.
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    rmm::mr::get_current_device_resource(), 1024 * 1024 * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  // Alternatively, one could define a pool allocator for temporary arrays (used within RAFT
  // algorithms). In that case only the internal arrays would use the pool, any other allocation
  // uses the default RMM memory resource. Here is how to change the workspace memory resource to
  // a pool with 2 GiB upper limit.
  // raft::resource::set_workspace_to_pool_resource(dev_resources, 2 * 1024 * 1024 * 1024ull);

  // Create input arrays.
  int64_t n_samples = 10000;
  int64_t n_dim     = 3;
  int64_t n_queries = 10;
  auto dataset      = raft::make_device_matrix<float, int64_t>(dev_resources, n_samples, n_dim);
  auto queries      = raft::make_device_matrix<float, int64_t>(dev_resources, n_queries, n_dim);
  generate_dataset(dev_resources, dataset.view(), queries.view());

  // Simple build and search example.
  ivf_flat_build_search_simple(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(queries.view()));

  // Build and extend example.
  ivf_flat_build_extend_search(dev_resources,
                               raft::make_const_mdspan(dataset.view()),
                               raft::make_const_mdspan(queries.view()));
}
