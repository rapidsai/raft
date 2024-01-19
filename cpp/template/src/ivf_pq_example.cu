/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <raft/neighbors/ivf_pq.cuh>
#include <raft/neighbors/refine.cuh>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cstdint>

void ivf_pq_build_search(raft::device_resources const& dev_resources,
                         raft::device_matrix_view<const float, int64_t> dataset,
                         raft::device_matrix_view<const float, int64_t> queries)
{
  using namespace raft::neighbors;  // NOLINT

  ivf_pq::index_params index_params;
  index_params.n_lists                  = 1024;
  index_params.kmeans_trainset_fraction = 0.1;
  index_params.metric                   = raft::distance::DistanceType::L2Expanded;
  index_params.pq_bits                  = 8;
  index_params.pq_dim                   = 2;

  std::cout << "Building IVF-PQ index" << std::endl;
  auto index = ivf_pq::build(dev_resources, index_params, dataset);

  std::cout << "Number of clusters " << index.n_lists() << ", number of vectors added to index "
            << index.size() << std::endl;

  // Set search parameters.
  ivf_pq::search_params search_params;
  search_params.n_probes = 50;
  // Set the internal search precision to 16-bit floats;
  // usually, this improves the performance at a slight cost to the recall.
  search_params.internal_distance_dtype = CUDA_R_16F;
  search_params.lut_dtype               = CUDA_R_16F;

  // Create output arrays.
  int64_t topk      = 10;
  int64_t n_queries = queries.extent(0);
  auto neighbors    = raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk);
  auto distances    = raft::make_device_matrix<float>(dev_resources, n_queries, topk);

  // Search K nearest neighbors for each of the queries.
  ivf_pq::search<float, int64_t>(
    dev_resources, search_params, index, queries, neighbors.view(), distances.view());

  // Re-ranking operation: refine the initial search results by computing exact distances
  int64_t topk_refined = 7;
  auto neighbors_refined =
    raft::make_device_matrix<int64_t>(dev_resources, n_queries, topk_refined);
  auto distances_refined = raft::make_device_matrix<float>(dev_resources, n_queries, topk_refined);

  // Note, refinement requires the original dataset and the queries.
  // Don't forget to specify the same distance metric as used by the index.
  raft::neighbors::refine(dev_resources,
                          dataset,
                          queries,
                          raft::make_const_mdspan(neighbors.view()),
                          neighbors_refined.view(),
                          distances_refined.view(),
                          index.metric());

  // Show both the original and the refined results
  std::cout << std::endl << "Original results:" << std::endl;
  print_results(dev_resources, neighbors.view(), distances.view());
  std::cout << std::endl << "Refined results:" << std::endl;
  print_results(dev_resources, neighbors_refined.view(), distances_refined.view());
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
  ivf_pq_build_search(dev_resources,
                      raft::make_const_mdspan(dataset.view()),
                      raft::make_const_mdspan(queries.view()));
}
