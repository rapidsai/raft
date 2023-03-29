/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once

#include "search_core.cuh"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/cagra/cagra.hpp>
#include <raft/neighbors/detail/cagra/search_plan.cuh>

// #include <raft/neighbors/detail/cagra/search_core.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace raft::neighbors::experimental::cagra::detail {

/**
 * @brief Search ANN using the constructed index.
 *
 * See the [build](#build) documentation for a usage example.
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices
 *
 * @param[in] handle
 * @param[in] params configure the search
 * @param[in] idx ivf-pq constructed index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 */

template <typename T, typename IdxT>
void search_main(raft::device_resources const& handle,
                 search_params params,
                 const index<T, IdxT>& index,
                 raft::device_matrix_view<const T, IdxT, row_major> queries,
                 raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
                 raft::device_matrix_view<float, IdxT, row_major> distances)
{
  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(index.dataset().extent(0)),
                 static_cast<size_t>(index.dataset().extent(1)));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  RAFT_EXPETS(queries.extent(1) == index.dim(), "Querise and index dim must match");

  search_plan splan(handle, params, index.dim(), index.graph_degree());
  const std::uint32_t topk = neighbors.extent(1);
  splan.check(topk);

  params                  = splan.plan;
  const std::string dtype = "float";  // tamas remove
  // Allocate memory for stats
  std::uint32_t* num_executed_iterations = nullptr;
  RAFT_CUDA_TRY(
    cudaMallocHost(&num_executed_iterations, sizeof(std::uint32_t) * queries.extent(0)));

  RAFT_LOG_INFO("Creating plan");
  // Create search plan
  void* plan;
  create_plan_dispatch(&plan,
                       dtype,
                       params.team_size,
                       params.search_mode,
                       topk,
                       params.itopk_size,
                       params.num_parents,
                       params.min_iterations,
                       params.max_iterations,
                       params.max_queries,
                       params.load_bit_length,
                       params.thread_block_size,
                       params.hashmap_mode,
                       params.hashmap_min_bitlen,
                       params.hashmap_max_fill_rate,
                       index.dataset().extent(0),
                       index.dim(),
                       index.graph_degree(),
                       (void*)index.dataset().data_handle(),
                       index.graph().data_handle());

  // Search
  IdxT* dev_seed_ptr = nullptr;
  uint32_t num_seeds = 0;

  RAFT_LOG_INFO("Cagra search");
  search_dispatch(plan,
                  neighbors.data_handle(),
                  distances.data_handle(),
                  (void*)queries.data_handle(),
                  queries.extent(0),
                  params.num_random_samplings,
                  params.rand_xor_mask,
                  dev_seed_ptr,
                  num_seeds,
                  num_executed_iterations,
                  0);

  // Destroy search plan
  destroy_plan_dispatch(plan);
}

/** @} */  // end group cagra

}  // namespace raft::neighbors::experimental::cagra::detail
