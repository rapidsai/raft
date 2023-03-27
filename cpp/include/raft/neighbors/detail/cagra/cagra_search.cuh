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
                 const search_params& params,
                 const index<T, IdxT>& index,
                 raft::device_matrix_view<const T, IdxT, row_major> queries,
                 raft::device_matrix_view<IdxT, IdxT, row_major> neighbors,
                 raft::device_matrix_view<float, IdxT, row_major> distances)
{
  const std::string dtype                  = "float";  // tamas remove
  std::string hashmap_mode                 = params.hashmap_mode;
  std::string search_mode                  = params.search_mode;
  const std::uint32_t batch_size           = params.max_queries;
  const std::uint32_t num_random_samplings = params.num_random_samplings;
  const std::uint32_t search_width         = params.num_parents;
  std::uint32_t min_iterations             = params.min_iterations;
  std::uint32_t max_iterations             = params.max_iterations;
  std::uint32_t internal_topk              = params.itopk_size;
  const std::uint32_t topk                 = neighbors.extent(1);
  std::uint32_t team_size                  = params.team_size;
  const std::uint32_t load_bit_length      = params.load_bit_length;
  const std::uint32_t thread_block_size    = params.thread_block_size;
  const std::uint32_t hashmap_min_bitlen   = params.hashmap_min_bitlen;
  const float hashmap_max_fill_rate        = params.hashmap_max_fill_rate;

  std::string error_message = "";
  if (internal_topk < topk) {
    error_message +=
      std::string("- `internal_topk` (" + std::to_string(internal_topk) +
                  ") must be larger or equal to `topk` (" + std::to_string(topk) + ").\n");
  }

  uint32_t _max_iterations = max_iterations;
  if (max_iterations == 0) {
    if (search_mode == "multi-cta") {
      _max_iterations = 1 + std::min(32 * 1.1, 32 + 10.0);  // TODO(anaruse)
    } else {
      _max_iterations =
        1 + std::min((internal_topk / search_width) * 1.1, (internal_topk / search_width) + 10.0);
    }
  }
  if (max_iterations < min_iterations) { _max_iterations = min_iterations; }
  if (max_iterations < _max_iterations) {
    RAFT_LOG_DEBUG(
      "# max_iterations is increased from %u to %u.\n", max_iterations, _max_iterations);
    max_iterations = _max_iterations;
  }

  if (internal_topk > 1024) {
    if (search_mode == "multi-cta") {
    } else {
      error_message += std::string("- `internal_topk` (" + std::to_string(internal_topk) +
                                   ") must be smaller or equal to 1024\n");
    }
  }
  if (internal_topk % 32) {
    uint32_t itopk32 = internal_topk;
    itopk32 += 32 - (internal_topk % 32);
    RAFT_LOG_DEBUG("# internal_topk is increased from %u to %u, as it must be multiple of 32.\n",
                   internal_topk,
                   itopk32);
    internal_topk = itopk32;
  }

  if (hashmap_mode != "auto" && hashmap_mode != "hash" && hashmap_mode != "small-hash") {
    error_message += "An invalid hashmap mode has been given: " + hashmap_mode + "\n";
  }

  if (search_mode != "auto" && search_mode != "single-cta" && search_mode != "multi-cta" &&
      search_mode != "multi-kernel") {
    error_message += "An invalid kernel mode has been given: " + search_mode + "\n";
  }

  if (team_size != 0 && team_size != 4 && team_size != 8 && team_size != 16 && team_size != 32) {
    error_message +=
      "`team_size` must be 0, 4, 8, 16 or 32. " + std::to_string(team_size) + " has been given.\n";
  }

  if (load_bit_length != 0 && load_bit_length != 64 && load_bit_length != 128) {
    error_message += "`load_bit_length` must be 0, 64 or 128. " + std::to_string(load_bit_length) +
                     " has been given.\n";
  }

  if (thread_block_size != 0 && thread_block_size != 64 && thread_block_size != 128 &&
      thread_block_size != 256 && thread_block_size != 512 && thread_block_size != 1024) {
    error_message += "`thread_block_size` must be 0, 64, 128, 256 or 512. " +
                     std::to_string(load_bit_length) + " has been given.\n";
  }

  if (hashmap_min_bitlen > 20) {
    error_message += "`hashmap_min_bitlen` must be equal to or smaller than 20. " +
                     std::to_string(hashmap_min_bitlen) + " has been given.\n";
  }
  if (hashmap_max_fill_rate < 0.1 || hashmap_max_fill_rate >= 0.9) {
    error_message +=
      "`hashmap_max_fill_rate` must be equal to or greater than 0.1 and smaller than 0.9. " +
      std::to_string(hashmap_max_fill_rate) + " has been given.\n";
  }

  if (search_mode == "multi-cta") {
    if (hashmap_mode == "small_hash") {
      error_message += "`small_hash` is not available when 'search_mode' is \"multi-cta\"\n";
    } else {
      hashmap_mode = "hash";
    }
    // const uint32_t mc_itopk_size  = 32;
    // const uint32_t mc_num_parents = 1;
    uint32_t mc_num_cta_per_query = max(search_width, internal_topk / 32);
    if (mc_num_cta_per_query * 32 < topk) {
      error_message += "`mc_num_cta_per_query` (" + std::to_string(mc_num_cta_per_query) +
                       ") * 32 must be equal to or greater than `topk` (" + std::to_string(topk) +
                       ") when 'search_mode' is \"multi-cta\"\n";
    }
  }

  if (error_message.length() != 0) { THROW("[CAGRA Error]\n%s", error_message.c_str()); }

  if (search_mode == "auto") {
    if (internal_topk <= 512) {
      search_mode = "single-cta";
    } else {
      search_mode = "multi-kernel";
    }
  }
  RAFT_LOG_DEBUG("# search_mode = %s\n", search_mode.c_str());

  // Load dataset and queries from file
  size_t dataset_size   = index.dataset().extent(0);
  void* dev_dataset_ptr = (void*)index.dataset().data_handle();
  void* dev_query_ptr   = (void*)queries.data_handle();

  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(index.dataset().extent(0)),
                 static_cast<size_t>(index.dataset().extent(1)));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  // assert(index.dataset_.extent(0) == graph_size);
  assert(queries.extent(1) == index.dataset().extent(1));

  // Allocate buffer for search results
  // todo(tfeher) handle different index types
  INDEX_T* dev_topk_indices_ptr      = neighbors.data_handle();  // [num_queries, topk]
  DISTANCE_T* dev_topk_distances_ptr = distances.data_handle();

  // Allocate memory for stats
  std::uint32_t* num_executed_iterations = nullptr;
  RAFT_CUDA_TRY(
    cudaMallocHost(&num_executed_iterations, sizeof(std::uint32_t) * queries.extent(0)));

  RAFT_LOG_INFO("Creating plan");
  // Create search plan
  void* plan;
  create_plan_dispatch(&plan,
                       dtype,
                       team_size,
                       search_mode,
                       topk,
                       internal_topk,
                       search_width,
                       min_iterations,
                       max_iterations,
                       batch_size,
                       load_bit_length,
                       thread_block_size,
                       hashmap_mode,
                       hashmap_min_bitlen,
                       hashmap_max_fill_rate,
                       dataset_size,
                       index.dim(),
                       index.graph_degree(),
                       dev_dataset_ptr,
                       index.graph().data_handle());

  // Search
  const uint64_t rand_xor_mask = 0x128394;
  INDEX_T* dev_seed_ptr        = nullptr;
  uint32_t num_seeds           = 0;

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  const auto start_clock = std::chrono::system_clock::now();

  RAFT_LOG_INFO("Cagra search");
  search_dispatch(plan,
                  dev_topk_indices_ptr,
                  nullptr,  // dev_topk_distances_ptr ,
                  dev_query_ptr,
                  queries.extent(0),
                  num_random_samplings,
                  rand_xor_mask,
                  dev_seed_ptr,
                  num_seeds,
                  num_executed_iterations,
                  0);

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  const auto end_clock = std::chrono::system_clock::now();
  double search_time =
    std::chrono::duration_cast<std::chrono::microseconds>(end_clock - start_clock).count() * 1e-6;

  RAFT_LOG_INFO("Cagra finished");
  // Destroy search plan
  RAFT_LOG_INFO("Destroying plan");
  destroy_plan_dispatch(plan);
  RAFT_LOG_INFO("Destroyed");

  RAFT_CUDA_TRY(cudaFreeHost(num_executed_iterations));
}

/** @} */  // end group cagra

}  // namespace raft::neighbors::experimental::cagra::detail
