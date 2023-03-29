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

inline search_params adjust_search_params(search_params params, uint32_t topk)
{
  uint32_t _max_iterations = params.max_iterations;
  if (params.max_iterations == 0) {
    if (params.algo == search_algo::MULTI_CTA) {
      _max_iterations = 1 + std::min(32 * 1.1, 32 + 10.0);  // TODO(anaruse)
    } else {
      _max_iterations = 1 + std::min((params.itopk_size / params.num_parents) * 1.1,
                                     (params.itopk_size / params.num_parents) + 10.0);
    }
  }
  if (params.max_iterations < params.min_iterations) { _max_iterations = params.min_iterations; }
  if (params.max_iterations < _max_iterations) {
    RAFT_LOG_DEBUG(
      "# max_iterations is increased from %u to %u.", params.max_iterations, _max_iterations);
    params.max_iterations = _max_iterations;
  }
  if (params.itopk_size % 32) {
    uint32_t itopk32 = params.itopk_size;
    itopk32 += 32 - (params.itopk_size % 32);
    RAFT_LOG_DEBUG("# internal_topk is increased from %u to %u, as it must be multiple of 32.",
                   params.itopk_size,
                   itopk32);
    params.itopk_size = itopk32;
  }
  if (params.algo == search_algo::AUTO) {
    if (params.itopk_size <= 512) {
      params.algo = search_algo::SINGLE_CTA;
    } else {
      params.algo = search_algo::MULTI_KERNEL;
    }
  }
  if (params.algo == search_algo::SINGLE_CTA)
    params.search_mode = "single-cta";
  else if (params.algo == search_algo::MULTI_CTA)
    params.search_mode = "multi-cta";
  else if (params.algo == search_algo::MULTI_KERNEL)
    params.search_mode = "multi-kernel";
  RAFT_LOG_DEBUG("# search_mode = %d", static_cast<int>(params.algo));
  return params;
}

inline void check_params(search_params params, uint32_t topk)
{
  std::string error_message = "";
  if (params.itopk_size < topk) {
    error_message +=
      std::string("- `internal_topk` (" + std::to_string(params.itopk_size) +
                  ") must be larger or equal to `topk` (" + std::to_string(topk) + ").\n");
  }
  if (params.itopk_size > 1024) {
    if (params.algo == search_algo::MULTI_CTA) {
    } else {
      error_message += std::string("- `internal_topk` (" + std::to_string(params.itopk_size) +
                                   ") must be smaller or equal to 1024\n");
    }
  }
  if (params.hashmap_mode != "auto" && params.hashmap_mode != "hash" &&
      params.hashmap_mode != "small-hash") {
    error_message += "An invalid hashmap mode has been given: " + params.hashmap_mode + "\n";
  }
  if (params.algo != search_algo::AUTO && params.algo != search_algo::SINGLE_CTA &&
      params.algo != search_algo::MULTI_CTA && params.algo != search_algo::MULTI_KERNEL) {
    error_message += "An invalid kernel mode has been given: " + params.search_mode + "\n";
  }
  if (params.team_size != 0 && params.team_size != 4 && params.team_size != 8 &&
      params.team_size != 16 && params.team_size != 32) {
    error_message += "`team_size` must be 0, 4, 8, 16 or 32. " + std::to_string(params.team_size) +
                     " has been given.\n";
  }
  if (params.load_bit_length != 0 && params.load_bit_length != 64 &&
      params.load_bit_length != 128) {
    error_message += "`load_bit_length` must be 0, 64 or 128. " +
                     std::to_string(params.load_bit_length) + " has been given.\n";
  }
  if (params.thread_block_size != 0 && params.thread_block_size != 64 &&
      params.thread_block_size != 128 && params.thread_block_size != 256 &&
      params.thread_block_size != 512 && params.thread_block_size != 1024) {
    error_message += "`thread_block_size` must be 0, 64, 128, 256 or 512. " +
                     std::to_string(params.load_bit_length) + " has been given.\n";
  }
  if (params.hashmap_min_bitlen > 20) {
    error_message += "`hashmap_min_bitlen` must be equal to or smaller than 20. " +
                     std::to_string(params.hashmap_min_bitlen) + " has been given.\n";
  }
  if (params.hashmap_max_fill_rate < 0.1 || params.hashmap_max_fill_rate >= 0.9) {
    error_message +=
      "`hashmap_max_fill_rate` must be equal to or greater than 0.1 and smaller than 0.9. " +
      std::to_string(params.hashmap_max_fill_rate) + " has been given.\n";
  }
  if (params.algo == search_algo::MULTI_CTA) {
    if (params.hashmap_mode == "small_hash") {
      error_message += "`small_hash` is not available when 'search_mode' is \"multi-cta\"\n";
    } else {
      params.hashmap_mode = "hash";
    }
    uint32_t mc_num_cta_per_query = max(params.num_parents, params.itopk_size / 32);
    if (mc_num_cta_per_query * 32 < topk) {
      error_message += "`mc_num_cta_per_query` (" + std::to_string(mc_num_cta_per_query) +
                       ") * 32 must be equal to or greater than `topk` (" + std::to_string(topk) +
                       ") when 'search_mode' is \"multi-cta\"\n";
    }
  }

  if (error_message.length() != 0) { THROW("[CAGRA Error]\n%s", error_message.c_str()); }
}

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
  const std::string dtype  = "float";  // tamas remove
  const std::uint32_t topk = neighbors.extent(1);
  params                   = adjust_search_params(params, topk);
  check_params(params, topk);

  RAFT_LOG_DEBUG("# dataset size = %lu, dim = %lu\n",
                 static_cast<size_t>(index.dataset().extent(0)),
                 static_cast<size_t>(index.dataset().extent(1)));
  RAFT_LOG_DEBUG("# query size = %lu, dim = %lu\n",
                 static_cast<size_t>(queries.extent(0)),
                 static_cast<size_t>(queries.extent(1)));
  assert(queries.extent(1) == index.dataset().extent(1));

  // Allocate buffer for search results

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
                  nullptr,  // distances.data_handle(),
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
