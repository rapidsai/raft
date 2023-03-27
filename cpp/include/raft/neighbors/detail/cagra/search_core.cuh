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

#include <cassert>
#include <iostream>

#include "fragment.hpp"
#include "hashmap.hpp"
#include "search_common.hpp"
#include "search_multi_cta.cuh"
#include "search_multi_kernel.cuh"
#include "search_single_cta.cuh"
#include <raft/util/cuda_rt_essentials.hpp>

using DISTANCE_T = float;
using INDEX_T    = std::uint32_t;
namespace raft::neighbors::experimental::cagra::detail {
template <class DATA_T, unsigned MAX_DATASET_DIM, unsigned TEAM_SIZE>
void create_plan(void** plan,
                 const std::string search_mode,
                 const std::size_t topk,
                 const std::size_t itopk_size,
                 const std::size_t num_parents,
                 const std::size_t min_iterations,
                 const std::size_t max_iterations,
                 const std::size_t max_queries,
                 const std::size_t load_bit_length,
                 const std::size_t thread_block_size,
                 const std::string hashmap_mode,
                 const std::size_t hashmap_min_bitlen,
                 const float hashmap_max_fill_rate,
                 const std::size_t dataset_size,
                 const std::size_t dataset_dim,
                 const std::size_t graph_degree,
                 const void* dev_dataset_ptr,  // device ptr, [dataset_size, dataset_dim]
                 const INDEX_T* dev_graph_ptr  // device ptr, [dataset_size, graph_degree]
)
{
  // for multipel CTA search
  uint32_t mc_num_cta_per_query = 0;
  uint32_t mc_num_parents       = 0;
  uint32_t mc_itopk_size        = 0;
  if (search_mode == "multi-cta") {
    mc_itopk_size        = 32;
    mc_num_parents       = 1;
    mc_num_cta_per_query = max(num_parents, itopk_size / 32);
    printf("# mc_itopk_size: %u\n", mc_itopk_size);
    printf("# mc_num_parents: %u\n", mc_num_parents);
    printf("# mc_num_cta_per_query: %u\n", mc_num_cta_per_query);
  }

  // Determine hash size (bit length)
  std::size_t hash_bitlen               = 0;
  std::size_t small_hash_bitlen         = 0;
  std::size_t small_hash_reset_interval = 1024 * 1024;
  float max_fill_rate                   = hashmap_max_fill_rate;
  while (hashmap_mode == "auto" || hashmap_mode == "small-hash") {
    //
    // The small-hash reduces hash table size by initializing the hash table
    // for each iteraton and re-registering only the nodes that should not be
    // re-visited in that iteration. Therefore, the size of small-hash should
    // be determined based on the internal topk size and the number of nodes
    // visited per iteration.
    //
    const auto max_visited_nodes = itopk_size + (num_parents * graph_degree * 1);
    unsigned min_bitlen          = 8;   // 256
    unsigned max_bitlen          = 13;  // 8K
    if (min_bitlen < hashmap_min_bitlen) { min_bitlen = hashmap_min_bitlen; }
    hash_bitlen = min_bitlen;
    while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
      hash_bitlen += 1;
    }
    if (hash_bitlen > max_bitlen) {
      // Switch to normal hash if hashmap_mode is "auto", otherwise exit.
      if (hashmap_mode == "auto") {
        hash_bitlen = 0;
        break;
      } else {
        fprintf(stderr,
                "[CAGRA Error]\n"
                "small-hash cannot be used because the required hash size exceeds the limit (%u)\n",
                hashmap::get_size(max_bitlen));
        exit(-1);
      }
    }
    small_hash_bitlen = hash_bitlen;
    //
    // Sincc the hash table size is limited to a power of 2, the requirement,
    // the maximum fill rate, may be satisfied even if the frequency of hash
    // table reset is reduced to once every 2 or more iterations without
    // changing the hash table size. In that case, reduce the reset frequency.
    //
    small_hash_reset_interval = 1;
    while (1) {
      const auto max_visited_nodes =
        itopk_size + (num_parents * graph_degree * (small_hash_reset_interval + 1));
      if (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) { break; }
      small_hash_reset_interval += 1;
    }
    break;
  }
  if (hash_bitlen == 0) {
    //
    // The size of hash table is determined based on the maximum number of
    // nodes that may be visited before the search is completed and the
    // maximum fill rate of the hash table.
    //
    uint32_t max_visited_nodes = itopk_size + (num_parents * graph_degree * max_iterations);
    if (search_mode == "multi-cta") {
      max_visited_nodes = mc_itopk_size + (mc_num_parents * graph_degree * max_iterations);
      max_visited_nodes *= mc_num_cta_per_query;
    }
    unsigned min_bitlen = 11;  // 2K
    if (min_bitlen < hashmap_min_bitlen) { min_bitlen = hashmap_min_bitlen; }
    hash_bitlen = min_bitlen;
    while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
      hash_bitlen += 1;
    }
    // unsigned max_bitlen = 20;  // 1M
    assert(hash_bitlen <= 20);
  }

  std::printf("# topK = %lu\n", topk);
  std::printf("# internal topK = %lu\n", itopk_size);
  std::printf("# parent size = %lu\n", num_parents);
  std::printf("# min_iterations = %lu\n", min_iterations);
  std::printf("# max_iterations = %lu\n", max_iterations);
  std::printf("# max_queries = %lu\n", max_queries);
  std::printf("# team size = %u\n", TEAM_SIZE);
  std::printf("# hashmap mode = %s%s-%u\n",
              (small_hash_bitlen > 0 ? "small-" : ""),
              "hash",
              hashmap::get_size(hash_bitlen));
  if (small_hash_bitlen > 0) {
    std::printf("# small_hash_reset_interval = %lu\n", small_hash_reset_interval);
  }
  size_t hashmap_size = sizeof(std::uint32_t) * max_queries * hashmap::get_size(hash_bitlen);
  printf("# hashmap size: %lu", hashmap_size);
  if (hashmap_size >= 1024 * 1024 * 1024) {
    printf(" (%.2f GiB)", (double)hashmap_size / (1024 * 1024 * 1024));
  } else if (hashmap_size >= 1024 * 1024) {
    printf(" (%.2f MiB)", (double)hashmap_size / (1024 * 1024));
  } else if (hashmap_size >= 1024) {
    printf(" (%.2f KiB)", (double)hashmap_size / (1024));
  }
  printf("\n");
  std::fflush(stdout);

  // Create plan
  if (search_mode == "single-cta") {
    // Single CTA search
    single_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>* desc =
      new single_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>(
        search_mode,
        topk,
        itopk_size,
        num_parents,
        max_queries,
        min_iterations,
        max_iterations,
        dataset_size,
        dataset_dim,
        graph_degree,
        hash_bitlen,
        (DATA_T*)dev_dataset_ptr,
        dev_graph_ptr,
        small_hash_bitlen,
        small_hash_reset_interval,
        load_bit_length,
        thread_block_size);
    *plan = (void*)desc;
  } else if (search_mode == "multi-cta") {
    // Multiple CTA search
    multi_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>* desc =
      new multi_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>(
        search_mode,
        topk,
        mc_itopk_size,
        mc_num_parents,
        max_queries,
        min_iterations,
        max_iterations,
        dataset_size,
        dataset_dim,
        graph_degree,
        hash_bitlen,
        (DATA_T*)dev_dataset_ptr,
        dev_graph_ptr,
        mc_num_cta_per_query,
        load_bit_length,
        thread_block_size);
    *plan = (void*)desc;
  } else {
    // Multiple KERNEL search
    multi_kernel_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>* desc =
      new multi_kernel_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>(
        search_mode,
        topk,
        itopk_size,
        num_parents,
        max_queries,
        min_iterations,
        max_iterations,
        dataset_size,
        dataset_dim,
        graph_degree,
        hash_bitlen,
        (DATA_T*)dev_dataset_ptr,
        dev_graph_ptr,
        small_hash_bitlen,
        small_hash_reset_interval);
    *plan = (void*)desc;
  }
}

template <class DATA_T, unsigned MAX_DATASET_DIM, unsigned TEAM_SIZE>
void search(void* plan,
            INDEX_T* dev_topk_indices_ptr,       // [num_queries, topk]
            DISTANCE_T* dev_topk_distances_ptr,  // [num_queries, topk]
            const void* dev_query_ptr,           // [num_queries, query_dim]
            const uint32_t num_queries,
            const uint32_t num_random_samplings,
            const uint64_t rand_xor_mask,
            const INDEX_T* dev_seed_ptr,  // [num_queries, num_seeds]
            const uint32_t num_seeds,
            uint32_t* num_executed_iterations,
            cudaStream_t cuda_stream)
{
  search_common* common_plan = (search_common*)plan;
  uint32_t topk              = common_plan->_topk;
  uint32_t max_queries       = common_plan->_max_queries;
  uint32_t query_dim         = common_plan->_dataset_dim;

  for (unsigned qid = 0; qid < num_queries; qid += max_queries) {
    const uint32_t n_queries   = std::min<std::size_t>(max_queries, num_queries - qid);
    INDEX_T* _topk_indices_ptr = dev_topk_indices_ptr + (topk * qid);
    DISTANCE_T* _topk_distances_ptr =
      dev_topk_distances_ptr ? dev_topk_distances_ptr + (topk * qid) : nullptr;
    const DATA_T* _query_ptr = (const DATA_T*)dev_query_ptr + (query_dim * qid);
    const INDEX_T* _seed_ptr = dev_seed_ptr ? dev_seed_ptr + (num_seeds * qid) : nullptr;
    uint32_t* _num_executed_iterations =
      num_executed_iterations ? num_executed_iterations + qid : nullptr;

    if (common_plan->_algo == SINGLE_CTA) {
      // Single CTA search
      (*(single_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>*)plan)(
        _topk_indices_ptr,
        _topk_distances_ptr,
        _query_ptr,
        n_queries,
        num_random_samplings,
        rand_xor_mask,
        _seed_ptr,
        num_seeds,
        _num_executed_iterations,
        cuda_stream);
    } else if (common_plan->_algo == MULTI_CTA) {
      // Multiple CTA search
      (*(multi_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>*)plan)(
        _topk_indices_ptr,
        _topk_distances_ptr,
        _query_ptr,
        n_queries,
        num_random_samplings,
        rand_xor_mask,
        _seed_ptr,
        num_seeds,
        _num_executed_iterations,
        cuda_stream);
    } else {
      // Multiple kernels search
      (*(
        multi_kernel_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>*)plan)(
        _topk_indices_ptr,
        _topk_distances_ptr,
        _query_ptr,
        n_queries,
        num_random_samplings,
        rand_xor_mask,
        _seed_ptr,
        num_seeds,
        _num_executed_iterations,
        cuda_stream);
    }
  }
}

template <class DATA_T, unsigned MAX_DATASET_DIM, unsigned TEAM_SIZE>
void destroy_plan(void* plan)
{
  search_common* common_plan = (search_common*)plan;
  if (common_plan->_algo == SINGLE_CTA) {
    delete (
      single_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>*)plan;
  } else if (common_plan->_algo == MULTI_CTA) {
    delete (multi_cta_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>*)plan;
  } else {
    delete (
      multi_kernel_search::search<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>*)plan;
  }
}

}  // namespace raft::neighbors::experimental::cagra::detail