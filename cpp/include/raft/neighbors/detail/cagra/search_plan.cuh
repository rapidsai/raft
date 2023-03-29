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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/neighbors/detail/cagra/cagra.hpp>

#include "hashmap.hpp"

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
                  ") must be larger or equal to `topk` (" + std::to_string(topk) + ").");
  }
  if (params.itopk_size > 1024) {
    if (params.algo == search_algo::MULTI_CTA) {
    } else {
      error_message += std::string("- `internal_topk` (" + std::to_string(params.itopk_size) +
                                   ") must be smaller or equal to 1024");
    }
  }
  if (params.hashmap_mode != "auto" && params.hashmap_mode != "hash" &&
      params.hashmap_mode != "small-hash") {
    error_message += "An invalid hashmap mode has been given: " + params.hashmap_mode + "";
  }
  if (params.algo != search_algo::AUTO && params.algo != search_algo::SINGLE_CTA &&
      params.algo != search_algo::MULTI_CTA && params.algo != search_algo::MULTI_KERNEL) {
    error_message += "An invalid kernel mode has been given: " + params.search_mode + "";
  }
  if (params.team_size != 0 && params.team_size != 4 && params.team_size != 8 &&
      params.team_size != 16 && params.team_size != 32) {
    error_message += "`team_size` must be 0, 4, 8, 16 or 32. " + std::to_string(params.team_size) +
                     " has been given.";
  }
  if (params.load_bit_length != 0 && params.load_bit_length != 64 &&
      params.load_bit_length != 128) {
    error_message += "`load_bit_length` must be 0, 64 or 128. " +
                     std::to_string(params.load_bit_length) + " has been given.";
  }
  if (params.thread_block_size != 0 && params.thread_block_size != 64 &&
      params.thread_block_size != 128 && params.thread_block_size != 256 &&
      params.thread_block_size != 512 && params.thread_block_size != 1024) {
    error_message += "`thread_block_size` must be 0, 64, 128, 256 or 512. " +
                     std::to_string(params.load_bit_length) + " has been given.";
  }
  if (params.hashmap_min_bitlen > 20) {
    error_message += "`hashmap_min_bitlen` must be equal to or smaller than 20. " +
                     std::to_string(params.hashmap_min_bitlen) + " has been given.";
  }
  if (params.hashmap_max_fill_rate < 0.1 || params.hashmap_max_fill_rate >= 0.9) {
    error_message +=
      "`hashmap_max_fill_rate` must be equal to or greater than 0.1 and smaller than 0.9. " +
      std::to_string(params.hashmap_max_fill_rate) + " has been given.";
  }
  if (params.algo == search_algo::MULTI_CTA) {
    if (params.hashmap_mode == "small_hash") {
      error_message += "`small_hash` is not available when 'search_mode' is \"multi-cta\"";
    } else {
      params.hashmap_mode = "hash";
    }
    uint32_t mc_num_cta_per_query = max(params.num_parents, params.itopk_size / 32);
    if (mc_num_cta_per_query * 32 < topk) {
      error_message += "`mc_num_cta_per_query` (" + std::to_string(mc_num_cta_per_query) +
                       ") * 32 must be equal to or greater than `topk` (" + std::to_string(topk) +
                       ") when 'search_mode' is \"multi-cta\"";
    }
  }

  if (error_message.length() != 0) { THROW("[CAGRA Error] %s", error_message.c_str()); }
}

template <uint32_t TEAM_SIZE>
inline void calc_hashmap_params(search_params params,
                                size_t topk,
                                size_t dataset_size,
                                size_t dataset_dim,
                                size_t graph_degree,
                                size_t& hash_bitlen,
                                size_t& small_hash_bitlen,
                                size_t& small_hash_reset_interval,
                                size_t& hashmap_size)
{
  // for multipel CTA search
  uint32_t mc_num_cta_per_query = 0;
  uint32_t mc_num_parents       = 0;
  uint32_t mc_itopk_size        = 0;
  if (params.algo == search_algo::MULTI_CTA) {
    mc_itopk_size        = 32;
    mc_num_parents       = 1;
    mc_num_cta_per_query = max(params.num_parents, params.itopk_size / 32);
    RAFT_LOG_DEBUG("# mc_itopk_size: %u", mc_itopk_size);
    RAFT_LOG_DEBUG("# mc_num_parents: %u", mc_num_parents);
    RAFT_LOG_DEBUG("# mc_num_cta_per_query: %u", mc_num_cta_per_query);
  }

  // Determine hash size (bit length)
  hash_bitlen               = 0;
  small_hash_bitlen         = 0;
  small_hash_reset_interval = 1024 * 1024;
  float max_fill_rate       = params.hashmap_max_fill_rate;
  while (params.hashmap_mode == "auto" || params.hashmap_mode == "small-hash") {
    //
    // The small-hash reduces hash table size by initializing the hash table
    // for each iteraton and re-registering only the nodes that should not be
    // re-visited in that iteration. Therefore, the size of small-hash should
    // be determined based on the internal topk size and the number of nodes
    // visited per iteration.
    //
    const auto max_visited_nodes = params.itopk_size + (params.num_parents * graph_degree * 1);
    unsigned min_bitlen          = 8;   // 256
    unsigned max_bitlen          = 13;  // 8K
    if (min_bitlen < params.hashmap_min_bitlen) { min_bitlen = params.hashmap_min_bitlen; }
    hash_bitlen = min_bitlen;
    while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
      hash_bitlen += 1;
    }
    if (hash_bitlen > max_bitlen) {
      // Switch to normal hash if hashmap_mode is "auto", otherwise exit.
      if (params.hashmap_mode == "auto") {
        hash_bitlen = 0;
        break;
      } else {
        RAFT_LOG_DEBUG(
          "[CAGRA Error]"
          "small-hash cannot be used because the required hash size exceeds the limit (%u)",
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
        params.itopk_size + (params.num_parents * graph_degree * (small_hash_reset_interval + 1));
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
    uint32_t max_visited_nodes =
      params.itopk_size + (params.num_parents * graph_degree * params.max_iterations);
    if (params.algo == search_algo::MULTI_CTA) {
      max_visited_nodes = mc_itopk_size + (mc_num_parents * graph_degree * params.max_iterations);
      max_visited_nodes *= mc_num_cta_per_query;
    }
    unsigned min_bitlen = 11;  // 2K
    if (min_bitlen < params.hashmap_min_bitlen) { min_bitlen = params.hashmap_min_bitlen; }
    hash_bitlen = min_bitlen;
    while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
      hash_bitlen += 1;
    }
    RAFT_EXPECTS(hash_bitlen <= 20, "hash_bitlen cannot be largen than 20 (1M)");
  }

  RAFT_LOG_DEBUG("# topK = %lu", topk);
  RAFT_LOG_DEBUG("# internal topK = %lu", params.itopk_size);
  RAFT_LOG_DEBUG("# parent size = %lu", params.num_parents);
  RAFT_LOG_DEBUG("# min_iterations = %lu", params.min_iterations);
  RAFT_LOG_DEBUG("# max_iterations = %lu", params.max_iterations);
  RAFT_LOG_DEBUG("# max_queries = %lu", params.max_queries);
  RAFT_LOG_DEBUG("# team size = %u", TEAM_SIZE);
  RAFT_LOG_DEBUG("# hashmap mode = %s%s-%u",
                 (small_hash_bitlen > 0 ? "small-" : ""),
                 "hash",
                 hashmap::get_size(hash_bitlen));
  if (small_hash_bitlen > 0) {
    RAFT_LOG_DEBUG("# small_hash_reset_interval = %lu", small_hash_reset_interval);
  }
  hashmap_size = sizeof(std::uint32_t) * params.max_queries * hashmap::get_size(hash_bitlen);
  RAFT_LOG_DEBUG("# hashmap size: %lu", hashmap_size);
  if (hashmap_size >= 1024 * 1024 * 1024) {
    RAFT_LOG_DEBUG(" (%.2f GiB)", (double)hashmap_size / (1024 * 1024 * 1024));
  } else if (hashmap_size >= 1024 * 1024) {
    RAFT_LOG_DEBUG(" (%.2f MiB)", (double)hashmap_size / (1024 * 1024));
  } else if (hashmap_size >= 1024) {
    RAFT_LOG_DEBUG(" (%.2f KiB)", (double)hashmap_size / (1024));
  }
  RAFT_LOG_DEBUG("");
}

inline void set_max_dim_team(search_plan& plan, size_t dim)
{
  plan.max_dim = 1;
  while (plan.max_dim < dim && plan.max_dim <= 1024)
    plan.max_dim *= 2;
  // check params already ensured that team size is one of 0, 4, 8, 16, 32.
  if (plan.params.team_size == 0) {
    switch (plan.max_dim) {
      case 128: plan.params.team_size = 8; break;
      case 256: plan.params.team_size = 16; break;
      case 512: plan.params.team_size = 32; break;
      case 1024: plan.params.team_size = 32; break;
      default: RAFT_LOG_DEBUG("[CAGRA Error]\nDataset dimension is too large (%lu)\n", dim);
    }
  }
}

inline search_plan set_single_cta_params(search_plan plan) { return plan; }

inline search_plan create_plan(
  search_params params, size_t topk, size_t n_rows, size_t n_cols, size_t graph_degree)
{
  search_plan plan;
  plan.params = adjust_search_params(params, topk);
  check_params(plan.params, topk);

  size_t hashmap_size = 0;
  // todo dispatch on dim
  calc_hashmap_params<128>(plan.params,
                           topk,
                           n_rows,
                           n_cols,
                           graph_degree,
                           plan.hash_bitlen,
                           plan.small_hash_bitlen,
                           plan.small_hash_reset_interval,
                           hashmap_size);

  set_max_dim_team(plan, n_cols);

  switch (params.algo) {
    case search_algo::SINGLE_CTA:
      plan = set_single_cta_params(plan);  //*this);
      break;
    case search_algo::MULTI_CTA:     // et_multi_cta_params(*this); break;
    case search_algo::MULTI_KERNEL:  // set_multi_kernel_params(*this); break;
    default: THROW("Incorrect search_algo for ann_cagra");
  }
  return plan;
}
/** @} */  // end group cagra

}  // namespace raft::neighbors::experimental::cagra::detail
