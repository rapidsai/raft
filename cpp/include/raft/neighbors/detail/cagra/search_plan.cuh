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

#include "hashmap.hpp"
#include "search_single_cta.cuh"
#include "topk_for_cagra/topk_core.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/neighbors/cagra_types.hpp>
// #include <raft/neighbors/detail/cagra/cagra.hpp>
#include <raft/util/pow2_utils.cuh>
namespace raft::neighbors::experimental::cagra::detail {

struct search_plan_impl : search_params {
  int64_t dim;
  int64_t graph_degree;
  int64_t hash_bitlen;

  size_t small_hash_bitlen;
  size_t small_hash_reset_interval;
  int64_t max_dim;
  size_t hashmap_size;
  uint32_t dataset_size;
  uint32_t result_buffer_size;

  uint32_t smem_size;
  uint32_t block_size;
  uint32_t load_bit_lenght;

  rmm::device_uvector<uint32_t> hashmap;
  // single_cta params
  uint32_t num_itopk_candidates;

  // multi_cta params
  uint32_t num_cta_per_query;
  // uint32_t num_intermediate_results;
  rmm::device_uvector<uint32_t> intermediate_indices;
  rmm::device_uvector<float> intermediate_distances;
  size_t topk_workspace_size;
  rmm::device_uvector<uint32_t> topk_workspace;

  // multi_kernel params
  rmm::device_uvector<uint32_t> result_indices;  // results_indices_buffer
  rmm::device_uvector<float> result_distances;   // result_distances_buffer
  rmm::device_uvector<uint32_t> parent_node_list;
  rmm::device_uvector<uint32_t> topk_hint;
  rmm::device_scalar<uint32_t> terminate_flag;  // dev_terminate_flag, host_terminate_flag.;
  // params to be removed
  void* dataset_ptr;
  uint32_t* graph_ptr;

  search_plan_impl(raft::device_resources const& res,
                   search_params params,
                   int64_t dim,
                   int64_t graph_degree)
    : search_params(params),
      dim(dim),
      graph_degree(graph_degree),
      hashmap(0, res.get_stream()),
      intermediate_indices(0, res.get_stream()),
      intermediate_distances(0, res.get_stream()),
      topk_workspace(0, res.get_stream()),
      result_indices(0, res.get_stream()),
      result_distances(0, res.get_stream()),
      parent_node_list(0, res.get_stream()),
      topk_hint(0, res.get_stream()),
      terminate_flag(res.get_stream())
  {
    adjust_search_params();
    check_params();
    calc_hashmap_params(res);
    set_max_dim_team();

    switch (algo) {
      case search_algo::SINGLE_CTA: set_single_cta_params<float, uint32_t, float>(res); break;
      case search_algo::MULTI_CTA: set_multi_cta_params<float, uint32_t, float>(res); break;
      case search_algo::MULTI_KERNEL: set_multi_kernel_params<float, uint32_t, float>(res); break;
      default: THROW("Incorrect search_algo for ann_cagra %d", static_cast<int>(algo));
    }
  }

  void adjust_search_params()
  {
    if (algo == search_algo::AUTO) {
      if (itopk_size <= 512) {
        algo = search_algo::SINGLE_CTA;
        RAFT_LOG_DEBUG("Auto strategy: selecting single-cta");
      } else {
        algo = search_algo::MULTI_KERNEL;
        RAFT_LOG_DEBUG("Auto strategy: selecting multi-kernel");
      }
    }
    uint32_t _max_iterations = max_iterations;
    if (max_iterations == 0) {
      if (algo == search_algo::MULTI_CTA) {
        _max_iterations = 1 + std::min(32 * 1.1, 32 + 10.0);  // TODO(anaruse)
      } else {
        _max_iterations =
          1 + std::min((itopk_size / num_parents) * 1.1, (itopk_size / num_parents) + 10.0);
      }
    }
    if (max_iterations < min_iterations) { _max_iterations = min_iterations; }
    if (max_iterations < _max_iterations) {
      RAFT_LOG_DEBUG(
        "# max_iterations is increased from %u to %u.", max_iterations, _max_iterations);
      max_iterations = _max_iterations;
    }
    if (itopk_size % 32) {
      uint32_t itopk32 = itopk_size;
      itopk32 += 32 - (itopk_size % 32);
      RAFT_LOG_DEBUG("# internal_topk is increased from %u to %u, as it must be multiple of 32.",
                     itopk_size,
                     itopk32);
      itopk_size = itopk32;
    }

    if (algo == search_algo::SINGLE_CTA)
      search_mode = "single-cta";
    else if (algo == search_algo::MULTI_CTA)
      search_mode = "multi-cta";
    else if (algo == search_algo::MULTI_KERNEL)
      search_mode = "multi-kernel";
    RAFT_LOG_DEBUG("# search_mode = %d (%s)", static_cast<int>(algo), search_mode);
  }

  inline void set_max_dim_team()
  {
    max_dim = 128;
    while (max_dim < dim && max_dim <= 1024)
      max_dim *= 2;
    // check params already ensured that team size is one of 0, 4, 8, 16, 32.
    if (team_size == 0) {
      switch (max_dim) {
        case 128: team_size = 8; break;
        case 256: team_size = 16; break;
        case 512: team_size = 32; break;
        case 1024: team_size = 32; break;
        default: RAFT_LOG_DEBUG("[CAGRA Error]\nDataset dimension is too large (%lu)\n", dim);
      }
    }
  }

  // defines hash_bitlen, small_hash_bitlen, small_hash_reset interval, hash_size
  inline void calc_hashmap_params(raft::device_resources const& res)
  {
    // for multipel CTA search
    uint32_t mc_num_cta_per_query = 0;
    uint32_t mc_num_parents       = 0;
    uint32_t mc_itopk_size        = 0;
    if (algo == search_algo::MULTI_CTA) {
      mc_itopk_size        = 32;
      mc_num_parents       = 1;
      mc_num_cta_per_query = max(num_parents, itopk_size / 32);
      RAFT_LOG_DEBUG("# mc_itopk_size: %u", mc_itopk_size);
      RAFT_LOG_DEBUG("# mc_num_parents: %u", mc_num_parents);
      RAFT_LOG_DEBUG("# mc_num_cta_per_query: %u", mc_num_cta_per_query);
    }

    // Determine hash size (bit length)
    hashmap_size              = 0;
    hash_bitlen               = 0;
    small_hash_bitlen         = 0;
    small_hash_reset_interval = 1024 * 1024;
    float max_fill_rate       = hashmap_max_fill_rate;
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
      if (algo == search_algo::MULTI_CTA) {
        max_visited_nodes = mc_itopk_size + (mc_num_parents * graph_degree * max_iterations);
        max_visited_nodes *= mc_num_cta_per_query;
      }
      unsigned min_bitlen = 11;  // 2K
      if (min_bitlen < hashmap_min_bitlen) { min_bitlen = hashmap_min_bitlen; }
      hash_bitlen = min_bitlen;
      while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
        hash_bitlen += 1;
      }
      RAFT_EXPECTS(hash_bitlen <= 20, "hash_bitlen cannot be largen than 20 (1M)");
    }

    RAFT_LOG_DEBUG("# internal topK = %lu", itopk_size);
    RAFT_LOG_DEBUG("# parent size = %lu", num_parents);
    RAFT_LOG_DEBUG("# min_iterations = %lu", min_iterations);
    RAFT_LOG_DEBUG("# max_iterations = %lu", max_iterations);
    RAFT_LOG_DEBUG("# max_queries = %lu", max_queries);
    RAFT_LOG_DEBUG("# hashmap mode = %s%s-%u",
                   (small_hash_bitlen > 0 ? "small-" : ""),
                   "hash",
                   hashmap::get_size(hash_bitlen));
    if (small_hash_bitlen > 0) {
      RAFT_LOG_DEBUG("# small_hash_reset_interval = %lu", small_hash_reset_interval);
    }
    hashmap_size = sizeof(std::uint32_t) * max_queries * hashmap::get_size(hash_bitlen);
    RAFT_LOG_DEBUG("# hashmap size: %lu", hashmap_size);
    if (hashmap_size >= 1024 * 1024 * 1024) {
      RAFT_LOG_DEBUG(" (%.2f GiB)", (double)hashmap_size / (1024 * 1024 * 1024));
    } else if (hashmap_size >= 1024 * 1024) {
      RAFT_LOG_DEBUG(" (%.2f MiB)", (double)hashmap_size / (1024 * 1024));
    } else if (hashmap_size >= 1024) {
      RAFT_LOG_DEBUG(" (%.2f KiB)", (double)hashmap_size / (1024));
    }
  }

  void check(uint32_t topk)
  {
    RAFT_EXPECTS(topk <= itopk_size, "topk must be smaller than itopk_size = %lu", itopk_size);
    if (algo == search_algo::MULTI_CTA) {
      uint32_t mc_num_cta_per_query = max(num_parents, itopk_size / 32);
      RAFT_EXPECTS(mc_num_cta_per_query * 32 >= topk,
                   "`mc_num_cta_per_query` (%u) * 32 must be equal to or greater than "
                   "`topk` /%u) when 'search_mode' is \"multi-cta\"",
                   mc_num_cta_per_query,
                   topk);
    }
  }

  inline void check_params()
  {
    std::string error_message = "";

    if (itopk_size > 1024) {
      if (algo == search_algo::MULTI_CTA) {
      } else {
        error_message += std::string("- `internal_topk` (" + std::to_string(itopk_size) +
                                     ") must be smaller or equal to 1024");
      }
    }
    if (hashmap_mode != "auto" && hashmap_mode != "hash" && hashmap_mode != "small-hash") {
      error_message += "An invalid hashmap mode has been given: " + hashmap_mode + "";
    }
    if (algo != search_algo::SINGLE_CTA && algo != search_algo::MULTI_CTA &&
        algo != search_algo::MULTI_KERNEL) {
      error_message += "An invalid kernel mode has been given: " + search_mode + "";
    }
    if (team_size != 0 && team_size != 4 && team_size != 8 && team_size != 16 && team_size != 32) {
      error_message +=
        "`team_size` must be 0, 4, 8, 16 or 32. " + std::to_string(team_size) + " has been given.";
    }
    if (load_bit_length != 0 && load_bit_length != 64 && load_bit_length != 128) {
      error_message += "`load_bit_length` must be 0, 64 or 128. " +
                       std::to_string(load_bit_length) + " has been given.";
    }
    if (thread_block_size != 0 && thread_block_size != 64 && thread_block_size != 128 &&
        thread_block_size != 256 && thread_block_size != 512 && thread_block_size != 1024) {
      error_message += "`thread_block_size` must be 0, 64, 128, 256 or 512. " +
                       std::to_string(load_bit_length) + " has been given.";
    }
    if (hashmap_min_bitlen > 20) {
      error_message += "`hashmap_min_bitlen` must be equal to or smaller than 20. " +
                       std::to_string(hashmap_min_bitlen) + " has been given.";
    }
    if (hashmap_max_fill_rate < 0.1 || hashmap_max_fill_rate >= 0.9) {
      error_message +=
        "`hashmap_max_fill_rate` must be equal to or greater than 0.1 and smaller than 0.9. " +
        std::to_string(hashmap_max_fill_rate) + " has been given.";
    }
    if (algo == search_algo::MULTI_CTA) {
      if (hashmap_mode == "small_hash") {
        error_message += "`small_hash` is not available when 'search_mode' is \"multi-cta\"";
      } else {
        hashmap_mode = "hash";
      }
      uint32_t mc_num_cta_per_query = max(num_parents, itopk_size / 32);
      if (mc_num_cta_per_query * 32 < topk) {
        error_message += "`mc_num_cta_per_query` (" + std::to_string(mc_num_cta_per_query) +
                         ") * 32 must be equal to or greater than `topk` (" + std::to_string(topk) +
                         ") when 'search_mode' is \"multi-cta\"";
      }
    }

    if (error_message.length() != 0) { THROW("[CAGRA Error] %s", error_message.c_str()); }
  }

  template <typename DATA_T, typename INDEX_T, typename DISTANCE_T>
  inline void set_single_cta_params(raft::device_resources const& res)
  {
    num_itopk_candidates = num_parents * graph_degree;
    result_buffer_size   = itopk_size + num_itopk_candidates;

    typedef raft::Pow2<32> AlignBytes;
    unsigned result_buffer_size_32 = AlignBytes::roundUp(result_buffer_size);

    constexpr unsigned max_itopk = 512;
    RAFT_EXPECTS(itopk_size <= max_itopk, "itopk_size cannot be larger than %u", max_itopk);

    RAFT_LOG_DEBUG("# num_itopk_candidates: %u", num_itopk_candidates);
    RAFT_LOG_DEBUG("# num_itopk: %u", itopk_size);
    //
    // Determine the thread block size
    //
    constexpr unsigned min_block_size       = 64;  // 32 or 64
    constexpr unsigned min_block_size_radix = 256;
    constexpr unsigned max_block_size       = 1024;
    //
    const std::uint32_t topk_ws_size = 3;
    const std::uint32_t base_smem_size =
      sizeof(float) * max_dim + (sizeof(INDEX_T) + sizeof(DISTANCE_T)) * result_buffer_size_32 +
      sizeof(std::uint32_t) * hashmap::get_size(small_hash_bitlen) +
      sizeof(std::uint32_t) * num_parents + sizeof(std::uint32_t) * topk_ws_size +
      sizeof(std::uint32_t);
    smem_size = base_smem_size;
    if (num_itopk_candidates > 256) {
      // Tentatively calculate the required share memory size when radix
      // sort based topk is used, assuming the block size is the maximum.
      if (itopk_size <= 256) {
        smem_size += single_cta_search::topk_by_radix_sort<256, max_block_size>::smem_size *
                     sizeof(std::uint32_t);
      } else {
        smem_size += single_cta_search::topk_by_radix_sort<512, max_block_size>::smem_size *
                     sizeof(std::uint32_t);
      }
    }

    uint32_t block_size = thread_block_size;
    if (block_size == 0) {
      block_size = min_block_size;

      if (num_itopk_candidates > 256) {
        // radix-based topk is used.
        block_size = min_block_size_radix;

        // Internal topk values per thread must be equlal to or less than 4
        // when radix-sort block_topk is used.
        while ((block_size < max_block_size) && (max_itopk / block_size > 4)) {
          block_size *= 2;
        }
      }

      // Increase block size according to shared memory requirements.
      // If block size is 32, upper limit of shared memory size per
      // thread block is set to 4096. This is GPU generation dependent.
      constexpr unsigned ulimit_smem_size_cta32 = 4096;
      while (smem_size > ulimit_smem_size_cta32 / 32 * block_size) {
        block_size *= 2;
      }

      // Increase block size to improve GPU occupancy when batch size
      // is small, that is, number of queries is low.
      cudaDeviceProp deviceProp = res.get_device_properties();
      RAFT_LOG_DEBUG("# multiProcessorCount: %d", deviceProp.multiProcessorCount);
      while ((block_size < max_block_size) &&
             (graph_degree * num_parents * team_size >= block_size * 2) &&
             (max_queries <= (1024 / (block_size * 2)) * deviceProp.multiProcessorCount)) {
        block_size *= 2;
      }
    }
    RAFT_LOG_DEBUG("# thread_block_size: %u", block_size);
    RAFT_EXPECTS(block_size >= min_block_size,
                 "block_size cannot be smaller than min_block size, %u",
                 min_block_size);
    RAFT_EXPECTS(block_size <= max_block_size,
                 "block_size cannot be larger than max_block size %u",
                 max_block_size);
    thread_block_size = block_size;

    // Determine load bit length
    const uint32_t total_bit_length = dim * sizeof(DATA_T) * 8;
    if (load_bit_length == 0) {
      load_bit_length = 128;
      while (total_bit_length % load_bit_length) {
        load_bit_length /= 2;
      }
    }
    RAFT_LOG_DEBUG("# load_bit_length: %u  (%u loads per vector)",
                   load_bit_length,
                   total_bit_length / load_bit_length);
    RAFT_EXPECTS(total_bit_length % load_bit_length == 0,
                 "load_bit_length must be a divisor of dim*sizeof(data_t)*8=%u",
                 total_bit_length);
    RAFT_EXPECTS(load_bit_length >= 64, "load_bit_lenght cannot be less than 64");

    if (num_itopk_candidates <= 256) {
      RAFT_LOG_DEBUG("# bitonic-sort based topk routine is used");
    } else {
      RAFT_LOG_DEBUG("# radix-sort based topk routine is used");
      smem_size = base_smem_size;
      if (itopk_size <= 256) {
        constexpr unsigned MAX_ITOPK = 256;
        if (block_size == 256) {
          constexpr unsigned BLOCK_SIZE = 256;
          smem_size += single_cta_search::topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size *
                       sizeof(std::uint32_t);
        } else if (block_size == 512) {
          constexpr unsigned BLOCK_SIZE = 512;
          smem_size += single_cta_search::topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size *
                       sizeof(std::uint32_t);
        } else {
          constexpr unsigned BLOCK_SIZE = 1024;
          smem_size += single_cta_search::topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size *
                       sizeof(std::uint32_t);
        }
      } else {
        constexpr unsigned MAX_ITOPK = 512;
        if (block_size == 256) {
          constexpr unsigned BLOCK_SIZE = 256;
          smem_size += single_cta_search::topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size *
                       sizeof(std::uint32_t);
        } else if (block_size == 512) {
          constexpr unsigned BLOCK_SIZE = 512;
          smem_size += single_cta_search::topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size *
                       sizeof(std::uint32_t);
        } else {
          constexpr unsigned BLOCK_SIZE = 1024;
          smem_size += single_cta_search::topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size *
                       sizeof(std::uint32_t);
        }
      }
    }
    RAFT_LOG_DEBUG("# smem_size: %u", smem_size);
    hashmap_size = 0;
    if (small_hash_bitlen == 0) {
      hashmap_size = sizeof(uint32_t) * max_queries * hashmap::get_size(hash_bitlen);
      hashmap.resize(hashmap_size, res.get_stream());
    }
    RAFT_LOG_DEBUG("# hashmap_size: %lu", hashmap_size);
  }

  template <typename DATA_T, typename INDEX_T, typename DISTANCE_T>
  inline void set_multi_cta_params(raft::device_resources const& res)
  {
    itopk_size         = 32;
    num_parents        = 1;
    num_cta_per_query  = max(num_parents, itopk_size / 32);
    result_buffer_size = itopk_size + num_parents * graph_degree;
    typedef raft::Pow2<32> AlignBytes;
    unsigned result_buffer_size_32 = AlignBytes::roundUp(result_buffer_size);
    // constexpr unsigned max_result_buffer_size = 256;
    RAFT_EXPECTS(result_buffer_size_32 <= 256, "Result buffer size cannot exceed 256");

    smem_size = sizeof(float) * max_dim +
                (sizeof(INDEX_T) + sizeof(DISTANCE_T)) * result_buffer_size_32 +
                sizeof(uint32_t) * num_parents + sizeof(uint32_t);
    RAFT_LOG_DEBUG("# smem_size: %u", smem_size);

    //
    // Determine the thread block size
    //
    constexpr unsigned min_block_size = 64;
    constexpr unsigned max_block_size = 1024;
    block_size                        = thread_block_size;
    if (block_size == 0) {
      block_size = min_block_size;

      // Increase block size according to shared memory requirements.
      // If block size is 32, upper limit of shared memory size per
      // thread block is set to 4096. This is GPU generation dependent.
      constexpr unsigned ulimit_smem_size_cta32 = 4096;
      while (smem_size > ulimit_smem_size_cta32 / 32 * block_size) {
        block_size *= 2;
      }

      // Increase block size to improve GPU occupancy when total number of
      // CTAs (= num_cta_per_query * max_queries) is small.
      cudaDeviceProp deviceProp = res.get_device_properties();
      RAFT_LOG_DEBUG("# multiProcessorCount: %d", deviceProp.multiProcessorCount);
      while ((block_size < max_block_size) &&
             (graph_degree * num_parents * team_size >= block_size * 2) &&
             (num_cta_per_query * max_queries <=
              (1024 / (block_size * 2)) * deviceProp.multiProcessorCount)) {
        block_size *= 2;
      }
    }
    RAFT_LOG_DEBUG("# thread_block_size: %u", block_size);
    RAFT_EXPECTS(block_size >= min_block_size,
                 "block_size cannot be smaller than min_block size, %u",
                 min_block_size);
    RAFT_EXPECTS(block_size <= max_block_size,
                 "block_size cannot be larger than max_block size %u",
                 max_block_size);
    thread_block_size = block_size;

    //
    // Determine load bit length
    //
    const uint32_t total_bit_length = dim * sizeof(DATA_T) * 8;
    if (load_bit_length == 0) {
      load_bit_length = 128;
      while (total_bit_length % load_bit_length) {
        load_bit_length /= 2;
      }
    }
    RAFT_LOG_DEBUG("# load_bit_length: %u  (%u loads per vector)",
                   load_bit_length,
                   total_bit_length / load_bit_length);
    RAFT_EXPECTS(total_bit_length % load_bit_length == 0,
                 "load_bit_length must be a divisor of dim*sizeof(data_t)*8=%u",
                 total_bit_length);
    RAFT_EXPECTS(load_bit_length >= 64, "load_bit_lenght cannot be less than 64");

    //
    // Allocate memory for intermediate buffer and workspace.
    //
    uint32_t num_intermediate_results = num_cta_per_query * itopk_size;
    intermediate_indices.resize(num_intermediate_results, res.get_stream());
    intermediate_distances.resize(num_intermediate_results, res.get_stream());

    hashmap.resize(hashmap_size, res.get_stream());

    topk_workspace_size = _cuann_find_topk_bufferSize(
      topk, max_queries, num_intermediate_results, utils::get_cuda_data_type<DATA_T>());
    RAFT_LOG_DEBUG("# topk_workspace_size: %lu", topk_workspace_size);
    topk_workspace.resize(topk_workspace_size, res.get_stream());
  }

  template <typename DATA_T, typename INDEX_T, typename DISTANCE_T>
  inline void set_multi_kernel_params(raft::device_resources const& res)
  {
    //
    // Allocate memory for intermediate buffer and workspace.
    //
    result_buffer_size                   = itopk_size + (num_parents * graph_degree);
    size_t result_buffer_allocation_size = result_buffer_size + itopk_size;
    result_indices.resize(result_buffer_allocation_size * max_queries, res.get_stream());
    result_distances.resize(result_buffer_allocation_size * max_queries, res.get_stream());

    parent_node_list.resize(max_queries * num_parents, res.get_stream());
    topk_hint.resize(max_queries, res.get_stream());

    topk_workspace_size = _cuann_find_topk_bufferSize(
      itopk_size, max_queries, result_buffer_size, utils::get_cuda_data_type<DATA_T>());
    RAFT_LOG_DEBUG("# topk_workspace_size: %lu", topk_workspace_size);
    topk_workspace.resize(topk_workspace_size, res.get_stream());

    hashmap.resize(hashmap_size, res.get_stream());
  }
};

struct search_plan {
  search_plan(raft::device_resources const& res,
              search_params param,
              int64_t dim,
              int64_t graph_degree)
    : plan(res, param, dim, graph_degree)
  {
  }
  void check(uint32_t topk) { plan.check(topk); }

  // private:
  detail::search_plan_impl plan;
};
/** @} */  // end group cagra

}  // namespace raft::neighbors::experimental::cagra::detail
