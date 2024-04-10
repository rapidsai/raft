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

#pragma once

#include "hashmap.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/neighbors/sample_filter_types.hpp>
// #include "search_single_cta.cuh"
// #include "topk_for_cagra/topk_core.cuh"

#include <raft/core/device_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/neighbors/cagra_types.hpp>
#include <raft/util/pow2_utils.cuh>

namespace raft::neighbors::cagra::detail {

struct search_plan_impl_base : public search_params {
  int64_t dataset_block_dim;
  int64_t dim;
  int64_t graph_degree;
  uint32_t topk;
  raft::distance::DistanceType metric;
  search_plan_impl_base(search_params params,
                        int64_t dim,
                        int64_t graph_degree,
                        uint32_t topk,
                        raft::distance::DistanceType metric)
    : search_params(params), dim(dim), graph_degree(graph_degree), topk(topk), metric(metric)
  {
    set_dataset_block_and_team_size(dim);
    if (algo == search_algo::AUTO) {
      const size_t num_sm = raft::getMultiProcessorCount();
      if (itopk_size <= 512 && search_params::max_queries >= num_sm * 2lu) {
        algo = search_algo::SINGLE_CTA;
        RAFT_LOG_DEBUG("Auto strategy: selecting single-cta");
      } else if (topk <= 1024) {
        algo = search_algo::MULTI_CTA;
        RAFT_LOG_DEBUG("Auto strategy: selecting multi-cta");
      } else {
        algo = search_algo::MULTI_KERNEL;
        RAFT_LOG_DEBUG("Auto strategy: selecting multi kernel");
      }
    }
  }

  void set_dataset_block_and_team_size(int64_t dim)
  {
    constexpr int64_t max_dataset_block_dim = 512;
    dataset_block_dim                       = 128;
    while (dataset_block_dim < dim && dataset_block_dim < max_dataset_block_dim) {
      dataset_block_dim *= 2;
    }
    // To keep binary size in check we limit only one team size specialization for each max_dim.
    // TODO(tfeher): revise this decision.
    switch (dataset_block_dim) {
      case 128: team_size = 8; break;
      case 256: team_size = 16; break;
      default: team_size = 32; break;
    }
  }
};

template <class DATASET_DESCRIPTOR_T, class SAMPLE_FILTER_T>
struct search_plan_impl : public search_plan_impl_base {
  using INDEX_T    = typename DATASET_DESCRIPTOR_T::INDEX_T;
  using DISTANCE_T = typename DATASET_DESCRIPTOR_T::DISTANCE_T;
  using DATA_T     = typename DATASET_DESCRIPTOR_T::DATA_T;

  int64_t hash_bitlen;

  size_t small_hash_bitlen;
  size_t small_hash_reset_interval;
  size_t hashmap_size;
  uint32_t dataset_size;
  uint32_t result_buffer_size;

  uint32_t smem_size;
  uint32_t topk;
  uint32_t num_seeds;

  rmm::device_uvector<INDEX_T> hashmap;
  rmm::device_uvector<uint32_t> num_executed_iterations;  // device or managed?
  rmm::device_uvector<INDEX_T> dev_seed;

  search_plan_impl(raft::resources const& res,
                   search_params params,
                   int64_t dim,
                   int64_t graph_degree,
                   uint32_t topk,
                   raft::distance::DistanceType metric)
    : search_plan_impl_base(params, dim, graph_degree, topk, metric),
      hashmap(0, resource::get_cuda_stream(res)),
      num_executed_iterations(0, resource::get_cuda_stream(res)),
      dev_seed(0, resource::get_cuda_stream(res)),
      num_seeds(0)
  {
    adjust_search_params();
    check_params();
    calc_hashmap_params(res);
    set_dataset_block_and_team_size(dim);
    num_executed_iterations.resize(max_queries, resource::get_cuda_stream(res));
    RAFT_LOG_DEBUG("# algo = %d", static_cast<int>(algo));
  }

  virtual ~search_plan_impl() {}

  virtual void operator()(raft::resources const& res,
                          DATASET_DESCRIPTOR_T dataset_desc,
                          raft::device_matrix_view<const INDEX_T, int64_t, row_major> graph,
                          INDEX_T* const result_indices_ptr,       // [num_queries, topk]
                          DISTANCE_T* const result_distances_ptr,  // [num_queries, topk]
                          const DATA_T* const queries_ptr,         // [num_queries, dataset_dim]
                          const std::uint32_t num_queries,
                          const INDEX_T* dev_seed_ptr,                   // [num_queries, num_seeds]
                          std::uint32_t* const num_executed_iterations,  // [num_queries]
                          uint32_t topk,
                          SAMPLE_FILTER_T sample_filter){};

  void adjust_search_params()
  {
    uint32_t _max_iterations = max_iterations;
    if (max_iterations == 0) {
      if (algo == search_algo::MULTI_CTA) {
        _max_iterations = 1 + std::min(32 * 1.1, 32 + 10.0);  // TODO(anaruse)
      } else {
        _max_iterations =
          1 + std::min((itopk_size / search_width) * 1.1, (itopk_size / search_width) + 10.0);
      }
    }
    if (max_iterations < min_iterations) { _max_iterations = min_iterations; }
    if (max_iterations < _max_iterations) {
      RAFT_LOG_DEBUG(
        "# max_iterations is increased from %lu to %u.", max_iterations, _max_iterations);
      max_iterations = _max_iterations;
    }
    if (itopk_size % 32) {
      uint32_t itopk32 = itopk_size;
      itopk32 += 32 - (itopk_size % 32);
      RAFT_LOG_DEBUG("# internal_topk is increased from %lu to %u, as it must be multiple of 32.",
                     itopk_size,
                     itopk32);
      itopk_size = itopk32;
    }
  }

  // defines hash_bitlen, small_hash_bitlen, small_hash_reset interval, hash_size
  inline void calc_hashmap_params(raft::resources const& res)
  {
    // for multiple CTA search
    uint32_t mc_num_cta_per_query = 0;
    uint32_t mc_search_width      = 0;
    uint32_t mc_itopk_size        = 0;
    if (algo == search_algo::MULTI_CTA) {
      mc_itopk_size        = 32;
      mc_search_width      = 1;
      mc_num_cta_per_query = max(search_width, raft::ceildiv(itopk_size, (size_t)32));
      RAFT_LOG_DEBUG("# mc_itopk_size: %u", mc_itopk_size);
      RAFT_LOG_DEBUG("# mc_search_width: %u", mc_search_width);
      RAFT_LOG_DEBUG("# mc_num_cta_per_query: %u", mc_num_cta_per_query);
    }

    // Determine hash size (bit length)
    hashmap_size              = 0;
    hash_bitlen               = 0;
    small_hash_bitlen         = 0;
    small_hash_reset_interval = 1024 * 1024;
    float max_fill_rate       = hashmap_max_fill_rate;
    while (hashmap_mode == hash_mode::AUTO || hashmap_mode == hash_mode::SMALL) {
      //
      // The small-hash reduces hash table size by initializing the hash table
      // for each iteraton and re-registering only the nodes that should not be
      // re-visited in that iteration. Therefore, the size of small-hash should
      // be determined based on the internal topk size and the number of nodes
      // visited per iteration.
      //
      const auto max_visited_nodes = itopk_size + (search_width * graph_degree * 1);
      unsigned min_bitlen          = 8;   // 256
      unsigned max_bitlen          = 13;  // 8K
      if (min_bitlen < hashmap_min_bitlen) { min_bitlen = hashmap_min_bitlen; }
      hash_bitlen = min_bitlen;
      while (max_visited_nodes > hashmap::get_size(hash_bitlen) * max_fill_rate) {
        hash_bitlen += 1;
      }
      if (hash_bitlen > max_bitlen) {
        // Switch to normal hash if hashmap_mode is AUTO, otherwise exit.
        if (hashmap_mode == hash_mode::AUTO) {
          hash_bitlen = 0;
          break;
        } else {
          RAFT_FAIL(
            "small-hash cannot be used because the required hash size exceeds the limit (%u)",
            hashmap::get_size(max_bitlen));
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
          itopk_size + (search_width * graph_degree * (small_hash_reset_interval + 1));
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
      uint32_t max_visited_nodes = itopk_size + (search_width * graph_degree * max_iterations);
      if (algo == search_algo::MULTI_CTA) {
        max_visited_nodes = mc_itopk_size + (mc_search_width * graph_degree * max_iterations);
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
    RAFT_LOG_DEBUG("# parent size = %lu", search_width);
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
    hashmap_size = sizeof(INDEX_T) * max_queries * hashmap::get_size(hash_bitlen);
    RAFT_LOG_DEBUG("# hashmap size: %lu", hashmap_size);
    if (hashmap_size >= 1024 * 1024 * 1024) {
      RAFT_LOG_DEBUG(" (%.2f GiB)", (double)hashmap_size / (1024 * 1024 * 1024));
    } else if (hashmap_size >= 1024 * 1024) {
      RAFT_LOG_DEBUG(" (%.2f MiB)", (double)hashmap_size / (1024 * 1024));
    } else if (hashmap_size >= 1024) {
      RAFT_LOG_DEBUG(" (%.2f KiB)", (double)hashmap_size / (1024));
    }
  }

  virtual void check(const uint32_t topk)
  {
    // For single-CTA and multi kernel
    RAFT_EXPECTS(
      topk <= itopk_size, "topk = %u must be smaller than itopk_size = %lu", topk, itopk_size);
  }

  inline void check_params()
  {
    std::string error_message = "";

    if (itopk_size > 1024) {
      if ((algo == search_algo::MULTI_CTA) || (algo == search_algo::MULTI_KERNEL)) {
      } else {
        error_message += std::string("- `internal_topk` (" + std::to_string(itopk_size) +
                                     ") must be smaller or equal to 1024");
      }
    }
    if (algo != search_algo::SINGLE_CTA && algo != search_algo::MULTI_CTA &&
        algo != search_algo::MULTI_KERNEL) {
      error_message += "An invalid kernel mode has been given: " + std::to_string((int)algo) + "";
    }
    if (team_size != 0 && team_size != 4 && team_size != 8 && team_size != 16 && team_size != 32) {
      error_message +=
        "`team_size` must be 0, 4, 8, 16 or 32. " + std::to_string(team_size) + " has been given.";
    }
    if (thread_block_size != 0 && thread_block_size != 64 && thread_block_size != 128 &&
        thread_block_size != 256 && thread_block_size != 512 && thread_block_size != 1024) {
      error_message += "`thread_block_size` must be 0, 64, 128, 256 or 512. " +
                       std::to_string(thread_block_size) + " has been given.";
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
    if constexpr (!std::is_same<SAMPLE_FILTER_T,
                                raft::neighbors::filtering::none_cagra_sample_filter>::value) {
      if (hashmap_mode == hash_mode::SMALL) {
        error_message += "`SMALL` hash is not available when filtering";
      } else {
        hashmap_mode = hash_mode::HASH;
      }
    }
    if (algo == search_algo::MULTI_CTA) {
      if (hashmap_mode == hash_mode::SMALL) {
        error_message += "`small_hash` is not available when 'search_mode' is \"multi-cta\"";
      } else {
        hashmap_mode = hash_mode::HASH;
      }
    }

    if (error_message.length() != 0) { THROW("[CAGRA Error] %s", error_message.c_str()); }
  }
};

// template <class DATA_T, class DISTANCE_T, class INDEX_T>
// struct search_plan {
//   search_plan(raft::resources const& res,
//               search_params param,
//               int64_t dim,
//               int64_t graph_degree)
//     : plan(res, param, dim, graph_degree)
//   {
//   }
//   void check(uint32_t topk) { plan.check(topk); }

//   // private:
//   detail::search_plan_impl<DATA_T, DISTANCE_T, INDEX_T> plan;
// };
/** @} */  // end group cagra

}  // namespace raft::neighbors::cagra::detail
