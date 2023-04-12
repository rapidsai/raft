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
#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

#include "bitonic.hpp"
#include "compute_distance.hpp"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_for_cagra/topk_core.cuh"  // TODO replace with raft topk
#include "utils.hpp"
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

namespace raft::neighbors::experimental::cagra::detail {
namespace single_cta_search {

// #define _CLK_BREAKDOWN

template <unsigned TOPK_BY_BITONIC_SORT, class INDEX_T>
__device__ void pickup_next_parents(std::uint32_t* const terminate_flag,
                                    INDEX_T* const next_parent_indices,
                                    INDEX_T* const internal_topk_indices,
                                    const std::size_t internal_topk_size,
                                    const std::size_t dataset_size,
                                    const std::uint32_t num_parents)
{
  // if (threadIdx.x >= 32) return;

  for (std::uint32_t i = threadIdx.x; i < num_parents; i += 32) {
    next_parent_indices[i] = utils::get_max_value<INDEX_T>();
  }
  std::uint32_t itopk_max = internal_topk_size;
  if (itopk_max % 32) { itopk_max += 32 - (itopk_max % 32); }
  std::uint32_t num_new_parents = 0;
  for (std::uint32_t j = threadIdx.x; j < itopk_max; j += 32) {
    std::uint32_t jj = j;
    if (TOPK_BY_BITONIC_SORT) { jj = device::swizzling(j); }
    INDEX_T index;
    int new_parent = 0;
    if (j < internal_topk_size) {
      index = internal_topk_indices[jj];
      if ((index & 0x80000000) == 0) {  // check if most significant bit is set
        new_parent = 1;
      }
    }
    const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
      if (i < num_parents) {
        next_parent_indices[i] = index;
        // set most significant bit as used node
        internal_topk_indices[jj] |= 0x80000000;
      }
    }
    num_new_parents += __popc(ballot_mask);
    if (num_new_parents >= num_parents) { break; }
  }
  if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}

template <unsigned MAX_INTERNAL_TOPK>
struct topk_by_radix_sort_base {
  static constexpr std::uint32_t smem_size        = MAX_INTERNAL_TOPK * 2 + 2048 + 8;
  static constexpr std::uint32_t state_bit_lenght = 0;
  static constexpr std::uint32_t vecLen           = 2;  // TODO
};
template <unsigned MAX_INTERNAL_TOPK, unsigned BLOCK_SIZE, class = void>
struct topk_by_radix_sort : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {
};

template <unsigned MAX_INTERNAL_TOPK, unsigned BLOCK_SIZE>
struct topk_by_radix_sort<MAX_INTERNAL_TOPK,
                          BLOCK_SIZE,
                          std::enable_if_t<((MAX_INTERNAL_TOPK <= 64))>>
  : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {
  __device__ void operator()(uint32_t topk,
                             uint32_t batch_size,
                             uint32_t len_x,
                             const uint32_t* _x,
                             const uint32_t* _in_vals,
                             uint32_t* _y,
                             uint32_t* _out_vals,
                             uint32_t* work,
                             uint32_t* _hints,
                             bool sort,
                             uint32_t* _smem)
  {
    std::uint8_t* state = (std::uint8_t*)work;
    topk_cta_11_core<BLOCK_SIZE,
                     topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_lenght,
                     topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen,
                     64,
                     32>(topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);
  }
};

#define TOP_FUNC_PARTIAL_SPECIALIZATION(V)                                           \
  template <unsigned MAX_INTERNAL_TOPK, unsigned BLOCK_SIZE>                         \
  struct topk_by_radix_sort<                                                         \
    MAX_INTERNAL_TOPK,                                                               \
    BLOCK_SIZE,                                                                      \
    std::enable_if_t<((MAX_INTERNAL_TOPK <= V) && (2 * MAX_INTERNAL_TOPK > V))>>     \
    : topk_by_radix_sort_base<MAX_INTERNAL_TOPK> {                                   \
    __device__ void operator()(uint32_t topk,                                        \
                               uint32_t batch_size,                                  \
                               uint32_t len_x,                                       \
                               const uint32_t* _x,                                   \
                               const uint32_t* _in_vals,                             \
                               uint32_t* _y,                                         \
                               uint32_t* _out_vals,                                  \
                               uint32_t* work,                                       \
                               uint32_t* _hints,                                     \
                               bool sort,                                            \
                               uint32_t* _smem)                                      \
    {                                                                                \
      assert(BLOCK_SIZE >= V / 4);                                                   \
      std::uint8_t* state = (std::uint8_t*)work;                                     \
      topk_cta_11_core<BLOCK_SIZE,                                                   \
                       topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::state_bit_lenght, \
                       topk_by_radix_sort_base<MAX_INTERNAL_TOPK>::vecLen,           \
                       V,                                                            \
                       V / 4>(                                                       \
        topk, len_x, _x, _in_vals, _y, _out_vals, state, _hints, sort, _smem);       \
    }                                                                                \
  };
TOP_FUNC_PARTIAL_SPECIALIZATION(128);
TOP_FUNC_PARTIAL_SPECIALIZATION(256);
TOP_FUNC_PARTIAL_SPECIALIZATION(512);
TOP_FUNC_PARTIAL_SPECIALIZATION(1024);

template <unsigned MAX_CANDIDATES, unsigned MULTI_WARPS = 0>
__device__ inline void topk_by_bitonic_sort_1st(
  float* candidate_distances,        // [num_candidates]
  std::uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  const std::uint32_t num_itopk)
{
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;
  if (MULTI_WARPS == 0) {
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_CANDIDATES + 31) / 32;
    float key[N];
    std::uint32_t val[N];
    /* Candidates -> Reg */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = lane_id + (32 * i);
      if (j < num_candidates) {
        key[i] = candidate_distances[j];
        val[i] = candidate_indices[j];
      } else {
        key[i] = utils::get_max_value<float>();
        val[i] = utils::get_max_value<std::uint32_t>();
      }
    }
    /* Sort */
    bitonic::warp_sort<float, std::uint32_t, N>(key, val);
    /* Reg -> Temp_itopk */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_candidates && j < num_itopk) {
        candidate_distances[device::swizzling(j)] = key[i];
        candidate_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    // Use two warps (64 threads)
    constexpr unsigned max_candidates_per_warp = (MAX_CANDIDATES + 1) / 2;
    constexpr unsigned N                       = (max_candidates_per_warp + 31) / 32;
    float key[N];
    std::uint32_t val[N];
    if (warp_id < 2) {
      /* Candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = lane_id + (32 * i);
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates) {
          key[i] = candidate_distances[j];
          val[i] = candidate_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<std::uint32_t>();
        }
      }
      /* Sort */
      bitonic::warp_sort<float, std::uint32_t, N>(key, val);
      /* Reg -> Temp_candidates */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && jl < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    __syncthreads();

    unsigned num_warps_used = (num_itopk + max_candidates_per_warp - 1) / max_candidates_per_warp;
    if (warp_id < num_warps_used) {
      /* Temp_candidates -> Reg */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned kl = max_candidates_per_warp - 1 - jl;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        unsigned k  = MAX_CANDIDATES - 1 - j;
        if (j >= num_candidates || k >= num_candidates || kl >= num_itopk) continue;
        float temp_key = candidate_distances[device::swizzling(k)];
        if (key[i] == temp_key) continue;
        if ((warp_id == 0) == (key[i] > temp_key)) {
          key[i] = temp_key;
          val[i] = candidate_indices[device::swizzling(k)];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
    if (warp_id < num_warps_used) {
      /* Merge */
      bitonic::warp_merge<float, std::uint32_t, N>(key, val, 32);
      /* Reg -> Temp_itopk */
      for (unsigned i = 0; i < N; i++) {
        unsigned jl = (N * lane_id) + i;
        unsigned j  = jl + (max_candidates_per_warp * warp_id);
        if (j < num_candidates && j < num_itopk) {
          candidate_distances[device::swizzling(j)] = key[i];
          candidate_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    if (num_warps_used > 1) { __syncthreads(); }
  }
}

template <unsigned MAX_ITOPK, unsigned MULTI_WARPS = 0>
__device__ inline void topk_by_bitonic_sort_2nd(
  float* itopk_distances,        // [num_itopk]
  std::uint32_t* itopk_indices,  // [num_itopk]
  const std::uint32_t num_itopk,
  float* candidate_distances,        // [num_candidates]
  std::uint32_t* candidate_indices,  // [num_candidates]
  const std::uint32_t num_candidates,
  std::uint32_t* work_buf,
  const bool first)
{
  const unsigned lane_id = threadIdx.x % 32;
  const unsigned warp_id = threadIdx.x / 32;
  if (MULTI_WARPS == 0) {
    if (warp_id > 0) { return; }
    constexpr unsigned N = (MAX_ITOPK + 31) / 32;
    float key[N];
    std::uint32_t val[N];
    if (first) {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = lane_id + (32 * i);
        if (j < num_itopk) {
          key[i] = itopk_distances[j];
          val[i] = itopk_indices[j];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<std::uint32_t>();
        }
      }
      /* Warp Sort */
      bitonic::warp_sort<float, std::uint32_t, N>(key, val);
    } else {
      /* Load itopk results */
      for (unsigned i = 0; i < N; i++) {
        unsigned j = (N * lane_id) + i;
        if (j < num_itopk) {
          key[i] = itopk_distances[device::swizzling(j)];
          val[i] = itopk_indices[device::swizzling(j)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<std::uint32_t>();
        }
      }
    }
    /* Merge candidates */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;  // [0:MAX_ITOPK-1]
      unsigned k = MAX_ITOPK - 1 - j;
      if (k >= num_itopk || k >= num_candidates) continue;
      float candidate_key = candidate_distances[device::swizzling(k)];
      if (key[i] > candidate_key) {
        key[i] = candidate_key;
        val[i] = candidate_indices[device::swizzling(k)];
      }
    }
    /* Warp Merge */
    bitonic::warp_merge<float, std::uint32_t, N>(key, val, 32);
    /* Store new itopk results */
    for (unsigned i = 0; i < N; i++) {
      unsigned j = (N * lane_id) + i;
      if (j < num_itopk) {
        itopk_distances[device::swizzling(j)] = key[i];
        itopk_indices[device::swizzling(j)]   = val[i];
      }
    }
  } else {
    // Use two warps (64 threads) or more
    constexpr unsigned max_itopk_per_warp = (MAX_ITOPK + 1) / 2;
    constexpr unsigned N                  = (max_itopk_per_warp + 31) / 32;
    float key[N];
    std::uint32_t val[N];
    if (first) {
      /* Load itop results (not sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = lane_id + (32 * i) + (max_itopk_per_warp * warp_id);
          if (j < num_itopk) {
            key[i] = itopk_distances[j];
            val[i] = itopk_indices[j];
          } else {
            key[i] = utils::get_max_value<float>();
            val[i] = utils::get_max_value<std::uint32_t>();
          }
        }
        /* Warp Sort */
        bitonic::warp_sort<float, std::uint32_t, N>(key, val);
        /* Store intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
      __syncthreads();
      if (warp_id < 2) {
        /* Load intermedidate results */
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          unsigned k = MAX_ITOPK - 1 - j;
          if (k >= num_itopk) continue;
          float temp_key = itopk_distances[device::swizzling(k)];
          if (key[i] == temp_key) continue;
          if ((warp_id == 0) == (key[i] > temp_key)) {
            key[i] = temp_key;
            val[i] = itopk_indices[device::swizzling(k)];
          }
        }
        /* Warp Merge */
        bitonic::warp_merge<float, std::uint32_t, N>(key, val, 32);
      }
      __syncthreads();
      /* Store itopk results (sorted) */
      if (warp_id < 2) {
        for (unsigned i = 0; i < N; i++) {
          unsigned j = (N * threadIdx.x) + i;
          if (j >= num_itopk) continue;
          itopk_distances[device::swizzling(j)] = key[i];
          itopk_indices[device::swizzling(j)]   = val[i];
        }
      }
    }
    const uint32_t num_itopk_div2 = num_itopk / 2;
    if (threadIdx.x < 3) {
      // work_buf is used to obtain turning points in 1st and 2nd half of itopk afer merge.
      work_buf[threadIdx.x] = num_itopk_div2;
    }
    __syncthreads();

    // Merge candidates (using whole threads)
    for (unsigned k = threadIdx.x; k < min(num_candidates, num_itopk); k += blockDim.x) {
      const unsigned j          = num_itopk - 1 - k;
      const float itopk_key     = itopk_distances[device::swizzling(j)];
      const float candidate_key = candidate_distances[device::swizzling(k)];
      if (itopk_key > candidate_key) {
        itopk_distances[device::swizzling(j)] = candidate_key;
        itopk_indices[device::swizzling(j)]   = candidate_indices[device::swizzling(k)];
        if (j < num_itopk_div2) {
          atomicMin(work_buf + 2, j);
        } else {
          atomicMin(work_buf + 1, j - num_itopk_div2);
        }
      }
    }
    __syncthreads();

    // Merge 1st and 2nd half of itopk (using whole threads)
    for (unsigned j = threadIdx.x; j < num_itopk_div2; j += blockDim.x) {
      const unsigned k = j + num_itopk_div2;
      float key_0      = itopk_distances[device::swizzling(j)];
      float key_1      = itopk_distances[device::swizzling(k)];
      if (key_0 > key_1) {
        itopk_distances[device::swizzling(j)] = key_1;
        itopk_distances[device::swizzling(k)] = key_0;
        std::uint32_t val_0                   = itopk_indices[device::swizzling(j)];
        std::uint32_t val_1                   = itopk_indices[device::swizzling(k)];
        itopk_indices[device::swizzling(j)]   = val_1;
        itopk_indices[device::swizzling(k)]   = val_0;
        atomicMin(work_buf + 0, j);
      }
    }
    if (threadIdx.x == blockDim.x - 1) {
      if (work_buf[2] < num_itopk_div2) { work_buf[1] = work_buf[2]; }
    }
    __syncthreads();
    // if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
    //     RAFT_LOG_DEBUG( "work_buf: %u, %u, %u\n", work_buf[0], work_buf[1], work_buf[2] );
    // }

    // Warp-0 merges 1st half of itopk, warp-1 does 2nd half.
    if (warp_id < 2) {
      // Load intermedidate itopk results
      const uint32_t turning_point = work_buf[warp_id];  // turning_point <= num_itopk_div2
      for (unsigned i = 0; i < N; i++) {
        unsigned k = num_itopk;
        unsigned j = (N * lane_id) + i;
        if (j < turning_point) {
          k = j + (num_itopk_div2 * warp_id);
        } else if (j >= (MAX_ITOPK / 2 - num_itopk_div2)) {
          j -= (MAX_ITOPK / 2 - num_itopk_div2);
          if ((turning_point <= j) && (j < num_itopk_div2)) { k = j + (num_itopk_div2 * warp_id); }
        }
        if (k < num_itopk) {
          key[i] = itopk_distances[device::swizzling(k)];
          val[i] = itopk_indices[device::swizzling(k)];
        } else {
          key[i] = utils::get_max_value<float>();
          val[i] = utils::get_max_value<std::uint32_t>();
        }
      }
      /* Warp Merge */
      bitonic::warp_merge<float, std::uint32_t, N>(key, val, 32);
      /* Store new itopk results */
      for (unsigned i = 0; i < N; i++) {
        const unsigned j = (N * lane_id) + i;
        if (j < num_itopk_div2) {
          unsigned k                            = j + (num_itopk_div2 * warp_id);
          itopk_distances[device::swizzling(k)] = key[i];
          itopk_indices[device::swizzling(k)]   = val[i];
        }
      }
    }
  }
}

template <unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned MULTI_WARPS_1,
          unsigned MULTI_WARPS_2>
__device__ void topk_by_bitonic_sort(float* itopk_distances,        // [num_itopk]
                                     std::uint32_t* itopk_indices,  // [num_itopk]
                                     const std::uint32_t num_itopk,
                                     float* candidate_distances,        // [num_candidates]
                                     std::uint32_t* candidate_indices,  // [num_candidates]
                                     const std::uint32_t num_candidates,
                                     std::uint32_t* work_buf,
                                     const bool first)
{
  // The results in candidate_distances/indices are sorted by bitonic sort.
  topk_by_bitonic_sort_1st<MAX_CANDIDATES, MULTI_WARPS_1>(
    candidate_distances, candidate_indices, num_candidates, num_itopk);

  // The results sorted above are merged with the internal intermediate top-k
  // results so far using bitonic merge.
  topk_by_bitonic_sort_2nd<MAX_ITOPK, MULTI_WARPS_2>(itopk_distances,
                                                     itopk_indices,
                                                     num_itopk,
                                                     candidate_distances,
                                                     candidate_indices,
                                                     num_candidates,
                                                     work_buf,
                                                     first);
}

template <unsigned FIRST_TID, unsigned LAST_TID, class INDEX_T>
__device__ inline void hashmap_restore(uint32_t* hashmap_ptr,
                                       const size_t hashmap_bitlen,
                                       const INDEX_T* itopk_indices,
                                       uint32_t itopk_size)
{
  if (threadIdx.x < FIRST_TID || threadIdx.x >= LAST_TID) return;
  for (unsigned i = threadIdx.x - FIRST_TID; i < itopk_size; i += LAST_TID - FIRST_TID) {
    auto key = itopk_indices[i] & ~0x80000000;  // clear most significant bit
    hashmap::insert(hashmap_ptr, hashmap_bitlen, key);
  }
}

template <class T, unsigned BLOCK_SIZE>
__device__ inline void set_value_device(T* const ptr, const T fill, const std::uint32_t count)
{
  for (std::uint32_t i = threadIdx.x; i < count; i += BLOCK_SIZE) {
    ptr[i] = fill;
  }
}

// One query one thread block
template <unsigned TEAM_SIZE,
          unsigned BLOCK_SIZE,
          unsigned BLOCK_COUNT,
          unsigned MAX_ITOPK,
          unsigned MAX_CANDIDATES,
          unsigned TOPK_BY_BITONIC_SORT,
          unsigned MAX_DATASET_DIM,
          class DATA_T,
          class DISTANCE_T,
          class INDEX_T,
          class LOAD_T>
__launch_bounds__(BLOCK_SIZE, BLOCK_COUNT) __global__
  void search_kernel(INDEX_T* const result_indices_ptr,       // [num_queries, top_k]
                     DISTANCE_T* const result_distances_ptr,  // [num_queries, top_k]
                     const std::uint32_t top_k,
                     const DATA_T* const dataset_ptr,  // [dataset_size, dataset_dim]
                     const std::size_t dataset_dim,
                     const std::size_t dataset_size,
                     const DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
                     const INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
                     const std::uint32_t graph_degree,
                     const unsigned num_distilation,
                     const uint64_t rand_xor_mask,
                     const INDEX_T* seed_ptr,  // [num_queries, num_seeds]
                     const uint32_t num_seeds,
                     std::uint32_t* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
                     const std::uint32_t internal_topk,
                     const std::uint32_t num_parents,
                     const std::uint32_t min_iteration,
                     const std::uint32_t max_iteration,
                     std::uint32_t* const num_executed_iterations,  // [num_queries]
                     const std::uint32_t hash_bitlen,
                     const std::uint32_t small_hash_bitlen,
                     const std::uint32_t small_hash_reset_interval)
{
  const auto query_id = blockIdx.y;

#ifdef _CLK_BREAKDOWN
  std::uint64_t clk_init                 = 0;
  std::uint64_t clk_compute_1st_distance = 0;
  std::uint64_t clk_topk                 = 0;
  std::uint64_t clk_reset_hash           = 0;
  std::uint64_t clk_pickup_parents       = 0;
  std::uint64_t clk_restore_hash         = 0;
  std::uint64_t clk_compute_distance     = 0;
  std::uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ std::uint32_t smem[];

  // Layout of result_buffer
  // +----------------------+------------------------------+---------+
  // | internal_top_k       | neighbors of internal_top_k  | padding |
  // | <internal_topk_size> | <num_parents * graph_degree> | upto 32 |
  // +----------------------+------------------------------+---------+
  // |<---             result_buffer_size              --->|
  std::uint32_t result_buffer_size    = internal_topk + (num_parents * graph_degree);
  std::uint32_t result_buffer_size_32 = result_buffer_size;
  if (result_buffer_size % 32) { result_buffer_size_32 += 32 - (result_buffer_size % 32); }
  const auto small_hash_size = hashmap::get_size(small_hash_bitlen);
  auto query_buffer          = reinterpret_cast<float*>(smem);
  auto result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + MAX_DATASET_DIM);
  auto result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto visited_hash_buffer =
    reinterpret_cast<std::uint32_t*>(result_distances_buffer + result_buffer_size_32);
  auto parent_list_buffer = reinterpret_cast<std::uint32_t*>(visited_hash_buffer + small_hash_size);
  auto topk_ws            = reinterpret_cast<std::uint32_t*>(parent_list_buffer + num_parents);
  auto terminate_flag     = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto smem_working_ptr   = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  const DATA_T* const query_ptr = queries_ptr + query_id * dataset_dim;
  for (unsigned i = threadIdx.x; i < MAX_DATASET_DIM; i += BLOCK_SIZE) {
    unsigned j = device::swizzling(i);
    if (i < dataset_dim) {
      query_buffer[j] = static_cast<float>(query_ptr[i]) * device::fragment_scale<DATA_T>();
    } else {
      query_buffer[j] = 0.0;
    }
  }
  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  // Init hashmap
  uint32_t* local_visited_hashmap_ptr;
  if (small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * query_id);
  }
  hashmap::init<0, BLOCK_SIZE>(local_visited_hashmap_ptr, hash_bitlen);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes
  _CLK_START();
  const INDEX_T* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  device::compute_distance_to_random_nodes<TEAM_SIZE, MAX_DATASET_DIM, LOAD_T>(
    result_indices_buffer,
    result_distances_buffer,
    query_buffer,
    dataset_ptr,
    dataset_dim,
    dataset_size,
    result_buffer_size,
    num_distilation,
    rand_xor_mask,
    local_seed_ptr,
    num_seeds,
    local_visited_hashmap_ptr,
    hash_bitlen);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  std::uint32_t iter = 0;
  while (1) {
    // sort
    if (TOPK_BY_BITONIC_SORT) {
      // [Notice]
      // It is good to use multiple warps in topk_by_bitonic_sort() when
      // batch size is small (short-latency), but it might not be always good
      // when batch size is large (high-throughput).
      // topk_by_bitonic_sort() consists of two operations:
      // if MAX_CANDIDATES is greater than 128, the first operation uses two warps;
      // if MAX_ITOPK is greater than 256, the second operation used two warps.
      constexpr unsigned multi_warps_1 = ((BLOCK_SIZE >= 64) && (MAX_CANDIDATES > 128)) ? 1 : 0;
      constexpr unsigned multi_warps_2 = ((BLOCK_SIZE >= 64) && (MAX_ITOPK > 256)) ? 1 : 0;

      // reset small-hash table.
      if ((iter + 1) % small_hash_reset_interval == 0) {
        // Depending on the block size and the number of warps used in
        // topk_by_bitonic_sort(), determine which warps are used to reset
        // the small hash and whether they are performed in overlap with
        // topk_by_bitonic_sort().
        _CLK_START();
        if (BLOCK_SIZE == 32) {
          hashmap::init<0, BLOCK_SIZE>(local_visited_hashmap_ptr, hash_bitlen);
        } else if (BLOCK_SIZE == 64) {
          if (multi_warps_1 || multi_warps_2) {
            hashmap::init<0, BLOCK_SIZE>(local_visited_hashmap_ptr, hash_bitlen);
          } else {
            hashmap::init<32, BLOCK_SIZE>(local_visited_hashmap_ptr, hash_bitlen);
          }
        } else {
          if (multi_warps_1 || multi_warps_2) {
            hashmap::init<64, BLOCK_SIZE>(local_visited_hashmap_ptr, hash_bitlen);
          } else {
            hashmap::init<32, BLOCK_SIZE>(local_visited_hashmap_ptr, hash_bitlen);
          }
        }
        _CLK_REC(clk_reset_hash);
      }

      // topk with bitonic sort
      _CLK_START();
      topk_by_bitonic_sort<MAX_ITOPK, MAX_CANDIDATES, multi_warps_1, multi_warps_2>(
        result_distances_buffer,
        result_indices_buffer,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        num_parents * graph_degree,
        topk_ws,
        (iter == 0));
      _CLK_REC(clk_topk);

    } else {
      _CLK_START();
      // topk with radix block sort
      topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>{}(
        internal_topk,
        gridDim.x,
        result_buffer_size,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        reinterpret_cast<std::uint32_t*>(result_distances_buffer),
        result_indices_buffer,
        nullptr,
        topk_ws,
        true,
        reinterpret_cast<std::uint32_t*>(smem_working_ptr));
      _CLK_REC(clk_topk);

      // reset small-hash table
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        hashmap::init<0, BLOCK_SIZE>(local_visited_hashmap_ptr, hash_bitlen);
        _CLK_REC(clk_reset_hash);
      }
    }
    __syncthreads();

    if (iter + 1 == max_iteration) { break; }

    // pick up next parents
    if (threadIdx.x < 32) {
      _CLK_START();
      pickup_next_parents<TOPK_BY_BITONIC_SORT, INDEX_T>(terminate_flag,
                                                         parent_list_buffer,
                                                         result_indices_buffer,
                                                         internal_topk,
                                                         dataset_size,
                                                         num_parents);
      _CLK_REC(clk_pickup_parents);
    }

    // restore small-hash table by putting internal-topk indices in it
    if ((iter + 1) % small_hash_reset_interval == 0) {
      constexpr unsigned first_tid = ((BLOCK_SIZE <= 32) ? 0 : 32);
      _CLK_START();
      hashmap_restore<first_tid, BLOCK_SIZE>(
        local_visited_hashmap_ptr, hash_bitlen, result_indices_buffer, internal_topk);
      _CLK_REC(clk_restore_hash);
    }
    __syncthreads();

    if (*terminate_flag && iter >= min_iteration) { break; }

    // compute the norms between child nodes and query node
    _CLK_START();
    constexpr unsigned max_n_frags = 16;
    device::
      compute_distance_to_child_nodes<TEAM_SIZE, BLOCK_SIZE, MAX_DATASET_DIM, max_n_frags, LOAD_T>(
        result_indices_buffer + internal_topk,
        result_distances_buffer + internal_topk,
        query_buffer,
        dataset_ptr,
        dataset_dim,
        knn_graph,
        graph_degree,
        local_visited_hashmap_ptr,
        hash_bitlen,
        parent_list_buffer,
        num_parents);
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    iter++;
  }
  for (std::uint32_t i = threadIdx.x; i < top_k; i += BLOCK_SIZE) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }
    result_indices_ptr[j] = result_indices_buffer[ii] & ~0x80000000;  // clear most significant bit
  }
  if (threadIdx.x == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }
#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == BLOCK_SIZE - 1) && ((query_id * 3) % gridDim.y < 3)) {
    RAFT_LOG_DEBUG(
      "query, %d, thread, %d"
      ", init, %d"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", reset_hash, %lu"
      ", pickup_parents, %lu"
      ", restore_hash, %lu"
      ", distance, %lu"
      "\n",
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_reset_hash,
      clk_pickup_parents,
      clk_restore_hash,
      clk_compute_distance);
  }
#endif
}

#define SET_KERNEL_3(                                                               \
  BLOCK_SIZE, BLOCK_COUNT, MAX_ITOPK, MAX_CANDIDATES, TOPK_BY_BITONIC_SORT, LOAD_T) \
  kernel = search_kernel<TEAM_SIZE,                                                 \
                         BLOCK_SIZE,                                                \
                         BLOCK_COUNT,                                               \
                         MAX_ITOPK,                                                 \
                         MAX_CANDIDATES,                                            \
                         TOPK_BY_BITONIC_SORT,                                      \
                         MAX_DATASET_DIM,                                           \
                         DATA_T,                                                    \
                         DISTANCE_T,                                                \
                         INDEX_T,                                                   \
                         LOAD_T>;

#define SET_KERNEL_2(BLOCK_SIZE, BLOCK_COUNT, MAX_ITOPK, MAX_CANDIDATES, TOPK_BY_BITONIC_SORT) \
  if (load_bit_length == 128) {                                                                \
    SET_KERNEL_3(BLOCK_SIZE,                                                                   \
                 BLOCK_COUNT,                                                                  \
                 MAX_ITOPK,                                                                    \
                 MAX_CANDIDATES,                                                               \
                 TOPK_BY_BITONIC_SORT,                                                         \
                 device::LOAD_128BIT_T)                                                        \
  } else if (load_bit_length == 64) {                                                          \
    SET_KERNEL_3(BLOCK_SIZE,                                                                   \
                 BLOCK_COUNT,                                                                  \
                 MAX_ITOPK,                                                                    \
                 MAX_CANDIDATES,                                                               \
                 TOPK_BY_BITONIC_SORT,                                                         \
                 device::LOAD_64BIT_T)                                                         \
  }

#define SET_KERNEL_1B(MAX_ITOPK, MAX_CANDIDATES)              \
  /* if ( block_size == 32 ) {                                \
      SET_KERNEL_2( 32, 20, MAX_ITOPK, MAX_CANDIDATES, 1 )    \
  } else */                                                   \
  if (block_size == 64) {                                     \
    SET_KERNEL_2(64, 16 /*20*/, MAX_ITOPK, MAX_CANDIDATES, 1) \
  } else if (block_size == 128) {                             \
    SET_KERNEL_2(128, 8, MAX_ITOPK, MAX_CANDIDATES, 1)        \
  } else if (block_size == 256) {                             \
    SET_KERNEL_2(256, 4, MAX_ITOPK, MAX_CANDIDATES, 1)        \
  } else if (block_size == 512) {                             \
    SET_KERNEL_2(512, 2, MAX_ITOPK, MAX_CANDIDATES, 1)        \
  } else {                                                    \
    SET_KERNEL_2(1024, 1, MAX_ITOPK, MAX_CANDIDATES, 1)       \
  }

#define SET_KERNEL_1R(MAX_ITOPK, MAX_CANDIDATES)        \
  if (block_size == 256) {                              \
    SET_KERNEL_2(256, 4, MAX_ITOPK, MAX_CANDIDATES, 0)  \
  } else if (block_size == 512) {                       \
    SET_KERNEL_2(512, 2, MAX_ITOPK, MAX_CANDIDATES, 0)  \
  } else {                                              \
    SET_KERNEL_2(1024, 1, MAX_ITOPK, MAX_CANDIDATES, 0) \
  }

#define SET_KERNEL                                                                \
  typedef void (*search_kernel_t)(INDEX_T* const result_indices_ptr,              \
                                  DISTANCE_T* const result_distances_ptr,         \
                                  const std::uint32_t top_k,                      \
                                  const DATA_T* const dataset_ptr,                \
                                  const std::size_t dataset_dim,                  \
                                  const std::size_t dataset_size,                 \
                                  const DATA_T* const queries_ptr,                \
                                  const INDEX_T* const knn_graph,                 \
                                  const std::uint32_t graph_degree,               \
                                  const unsigned num_distilation,                 \
                                  const uint64_t rand_xor_mask,                   \
                                  const INDEX_T* seed_ptr,                        \
                                  const uint32_t num_seeds,                       \
                                  std::uint32_t* const visited_hashmap_ptr,       \
                                  const std::uint32_t itopk_size,                 \
                                  const std::uint32_t num_parents,                \
                                  const std::uint32_t min_iteration,              \
                                  const std::uint32_t max_iteration,              \
                                  std::uint32_t* const num_executed_iterations,   \
                                  const std::uint32_t hash_bitlen,                \
                                  const std::uint32_t small_hash_bitlen,          \
                                  const std::uint32_t small_hash_reset_interval); \
  search_kernel_t kernel;                                                         \
  if (num_itopk_candidates <= 64) {                                               \
    constexpr unsigned max_candidates = 64;                                       \
    if (itopk_size <= 64) {                                                       \
      SET_KERNEL_1B(64, max_candidates)                                           \
    } else if (itopk_size <= 128) {                                               \
      SET_KERNEL_1B(128, max_candidates)                                          \
    } else if (itopk_size <= 256) {                                               \
      SET_KERNEL_1B(256, max_candidates)                                          \
    } else if (itopk_size <= 512) {                                               \
      SET_KERNEL_1B(512, max_candidates)                                          \
    }                                                                             \
  } else if (num_itopk_candidates <= 128) {                                       \
    constexpr unsigned max_candidates = 128;                                      \
    if (itopk_size <= 64) {                                                       \
      SET_KERNEL_1B(64, max_candidates)                                           \
    } else if (itopk_size <= 128) {                                               \
      SET_KERNEL_1B(128, max_candidates)                                          \
    } else if (itopk_size <= 256) {                                               \
      SET_KERNEL_1B(256, max_candidates)                                          \
    } else if (itopk_size <= 512) {                                               \
      SET_KERNEL_1B(512, max_candidates)                                          \
    }                                                                             \
  } else if (num_itopk_candidates <= 256) {                                       \
    constexpr unsigned max_candidates = 256;                                      \
    if (itopk_size <= 64) {                                                       \
      SET_KERNEL_1B(64, max_candidates)                                           \
    } else if (itopk_size <= 128) {                                               \
      SET_KERNEL_1B(128, max_candidates)                                          \
    } else if (itopk_size <= 256) {                                               \
      SET_KERNEL_1B(256, max_candidates)                                          \
    } else if (itopk_size <= 512) {                                               \
      SET_KERNEL_1B(512, max_candidates)                                          \
    }                                                                             \
  } else {                                                                        \
    /* Radix-based topk is used */                                                \
    if (itopk_size <= 256) {                                                      \
      SET_KERNEL_1R(256, /*to avoid build failure*/ 32)                           \
    } else if (itopk_size <= 512) {                                               \
      SET_KERNEL_1R(512, /*to avoid build failure*/ 32)                           \
    }                                                                             \
  }

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          typename DATA_T,
          typename INDEX_T,
          typename DISTANCE_T>
struct search : search_plan_impl<DATA_T, INDEX_T, DISTANCE_T> {
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::max_queries;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::itopk_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::algo;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::team_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_parents;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::min_iterations;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::max_iterations;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::load_bit_length;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::thread_block_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_mode;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_min_bitlen;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_max_fill_rate;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_random_samplings;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::rand_xor_mask;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::max_dim;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::dim;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::graph_degree;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::topk;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hash_bitlen;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::small_hash_bitlen;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::small_hash_reset_interval;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::dataset_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::result_buffer_size;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::smem_size;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::load_bit_lenght;

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_executed_iterations;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::dev_seed;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_seeds;

  uint32_t num_itopk_candidates;

  search(raft::device_resources const& res,
         search_params params,
         int64_t dim,
         int64_t graph_degree,
         uint32_t topk)
    : search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>(res, params, dim, graph_degree, topk)
  {
    set_params(res);
  }

  ~search() {}

  inline void set_params(raft::device_resources const& res)
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
        smem_size += topk_by_radix_sort<256, max_block_size>::smem_size * sizeof(std::uint32_t);
      } else {
        smem_size += topk_by_radix_sort<512, max_block_size>::smem_size * sizeof(std::uint32_t);
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
          smem_size += topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size * sizeof(std::uint32_t);
        } else if (block_size == 512) {
          constexpr unsigned BLOCK_SIZE = 512;
          smem_size += topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size * sizeof(std::uint32_t);
        } else {
          constexpr unsigned BLOCK_SIZE = 1024;
          smem_size += topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size * sizeof(std::uint32_t);
        }
      } else {
        constexpr unsigned MAX_ITOPK = 512;
        if (block_size == 256) {
          constexpr unsigned BLOCK_SIZE = 256;
          smem_size += topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size * sizeof(std::uint32_t);
        } else if (block_size == 512) {
          constexpr unsigned BLOCK_SIZE = 512;
          smem_size += topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size * sizeof(std::uint32_t);
        } else {
          constexpr unsigned BLOCK_SIZE = 1024;
          smem_size += topk_by_radix_sort<MAX_ITOPK, BLOCK_SIZE>::smem_size * sizeof(std::uint32_t);
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

  void operator()(raft::device_resources const& res,
                  raft::device_matrix_view<const DATA_T, INDEX_T, row_major> dataset,
                  raft::device_matrix_view<const INDEX_T, INDEX_T, row_major> graph,
                  INDEX_T* const result_indices_ptr,       // [num_queries, topk]
                  DISTANCE_T* const result_distances_ptr,  // [num_queries, topk]
                  const DATA_T* const queries_ptr,         // [num_queries, dataset_dim]
                  const std::uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,                   // [num_queries, num_seeds]
                  std::uint32_t* const num_executed_iterations,  // [num_queries]
                  uint32_t topk)
  {
    cudaStream_t stream = res.get_stream();
    uint32_t block_size = thread_block_size;
    SET_KERNEL;
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    dim3 thread_dims(block_size, 1, 1);
    dim3 block_dims(1, num_queries, 1);
    RAFT_LOG_DEBUG(
      "Launching kernel with %u threads, %u block %lu smem", block_size, num_queries, smem_size);
    kernel<<<block_dims, thread_dims, smem_size, stream>>>(result_indices_ptr,
                                                           result_distances_ptr,
                                                           topk,
                                                           dataset.data_handle(),
                                                           dataset.extent(1),
                                                           dataset.extent(0),
                                                           queries_ptr,
                                                           graph.data_handle(),
                                                           graph.extent(1),
                                                           num_random_samplings,
                                                           rand_xor_mask,
                                                           dev_seed_ptr,
                                                           num_seeds,
                                                           hashmap.data(),
                                                           itopk_size,
                                                           num_parents,
                                                           min_iterations,
                                                           max_iterations,
                                                           num_executed_iterations,
                                                           hash_bitlen,
                                                           small_hash_bitlen,
                                                           small_hash_reset_interval);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
};

}  // namespace single_cta_search
}  // namespace raft::neighbors::experimental::cagra::detail
