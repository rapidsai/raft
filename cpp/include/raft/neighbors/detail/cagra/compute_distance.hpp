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

#include <raft/spatial/knn/detail/ann_utils.cuh>

#include "device_common.hpp"
#include "hashmap.hpp"
#include "utils.hpp"
#include <type_traits>

namespace raft::neighbors::cagra::detail {
namespace device {

// using LOAD_256BIT_T = ulonglong4;
using LOAD_128BIT_T = uint4;
using LOAD_64BIT_T  = uint64_t;

template <class LOAD_T, class DATA_T>
_RAFT_DEVICE constexpr unsigned get_vlen()
{
  return utils::size_of<LOAD_T>() / utils::size_of<DATA_T>();
}

template <class LOAD_T, class DATA_T, unsigned VLEN>
struct data_load_t {
  union {
    LOAD_T load;
    DATA_T data[VLEN];
  };
};

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          class LOAD_T,
          class DATA_T,
          class DISTANCE_T,
          class INDEX_T>
_RAFT_DEVICE void compute_distance_to_random_nodes(
  INDEX_T* const result_indices_ptr,       // [num_pickup]
  DISTANCE_T* const result_distances_ptr,  // [num_pickup]
  const float* const query_buffer,
  const DATA_T* const dataset_ptr,         // [dataset_size, dataset_dim]
  const std::size_t dataset_dim,
  const std::size_t dataset_size,
  const std::size_t dataset_ld,
  const std::size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const INDEX_T* const seed_ptr,  // [num_seeds]
  const uint32_t num_seeds,
  INDEX_T* const visited_hash_ptr,
  const uint32_t hash_bitlen,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1)
{
  const unsigned lane_id   = threadIdx.x % TEAM_SIZE;
  constexpr unsigned vlen  = get_vlen<LOAD_T, DATA_T>();
  constexpr unsigned nelem = (MAX_DATASET_DIM + (TEAM_SIZE * vlen) - 1) / (TEAM_SIZE * vlen);
  struct data_load_t<LOAD_T, DATA_T, vlen> dl_buff[nelem];
  uint32_t max_i = num_pickup;
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }
  for (uint32_t i = threadIdx.x / TEAM_SIZE; i < max_i; i += blockDim.x / TEAM_SIZE) {
    const bool valid_i = (i < num_pickup);

    INDEX_T best_index_team_local;
    DISTANCE_T best_norm2_team_local = utils::get_max_value<DISTANCE_T>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      // Select a node randomly and compute the distance to it
      INDEX_T seed_index;
      DISTANCE_T norm2 = 0.0;
      if (valid_i) {
        // uint32_t gid = i + (num_pickup * (j + (num_distilation * block_id)));
        uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % dataset_size;
        }
#pragma unroll
        for (uint32_t e = 0; e < nelem; e++) {
          const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
          if (k >= dataset_dim) break;
          dl_buff[e].load = ((LOAD_T*)(dataset_ptr + k + (dataset_ld * seed_index)))[0];
        }
#pragma unroll
        for (uint32_t e = 0; e < nelem; e++) {
          const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen;
          if (k >= dataset_dim) break;
#pragma unroll
          for (uint32_t v = 0; v < vlen; v++) {
            const uint32_t kv = k + v;
            // if (kv >= dataset_dim) break;
            DISTANCE_T diff = query_buffer[device::swizzling(kv)];
            diff -= spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].data[v]);
            norm2 += diff * diff;
          }
        }
      }
      for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
        norm2 += __shfl_xor_sync(0xffffffff, norm2, offset);
      }

      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }

    if (valid_i && (threadIdx.x % TEAM_SIZE == 0)) {
      if (hashmap::insert(visited_hash_ptr, hash_bitlen, best_index_team_local)) {
        result_distances_ptr[i] = best_norm2_team_local;
        result_indices_ptr[i]   = best_index_team_local;
      } else {
        result_distances_ptr[i] = utils::get_max_value<DISTANCE_T>();
        result_indices_ptr[i]   = utils::get_max_value<INDEX_T>();
      }
    }
  }
}

template <unsigned TEAM_SIZE,
          unsigned BLOCK_SIZE,
          unsigned MAX_DATASET_DIM,
          unsigned MAX_N_FRAGS,
          class LOAD_T,
          class DATA_T,
          class DISTANCE_T,
          class INDEX_T>
_RAFT_DEVICE void compute_distance_to_child_nodes(INDEX_T* const result_child_indices_ptr,
                                                  DISTANCE_T* const result_child_distances_ptr,
                                                  // query
                                                  const float* const query_buffer,
                                                  // [dataset_dim, dataset_size]
                                                  const DATA_T* const dataset_ptr,
                                                  const std::size_t dataset_dim,
                                                  const std::size_t dataset_ld,
                                                  // [knn_k, dataset_size]
                                                  const INDEX_T* const knn_graph,
                                                  const std::uint32_t knn_k,
                                                  // hashmap
                                                  INDEX_T* const visited_hashmap_ptr,
                                                  const std::uint32_t hash_bitlen,
                                                  const INDEX_T* const parent_indices,
                                                  const std::uint32_t num_parents)
{
  const INDEX_T invalid_index = utils::get_max_value<INDEX_T>();

  // Read child indices of parents from knn graph and check if the distance
  // computaiton is necessary.
  for (uint32_t i = threadIdx.x; i < knn_k * num_parents; i += BLOCK_SIZE) {
    const INDEX_T parent_id = parent_indices[i / knn_k];
    INDEX_T child_id        = invalid_index;
    if (parent_id != invalid_index) {
      child_id = knn_graph[(i % knn_k) + ((uint64_t)knn_k * parent_id)];
    }
    if (child_id != invalid_index) {
      if (hashmap::insert(visited_hashmap_ptr, hash_bitlen, child_id) == 0) {
        child_id = invalid_index;
      }
    }
    result_child_indices_ptr[i] = child_id;
  }

  constexpr unsigned vlen  = get_vlen<LOAD_T, DATA_T>();
  constexpr unsigned nelem = (MAX_DATASET_DIM + (TEAM_SIZE * vlen) - 1) / (TEAM_SIZE * vlen);
  const unsigned lane_id   = threadIdx.x % TEAM_SIZE;

  // [Notice]
  //   Loading the query vector here from shared memory into registers reduces
  //   shared memory trafiic. However, register usage increase. The
  //   MAX_N_FRAGS below is used as the threshold to enable or disable this,
  //   but the appropriate value should be discussed.
  constexpr unsigned N_FRAGS = (MAX_DATASET_DIM + TEAM_SIZE - 1) / TEAM_SIZE;
  float query_frags[N_FRAGS];
  if (N_FRAGS <= MAX_N_FRAGS) {
    // Pre-load query vectors into registers when register usage is not too large.
#pragma unroll
    for (unsigned e = 0; e < nelem; e++) {
      const unsigned k = (lane_id + (TEAM_SIZE * e)) * vlen;
      // if (k >= dataset_dim) break;
#pragma unroll
      for (unsigned v = 0; v < vlen; v++) {
        const unsigned kv = k + v;
        const unsigned ev = (vlen * e) + v;
        query_frags[ev]   = query_buffer[device::swizzling(kv)];
      }
    }
  }
  __syncthreads();

  // Compute the distance to child nodes
  std::uint32_t max_i = knn_k * num_parents;
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }
  for (std::uint32_t i = threadIdx.x / TEAM_SIZE; i < max_i; i += BLOCK_SIZE / TEAM_SIZE) {
    const bool valid_i = (i < (knn_k * num_parents));
    INDEX_T child_id   = invalid_index;
    if (valid_i) { child_id = result_child_indices_ptr[i]; }

    DISTANCE_T norm2 = 0.0;
    struct data_load_t<LOAD_T, DATA_T, vlen> dl_buff[nelem];
    if (child_id != invalid_index) {
#pragma unroll
      for (unsigned e = 0; e < nelem; e++) {
        const unsigned k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dataset_dim) break;
        dl_buff[e].load = ((LOAD_T*)(dataset_ptr + k + (dataset_ld * child_id)))[0];
      }
#pragma unroll
      for (unsigned e = 0; e < nelem; e++) {
        const unsigned k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dataset_dim) break;
#pragma unroll
        for (unsigned v = 0; v < vlen; v++) {
          DISTANCE_T diff;
          if (N_FRAGS <= MAX_N_FRAGS) {
            const unsigned ev = (vlen * e) + v;
            diff              = query_frags[ev];
          } else {
            const unsigned kv = k + v;
            diff              = query_buffer[device::swizzling(kv)];
          }
          diff -= spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].data[v]);
          norm2 += diff * diff;
        }
      }
    }
    for (unsigned offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
      norm2 += __shfl_xor_sync(0xffffffff, norm2, offset);
    }

    // Store the distance
    if (valid_i && (threadIdx.x % TEAM_SIZE == 0)) {
      if (child_id != invalid_index) {
        result_child_distances_ptr[i] = norm2;
      } else {
        result_child_distances_ptr[i] = utils::get_max_value<DISTANCE_T>();
      }
    }
  }
}

}  // namespace device
}  // namespace raft::neighbors::cagra::detail
