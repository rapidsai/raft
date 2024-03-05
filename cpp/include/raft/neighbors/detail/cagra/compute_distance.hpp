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

#include <raft/distance/distance_types.hpp>
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

template <class LOAD_T,
          class DATA_T,
          class DISTANCE_T,
          std::uint32_t DATASET_BLOCK_DIM,
          std::uint32_t TEAM_SIZE,
          bool use_reg_fragment>
struct distance_op;
template <class LOAD_T,
          class DATA_T,
          class DISTANCE_T,
          std::uint32_t DATASET_BLOCK_DIM,
          std::uint32_t TEAM_SIZE>
struct distance_op<LOAD_T, DATA_T, DISTANCE_T, DATASET_BLOCK_DIM, TEAM_SIZE, false> {
  const float* const query_buffer;
  __device__ distance_op(const float* const query_buffer) : query_buffer(query_buffer) {}

  __device__ DISTANCE_T operator()(const DATA_T* const dataset_ptr,
                                   const std::uint32_t dataset_dim,
                                   const bool valid,
                                   raft::distance::DistanceType metric)
  {
    const unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = get_vlen<LOAD_T, DATA_T>();
    constexpr unsigned reg_nelem =
      (DATASET_BLOCK_DIM + (TEAM_SIZE * vlen) - 1) / (TEAM_SIZE * vlen);
    data_load_t<LOAD_T, DATA_T, vlen> dl_buff[reg_nelem];

    DISTANCE_T norm2 = 0;
    if (valid) {
      for (uint32_t elem_offset = 0; elem_offset < dataset_dim; elem_offset += DATASET_BLOCK_DIM) {
#pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
          const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen + elem_offset;
          if (k >= dataset_dim) break;
          dl_buff[e].load = *reinterpret_cast<const LOAD_T*>(dataset_ptr + k);
        }
#pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
          const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen + elem_offset;
          if (k >= dataset_dim) break;
#pragma unroll
          for (uint32_t v = 0; v < vlen; v++) {
            const uint32_t kv = k + v;
            // if (kv >= dataset_dim) break;
            DISTANCE_T diff = query_buffer[device::swizzling(kv)];
            if (metric == raft::distance::L2Expanded) {
              diff -= spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].data[v]);
              norm2 += diff * diff;
            } else {
              diff *= spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].data[v]);
              norm2 -= diff;
            }
          }
        }
      }
    }
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
      norm2 += __shfl_xor_sync(0xffffffff, norm2, offset);
    }
    return norm2;
  }
};
template <class LOAD_T,
          class DATA_T,
          class DISTANCE_T,
          std::uint32_t DATASET_BLOCK_DIM,
          std::uint32_t TEAM_SIZE>
struct distance_op<LOAD_T, DATA_T, DISTANCE_T, DATASET_BLOCK_DIM, TEAM_SIZE, true> {
  static constexpr unsigned N_FRAGS = (DATASET_BLOCK_DIM + TEAM_SIZE - 1) / TEAM_SIZE;
  float query_frags[N_FRAGS];

  __device__ distance_op(const float* const query_buffer)
  {
    constexpr unsigned vlen = get_vlen<LOAD_T, DATA_T>();
    constexpr unsigned reg_nelem =
      (DATASET_BLOCK_DIM + (TEAM_SIZE * vlen) - 1) / (TEAM_SIZE * vlen);
    const std::uint32_t lane_id = threadIdx.x % TEAM_SIZE;
    // Pre-load query vectors into registers when register usage is not too large.
#pragma unroll
    for (unsigned e = 0; e < reg_nelem; e++) {
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

  __device__ DISTANCE_T operator()(const DATA_T* const dataset_ptr,
                                   const std::uint32_t dataset_dim,
                                   const bool valid,
                                   raft::distance::DistanceType metric)
  {
    const unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = get_vlen<LOAD_T, DATA_T>();
    constexpr unsigned reg_nelem =
      (DATASET_BLOCK_DIM + (TEAM_SIZE * vlen) - 1) / (TEAM_SIZE * vlen);
    data_load_t<LOAD_T, DATA_T, vlen> dl_buff[reg_nelem];

    DISTANCE_T norm2 = 0;
    if (valid) {
#pragma unroll
      for (unsigned e = 0; e < reg_nelem; e++) {
        const unsigned k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dataset_dim) break;
        dl_buff[e].load = *reinterpret_cast<const LOAD_T*>(dataset_ptr + k);
      }
#pragma unroll
      for (unsigned e = 0; e < reg_nelem; e++) {
        const unsigned k = (lane_id + (TEAM_SIZE * e)) * vlen;
        if (k >= dataset_dim) break;
#pragma unroll
        for (unsigned v = 0; v < vlen; v++) {
          DISTANCE_T diff;
          const unsigned ev = (vlen * e) + v;
          diff              = query_frags[ev];
          if (metric == raft::distance::L2Expanded) {
              diff -= spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].data[v]);
              norm2 += diff * diff;
            } else {
              diff *= spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].data[v]);
              norm2 -= diff;
            }
        }
      }
    }
    for (uint32_t offset = TEAM_SIZE / 2; offset > 0; offset >>= 1) {
      norm2 += __shfl_xor_sync(0xffffffff, norm2, offset);
    }
    return norm2;
  }
};

template <unsigned TEAM_SIZE,
          unsigned DATASET_BLOCK_DIM,
          class LOAD_T,
          class DATA_T,
          class DISTANCE_T,
          class INDEX_T>
_RAFT_DEVICE void compute_distance_to_random_nodes(
  INDEX_T* const result_indices_ptr,       // [num_pickup]
  DISTANCE_T* const result_distances_ptr,  // [num_pickup]
  const float* const query_buffer,
  const DATA_T* const dataset_ptr,  // [dataset_size, dataset_dim]
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
  raft::distance::DistanceType metric,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1)
{
  uint32_t max_i = num_pickup;
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }

  distance_op<LOAD_T, DATA_T, DISTANCE_T, DATASET_BLOCK_DIM, TEAM_SIZE, false> dist_op(
    query_buffer);

  for (uint32_t i = threadIdx.x / TEAM_SIZE; i < max_i; i += blockDim.x / TEAM_SIZE) {
    const bool valid_i = (i < num_pickup);

    INDEX_T best_index_team_local;
    DISTANCE_T best_norm2_team_local = utils::get_max_value<DISTANCE_T>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      // Select a node randomly and compute the distance to it
      INDEX_T seed_index;
      if (valid_i) {
        // uint32_t gid = i + (num_pickup * (j + (num_distilation * block_id)));
        uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % dataset_size;
        }
      }

      const auto norm2 = dist_op(dataset_ptr + dataset_ld * seed_index, dataset_dim, valid_i, metric);

      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x % TEAM_SIZE;
    if (valid_i && lane_id == 0) {
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
          unsigned DATASET_BLOCK_DIM,
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
                                                  const INDEX_T* const internal_topk_list,
                                                  const std::uint32_t search_width,
                                                  raft::distance::DistanceType metric)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;
  const INDEX_T invalid_index        = utils::get_max_value<INDEX_T>();

  // Read child indices of parents from knn graph and check if the distance
  // computaiton is necessary.
  for (uint32_t i = threadIdx.x; i < knn_k * search_width; i += blockDim.x) {
    const INDEX_T smem_parent_id = parent_indices[i / knn_k];
    INDEX_T child_id             = invalid_index;
    if (smem_parent_id != invalid_index) {
      const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
      child_id             = knn_graph[(i % knn_k) + (static_cast<int64_t>(knn_k) * parent_id)];
    }
    if (child_id != invalid_index) {
      if (hashmap::insert(visited_hashmap_ptr, hash_bitlen, child_id) == 0) {
        child_id = invalid_index;
      }
    }
    result_child_indices_ptr[i] = child_id;
  }

  // [Notice]
  //   Loading the query vector here from shared memory into registers reduces
  //   shared memory trafiic. However, register usage increase. The
  //   MAX_N_FRAGS below is used as the threshold to enable or disable this,
  //   but the appropriate value should be discussed.
  constexpr unsigned N_FRAGS  = (DATASET_BLOCK_DIM + TEAM_SIZE - 1) / TEAM_SIZE;
  constexpr bool use_fragment = N_FRAGS <= MAX_N_FRAGS;
  distance_op<LOAD_T, DATA_T, DISTANCE_T, DATASET_BLOCK_DIM, TEAM_SIZE, use_fragment> dist_op(
    query_buffer);
  __syncthreads();

  // Compute the distance to child nodes
  std::uint32_t max_i = knn_k * search_width;
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }
  for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x) {
    const auto i       = tid / TEAM_SIZE;
    const bool valid_i = (i < (knn_k * search_width));
    INDEX_T child_id   = invalid_index;
    if (valid_i) { child_id = result_child_indices_ptr[i]; }

    DISTANCE_T norm2 =
      dist_op(dataset_ptr + child_id * dataset_ld, dataset_dim, child_id != invalid_index, metric);

    // Store the distance
    const unsigned lane_id = threadIdx.x % TEAM_SIZE;
    if (valid_i && lane_id == 0) {
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
