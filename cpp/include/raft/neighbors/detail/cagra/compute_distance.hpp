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

#include "device_common.hpp"
#include "hashmap.hpp"
#include "utils.hpp"

#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/vectorized.cuh>

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

template <unsigned TEAM_SIZE,
          unsigned DATASET_BLOCK_DIM,
          class DATASET_DESCRIPTOR_T,
          class DISTANCE_T,
          class INDEX_T>
_RAFT_DEVICE void compute_distance_to_random_nodes(
  INDEX_T* const result_indices_ptr,       // [num_pickup]
  DISTANCE_T* const result_distances_ptr,  // [num_pickup]
  const typename DATASET_DESCRIPTOR_T::QUERY_T* const query_buffer,
  const DATASET_DESCRIPTOR_T& dataset_desc,
  const std::size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const INDEX_T* const seed_ptr,  // [num_seeds]
  const uint32_t num_seeds,
  INDEX_T* const visited_hash_ptr,
  const uint32_t hash_bitlen,
  const raft::distance::DistanceType metric,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1)
{
  uint32_t max_i = num_pickup;
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }

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
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % dataset_desc.size;
        }
      }

      DISTANCE_T norm2;
      switch (metric) {
        case raft::distance::L2Expanded:
          norm2 = dataset_desc.template compute_similarity<DATASET_BLOCK_DIM,
                                                           TEAM_SIZE,
                                                           raft::distance::L2Expanded>(
            query_buffer, seed_index, valid_i);
          break;
        case raft::distance::InnerProduct:
          norm2 = dataset_desc.template compute_similarity<DATASET_BLOCK_DIM,
                                                           TEAM_SIZE,
                                                           raft::distance::InnerProduct>(
            query_buffer, seed_index, valid_i);
          break;
        default: break;
      }

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
          class DATASET_DESCRIPTOR_T,
          class DISTANCE_T,
          class INDEX_T>
_RAFT_DEVICE void compute_distance_to_child_nodes(
  INDEX_T* const result_child_indices_ptr,
  DISTANCE_T* const result_child_distances_ptr,
  // query
  const typename DATASET_DESCRIPTOR_T::QUERY_T* const query_buffer,
  // [dataset_dim, dataset_size]
  const DATASET_DESCRIPTOR_T& dataset_desc,
  // [knn_k, dataset_size]
  const INDEX_T* const knn_graph,
  const std::uint32_t knn_k,
  // hashmap
  INDEX_T* const visited_hashmap_ptr,
  const std::uint32_t hash_bitlen,
  const INDEX_T* const parent_indices,
  const INDEX_T* const internal_topk_list,
  const std::uint32_t search_width,
  const raft::distance::DistanceType metric)
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
  __syncthreads();

  // Compute the distance to child nodes
  std::uint32_t max_i = knn_k * search_width;
  if (max_i % (32 / TEAM_SIZE)) { max_i += (32 / TEAM_SIZE) - (max_i % (32 / TEAM_SIZE)); }
  for (std::uint32_t tid = threadIdx.x; tid < max_i * TEAM_SIZE; tid += blockDim.x) {
    const auto i       = tid / TEAM_SIZE;
    const bool valid_i = (i < (knn_k * search_width));
    INDEX_T child_id   = invalid_index;
    if (valid_i) { child_id = result_child_indices_ptr[i]; }

    DISTANCE_T norm2;
    switch (metric) {
      case raft::distance::L2Expanded:
        norm2 =
          dataset_desc
            .template compute_similarity<DATASET_BLOCK_DIM, TEAM_SIZE, raft::distance::L2Expanded>(
              query_buffer, child_id, child_id != invalid_index);
        break;
      case raft::distance::InnerProduct:
        norm2 = dataset_desc.template compute_similarity<DATASET_BLOCK_DIM,
                                                         TEAM_SIZE,
                                                         raft::distance::InnerProduct>(
          query_buffer, child_id, child_id != invalid_index);
        break;
      default: break;
    }

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

template <class QUERY_T_, class DISTANCE_T_, class INDEX_T_>
struct dataset_descriptor_base_t {
  using INDEX_T    = INDEX_T_;
  using QUERY_T    = QUERY_T_;
  using DISTANCE_T = DISTANCE_T_;

  const INDEX_T size;
  const std::uint32_t dim;

  dataset_descriptor_base_t(const INDEX_T size, const std::uint32_t dim) : size(size), dim(dim) {}
};

template <class DATA_T_, class INDEX_T, class DISTANCE_T = float>
struct standard_dataset_descriptor_t
  : public dataset_descriptor_base_t<float, DISTANCE_T, INDEX_T> {
  using LOAD_T  = device::LOAD_128BIT_T;
  using DATA_T  = DATA_T_;
  using QUERY_T = typename dataset_descriptor_base_t<float, DISTANCE_T, INDEX_T>::QUERY_T;

  const DATA_T* const ptr;
  const std::size_t ld;
  using dataset_descriptor_base_t<float, DISTANCE_T, INDEX_T>::size;
  using dataset_descriptor_base_t<float, DISTANCE_T, INDEX_T>::dim;

  standard_dataset_descriptor_t(const DATA_T* const ptr,
                                const std::size_t size,
                                const std::uint32_t dim,
                                const std::size_t ld)
    : dataset_descriptor_base_t<float, DISTANCE_T, INDEX_T>(size, dim), ptr(ptr), ld(ld)
  {
  }

  static const std::uint32_t smem_buffer_size_in_byte = 0;
  __device__ void set_smem_ptr(void* const){};

  template <uint32_t DATASET_BLOCK_DIM>
  __device__ void copy_query(const DATA_T* const dmem_query_ptr,
                             QUERY_T* const smem_query_ptr,
                             const std::uint32_t query_smem_buffer_length)
  {
    for (unsigned i = threadIdx.x; i < query_smem_buffer_length; i += blockDim.x) {
      unsigned j = device::swizzling(i);
      if (i < dim) {
        smem_query_ptr[j] = spatial::knn::detail::utils::mapping<QUERY_T>{}(dmem_query_ptr[i]);
      } else {
        smem_query_ptr[j] = 0.0;
      }
    }
  }

  template <typename T, raft::distance::DistanceType METRIC>
  std::enable_if_t<METRIC == raft::distance::DistanceType::L2Expanded, T> __device__
  dist_op(T a, T b) const
  {
    T diff = a - b;
    return diff * diff;
  }

  template <typename T, raft::distance::DistanceType METRIC>
  std::enable_if_t<METRIC == raft::distance::DistanceType::InnerProduct, T> __device__
  dist_op(T a, T b) const
  {
    return -a * b;
  }

  template <uint32_t DATASET_BLOCK_DIM, uint32_t TEAM_SIZE, raft::distance::DistanceType METRIC>
  __device__ DISTANCE_T compute_similarity(const QUERY_T* const query_ptr,
                                           const INDEX_T dataset_i,
                                           const bool valid) const
  {
    const auto dataset_ptr  = ptr + dataset_i * ld;
    const unsigned lane_id  = threadIdx.x % TEAM_SIZE;
    constexpr unsigned vlen = device::get_vlen<LOAD_T, DATA_T>();
    // #include <raft/util/cuda_dev_essentials.cuh
    constexpr unsigned reg_nelem = raft::ceildiv<unsigned>(DATASET_BLOCK_DIM, TEAM_SIZE * vlen);
    raft::TxN_t<DATA_T, vlen> dl_buff[reg_nelem];

    DISTANCE_T norm2 = 0;
    if (valid) {
      for (uint32_t elem_offset = 0; elem_offset < dim; elem_offset += DATASET_BLOCK_DIM) {
#pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
          const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen + elem_offset;
          if (k >= dim) break;
          dl_buff[e].load(dataset_ptr, k);
        }
#pragma unroll
        for (uint32_t e = 0; e < reg_nelem; e++) {
          const uint32_t k = (lane_id + (TEAM_SIZE * e)) * vlen + elem_offset;
          if (k >= dim) break;
#pragma unroll
          for (uint32_t v = 0; v < vlen; v++) {
            const uint32_t kv = k + v;
            // Note this loop can go above the dataset_dim for padded arrays. This is not a problem
            // because:
            // - Above the last element (dataset_dim-1), the query array is filled with zeros.
            // - The data buffer has to be also padded with zeros.
            DISTANCE_T d = query_ptr[device::swizzling(kv)];
            norm2 += dist_op<DISTANCE_T, METRIC>(
              d, spatial::knn::detail::utils::mapping<float>{}(dl_buff[e].val.data[v]));
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

}  // namespace raft::neighbors::cagra::detail
