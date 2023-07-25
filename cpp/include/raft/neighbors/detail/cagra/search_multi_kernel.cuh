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

#include <algorithm>
#include <cassert>
#include <iostream>
#include <memory>
#include <numeric>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

#include "compute_distance.hpp"
#include "device_common.hpp"
#include "fragment.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_for_cagra/topk_core.cuh"  //todo replace with raft kernel
#include "utils.hpp"
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

namespace raft::neighbors::cagra::detail {
namespace multi_kernel_search {

template <class T>
__global__ void set_value_kernel(T* const dev_ptr, const T val)
{
  *dev_ptr = val;
}

template <class T>
__global__ void set_value_kernel(T* const dev_ptr, const T val, const std::size_t count)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count) { return; }
  dev_ptr[tid] = val;
}

template <class T>
void set_value(T* const dev_ptr, const T val, cudaStream_t cuda_stream)
{
  set_value_kernel<T><<<1, 1, 0, cuda_stream>>>(dev_ptr, val);
}

template <class T>
void set_value(T* const dev_ptr, const T val, const std::size_t count, cudaStream_t cuda_stream)
{
  constexpr std::uint32_t block_size = 256;
  const auto grid_size               = (count + block_size - 1) / block_size;
  set_value_kernel<T><<<grid_size, block_size, 0, cuda_stream>>>(dev_ptr, val, count);
}

template <class T>
__global__ void get_value_kernel(T* const host_ptr, const T* const dev_ptr)
{
  *host_ptr = *dev_ptr;
}

template <class T>
void get_value(T* const host_ptr, const T* const dev_ptr, cudaStream_t cuda_stream)
{
  get_value_kernel<T><<<1, 1, 0, cuda_stream>>>(host_ptr, dev_ptr);
}

// MAX_DATASET_DIM : must equal to or greater than dataset_dim
template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          class DATA_T,
          class DISTANCE_T,
          class INDEX_T>
__global__ void random_pickup_kernel(
  const DATA_T* const dataset_ptr,  // [dataset_size, dataset_dim]
  const std::size_t dataset_dim,
  const std::size_t dataset_size,
  const std::size_t dataset_ld,
  const DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const std::size_t num_pickup,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const INDEX_T* seed_ptr,                 // [num_queries, num_seeds]
  const uint32_t num_seeds,
  INDEX_T* const result_indices_ptr,       // [num_queries, ldr]
  DISTANCE_T* const result_distances_ptr,  // [num_queries, ldr]
  const std::uint32_t ldr,                 // (*) ldr >= num_pickup
  INDEX_T* const visited_hashmap_ptr,      // [num_queries, 1 << bitlen]
  const std::uint32_t hash_bitlen)
{
  const auto ldb               = hashmap::get_size(hash_bitlen);
  const auto global_team_index = (blockIdx.x * blockDim.x + threadIdx.x) / TEAM_SIZE;
  const uint32_t query_id      = blockIdx.y;
  if (global_team_index >= num_pickup) { return; }
  // Load a query
  device::fragment<MAX_DATASET_DIM, DATA_T, TEAM_SIZE> query_frag;
  device::load_vector_sync(query_frag, queries_ptr + query_id * dataset_dim, dataset_dim);

  INDEX_T best_index_team_local;
  DISTANCE_T best_norm2_team_local = utils::get_max_value<DISTANCE_T>();
  for (unsigned i = 0; i < num_distilation; i++) {
    INDEX_T seed_index;
    if (seed_ptr && (global_team_index < num_seeds)) {
      seed_index = seed_ptr[global_team_index + (num_seeds * query_id)];
    } else {
      // Chose a seed node randomly
      seed_index = device::xorshift64((global_team_index ^ rand_xor_mask) * (i + 1)) % dataset_size;
    }
    device::fragment<MAX_DATASET_DIM, DATA_T, TEAM_SIZE> random_data_frag;
    device::load_vector_sync(
      random_data_frag, dataset_ptr + (dataset_ld * seed_index), dataset_dim);

    // Compute the norm of two data
    const auto norm2 = device::norm2<DISTANCE_T>(
      query_frag,
      random_data_frag,
      static_cast<float>(1.0 / spatial::knn::detail::utils::config<DATA_T>::kDivisor)
      /*, scale*/
    );

    if (norm2 < best_norm2_team_local) {
      best_norm2_team_local = norm2;
      best_index_team_local = seed_index;
    }
  }

  const auto store_gmem_index = global_team_index + (ldr * query_id);
  if (threadIdx.x % TEAM_SIZE == 0) {
    if (hashmap::insert(
          visited_hashmap_ptr + (ldb * query_id), hash_bitlen, best_index_team_local)) {
      result_distances_ptr[store_gmem_index] = best_norm2_team_local;
      result_indices_ptr[store_gmem_index]   = best_index_team_local;
    } else {
      result_distances_ptr[store_gmem_index] = utils::get_max_value<DISTANCE_T>();
      result_indices_ptr[store_gmem_index]   = utils::get_max_value<INDEX_T>();
    }
  }
}

// MAX_DATASET_DIM : must be equal to or greater than dataset_dim
template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          class DATA_T,
          class DISTANCE_T,
          class INDEX_T>
void random_pickup(const DATA_T* const dataset_ptr,  // [dataset_size, dataset_dim]
                   const std::size_t dataset_dim,
                   const std::size_t dataset_size,
                   const std::size_t dataset_ld,
                   const DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
                   const std::size_t num_queries,
                   const std::size_t num_pickup,
                   const unsigned num_distilation,
                   const uint64_t rand_xor_mask,
                   const INDEX_T* seed_ptr,                 // [num_queries, num_seeds]
                   const uint32_t num_seeds,
                   INDEX_T* const result_indices_ptr,       // [num_queries, ldr]
                   DISTANCE_T* const result_distances_ptr,  // [num_queries, ldr]
                   const std::size_t ldr,                   // (*) ldr >= num_pickup
                   INDEX_T* const visited_hashmap_ptr,      // [num_queries, 1 << bitlen]
                   const std::uint32_t hash_bitlen,
                   cudaStream_t const cuda_stream = 0)
{
  const auto block_size                = 256u;
  const auto num_teams_per_threadblock = block_size / TEAM_SIZE;
  const dim3 grid_size((num_pickup + num_teams_per_threadblock - 1) / num_teams_per_threadblock,
                       num_queries);

  random_pickup_kernel<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>
    <<<grid_size, block_size, 0, cuda_stream>>>(dataset_ptr,
                                                dataset_dim,
                                                dataset_size,
                                                dataset_ld,
                                                queries_ptr,
                                                num_pickup,
                                                num_distilation,
                                                rand_xor_mask,
                                                seed_ptr,
                                                num_seeds,
                                                result_indices_ptr,
                                                result_distances_ptr,
                                                ldr,
                                                visited_hashmap_ptr,
                                                hash_bitlen);
}

template <class INDEX_T>
__global__ void pickup_next_parents_kernel(
  INDEX_T* const parent_candidates_ptr,        // [num_queries, lds]
  const std::size_t lds,                       // (*) lds >= parent_candidates_size
  const std::uint32_t parent_candidates_size,  //
  INDEX_T* const visited_hashmap_ptr,          // [num_queries, 1 << hash_bitlen]
  const std::size_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  INDEX_T* const parent_list_ptr,      // [num_queries, ldd]
  const std::size_t ldd,               // (*) ldd >= parent_list_size
  const std::size_t parent_list_size,  //
  std::uint32_t* const terminate_flag)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  const std::size_t ldb   = hashmap::get_size(hash_bitlen);
  const uint32_t query_id = blockIdx.x;
  if (threadIdx.x < 32) {
    // pickup next parents with single warp
    for (std::uint32_t i = threadIdx.x; i < parent_list_size; i += 32) {
      parent_list_ptr[i + (ldd * query_id)] = utils::get_max_value<INDEX_T>();
    }
    std::uint32_t parent_candidates_size_max = parent_candidates_size;
    if (parent_candidates_size % 32) {
      parent_candidates_size_max += 32 - (parent_candidates_size % 32);
    }
    std::uint32_t num_new_parents = 0;
    for (std::uint32_t j = threadIdx.x; j < parent_candidates_size_max; j += 32) {
      INDEX_T index;
      int new_parent = 0;
      if (j < parent_candidates_size) {
        index = parent_candidates_ptr[j + (lds * query_id)];
        if ((index & index_msb_1_mask) == 0) {  // check most significant bit
          new_parent = 1;
        }
      }
      const std::uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
      if (new_parent) {
        const auto i = __popc(ballot_mask & ((1 << threadIdx.x) - 1)) + num_new_parents;
        if (i < parent_list_size) {
          parent_list_ptr[i + (ldd * query_id)] = index;
          parent_candidates_ptr[j + (lds * query_id)] |=
            index_msb_1_mask;  // set most significant bit as used node
        }
      }
      num_new_parents += __popc(ballot_mask);
      if (num_new_parents >= parent_list_size) { break; }
    }
    if ((num_new_parents > 0) && (threadIdx.x == 0)) { *terminate_flag = 0; }
  } else if (small_hash_bitlen) {
    // reset small-hash
    hashmap::init<32>(visited_hashmap_ptr + (ldb * query_id), hash_bitlen);
  }

  if (small_hash_bitlen) {
    __syncthreads();
    // insert internal-topk indices into small-hash
    for (unsigned i = threadIdx.x; i < parent_candidates_size; i += blockDim.x) {
      auto key = parent_candidates_ptr[i + (lds * query_id)] &
                 ~index_msb_1_mask;  // clear most significant bit
      hashmap::insert(visited_hashmap_ptr + (ldb * query_id), hash_bitlen, key);
    }
  }
}

template <class INDEX_T>
void pickup_next_parents(INDEX_T* const parent_candidates_ptr,  // [num_queries, lds]
                         const std::size_t lds,                 // (*) lds >= parent_candidates_size
                         const std::size_t parent_candidates_size,  //
                         const std::size_t num_queries,
                         INDEX_T* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
                         const std::size_t hash_bitlen,
                         const std::size_t small_hash_bitlen,
                         INDEX_T* const parent_list_ptr,      // [num_queries, ldd]
                         const std::size_t ldd,               // (*) ldd >= parent_list_size
                         const std::size_t parent_list_size,  //
                         std::uint32_t* const terminate_flag,
                         cudaStream_t cuda_stream = 0)
{
  std::uint32_t block_size = 32;
  if (small_hash_bitlen) {
    block_size = 128;
    while (parent_candidates_size > block_size) {
      block_size *= 2;
    }
    block_size = min(block_size, (uint32_t)512);
  }
  pickup_next_parents_kernel<INDEX_T>
    <<<num_queries, block_size, 0, cuda_stream>>>(parent_candidates_ptr,
                                                  lds,
                                                  parent_candidates_size,
                                                  visited_hashmap_ptr,
                                                  hash_bitlen,
                                                  small_hash_bitlen,
                                                  parent_list_ptr,
                                                  ldd,
                                                  parent_list_size,
                                                  terminate_flag);
}

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          class DATA_T,
          class INDEX_T,
          class DISTANCE_T>
__global__ void compute_distance_to_child_nodes_kernel(
  const INDEX_T* const parent_node_list,  // [num_queries, num_parents]
  const std::uint32_t num_parents,
  const DATA_T* const dataset_ptr,        // [dataset_size, data_dim]
  const std::uint32_t data_dim,
  const std::uint32_t dataset_size,
  const std::uint32_t dataset_ld,
  const INDEX_T* const neighbor_graph_ptr,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const DATA_T* query_ptr,                  // [num_queries, data_dim]
  INDEX_T* const visited_hashmap_ptr,       // [num_queries, 1 << hash_bitlen]
  const std::uint32_t hash_bitlen,
  INDEX_T* const result_indices_ptr,        // [num_queries, ldd]
  DISTANCE_T* const result_distances_ptr,   // [num_queries, ldd]
  const std::uint32_t ldd                   // (*) ldd >= num_parents * graph_degree
)
{
  const uint32_t ldb        = hashmap::get_size(hash_bitlen);
  const auto tid            = threadIdx.x + blockDim.x * blockIdx.x;
  const auto global_team_id = tid / TEAM_SIZE;
  if (global_team_id >= num_parents * graph_degree) { return; }

  const std::size_t parent_index =
    parent_node_list[global_team_id / graph_degree + (num_parents * blockIdx.y)];
  if (parent_index == utils::get_max_value<INDEX_T>()) {
    result_distances_ptr[ldd * blockIdx.y + global_team_id] = utils::get_max_value<DISTANCE_T>();
    return;
  }
  const auto neighbor_list_head_ptr = neighbor_graph_ptr + (graph_degree * parent_index);

  const std::size_t child_id = neighbor_list_head_ptr[global_team_id % graph_degree];

  if (hashmap::insert<TEAM_SIZE, INDEX_T>(
        visited_hashmap_ptr + (ldb * blockIdx.y), hash_bitlen, child_id)) {
    device::fragment<MAX_DATASET_DIM, DATA_T, TEAM_SIZE> frag_target;
    device::load_vector_sync(frag_target, dataset_ptr + (dataset_ld * child_id), data_dim);

    device::fragment<MAX_DATASET_DIM, DATA_T, TEAM_SIZE> frag_query;
    device::load_vector_sync(frag_query, query_ptr + blockIdx.y * data_dim, data_dim);

    const auto norm2 = device::norm2<DISTANCE_T>(
      frag_target,
      frag_query,
      static_cast<float>(1.0 / spatial::knn::detail::utils::config<DATA_T>::kDivisor));

    if (threadIdx.x % TEAM_SIZE == 0) {
      result_indices_ptr[ldd * blockIdx.y + global_team_id]   = child_id;
      result_distances_ptr[ldd * blockIdx.y + global_team_id] = norm2;
    }
  } else {
    if (threadIdx.x % TEAM_SIZE == 0) {
      result_distances_ptr[ldd * blockIdx.y + global_team_id] = utils::get_max_value<DISTANCE_T>();
    }
  }
}

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          class DATA_T,
          class INDEX_T,
          class DISTANCE_T>
void compute_distance_to_child_nodes(
  const INDEX_T* const parent_node_list,  // [num_queries, num_parents]
  const uint32_t num_parents,
  const DATA_T* const dataset_ptr,        // [dataset_size, data_dim]
  const std::uint32_t data_dim,
  const std::uint32_t dataset_size,
  const std::uint32_t dataset_ld,
  const INDEX_T* const neighbor_graph_ptr,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const DATA_T* query_ptr,                  // [num_queries, data_dim]
  const std::uint32_t num_queries,
  INDEX_T* const visited_hashmap_ptr,       // [num_queries, 1 << hash_bitlen]
  const std::uint32_t hash_bitlen,
  INDEX_T* const result_indices_ptr,        // [num_queries, ldd]
  DISTANCE_T* const result_distances_ptr,   // [num_queries, ldd]
  const std::uint32_t ldd,                  // (*) ldd >= num_parents * graph_degree
  cudaStream_t cuda_stream = 0)
{
  const auto block_size = 128;
  const dim3 grid_size(
    (num_parents * graph_degree + (block_size / TEAM_SIZE) - 1) / (block_size / TEAM_SIZE),
    num_queries);
  compute_distance_to_child_nodes_kernel<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, INDEX_T, DISTANCE_T>
    <<<grid_size, block_size, 0, cuda_stream>>>(parent_node_list,
                                                num_parents,
                                                dataset_ptr,
                                                data_dim,
                                                dataset_size,
                                                dataset_ld,
                                                neighbor_graph_ptr,
                                                graph_degree,
                                                query_ptr,
                                                visited_hashmap_ptr,
                                                hash_bitlen,
                                                result_indices_ptr,
                                                result_distances_ptr,
                                                ldd);
}

template <class INDEX_T>
__global__ void remove_parent_bit_kernel(const std::uint32_t num_queries,
                                         const std::uint32_t num_topk,
                                         INDEX_T* const topk_indices_ptr,  // [ld, num_queries]
                                         const std::uint32_t ld)
{
  constexpr INDEX_T index_msb_1_mask = utils::gen_index_msb_1_mask<INDEX_T>::value;

  uint32_t i_query = blockIdx.x;
  if (i_query >= num_queries) return;

  for (unsigned i = threadIdx.x; i < num_topk; i += blockDim.x) {
    topk_indices_ptr[i + (ld * i_query)] &= ~index_msb_1_mask;  // clear most significant bit
  }
}

template <class INDEX_T>
void remove_parent_bit(const std::uint32_t num_queries,
                       const std::uint32_t num_topk,
                       INDEX_T* const topk_indices_ptr,  // [ld, num_queries]
                       const std::uint32_t ld,
                       cudaStream_t cuda_stream = 0)
{
  const std::size_t grid_size  = num_queries;
  const std::size_t block_size = 256;
  remove_parent_bit_kernel<<<grid_size, block_size, 0, cuda_stream>>>(
    num_queries, num_topk, topk_indices_ptr, ld);
}

template <class T>
__global__ void batched_memcpy_kernel(T* const dst,        // [batch_size, ld_dst]
                                      const uint64_t ld_dst,
                                      const T* const src,  // [batch_size, ld_src]
                                      const uint64_t ld_src,
                                      const uint64_t count,
                                      const uint64_t batch_size)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count * batch_size) { return; }
  const auto i          = tid % count;
  const auto j          = tid / count;
  dst[i + (ld_dst * j)] = src[i + (ld_src * j)];
}

template <class T>
void batched_memcpy(T* const dst,        // [batch_size, ld_dst]
                    const uint64_t ld_dst,
                    const T* const src,  // [batch_size, ld_src]
                    const uint64_t ld_src,
                    const uint64_t count,
                    const uint64_t batch_size,
                    cudaStream_t cuda_stream)
{
  assert(ld_dst >= count);
  assert(ld_src >= count);
  constexpr uint32_t block_size = 256;
  const auto grid_size          = (batch_size * count + block_size - 1) / block_size;
  batched_memcpy_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(dst, ld_dst, src, ld_src, count, batch_size);
}

template <class T>
__global__ void set_value_batch_kernel(T* const dev_ptr,
                                       const std::size_t ld,
                                       const T val,
                                       const std::size_t count,
                                       const std::size_t batch_size)
{
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count * batch_size) { return; }
  const auto batch_id              = tid / count;
  const auto elem_id               = tid % count;
  dev_ptr[elem_id + ld * batch_id] = val;
}

template <class T>
void set_value_batch(T* const dev_ptr,
                     const std::size_t ld,
                     const T val,
                     const std::size_t count,
                     const std::size_t batch_size,
                     cudaStream_t cuda_stream)
{
  constexpr std::uint32_t block_size = 256;
  const auto grid_size               = (count * batch_size + block_size - 1) / block_size;
  set_value_batch_kernel<T>
    <<<grid_size, block_size, 0, cuda_stream>>>(dev_ptr, ld, val, count, batch_size);
}

// result_buffer (work buffer) for "multi-kernel"
// +--------------------+------------------------------+-------------------+
// | internal_top_k (A) | neighbors of internal_top_k  | internal_topk (B) |
// | <itopk_size>       | <num_parents * graph_degree> | <itopk_size>      |
// +--------------------+------------------------------+-------------------+
// |<---                 result_buffer_allocation_size                 --->|
// |<---                       result_buffer_size  --->|                     // Double buffer (A)
//                      |<---  result_buffer_size                      --->| // Double buffer (B)
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

  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::hashmap;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_executed_iterations;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::dev_seed;
  using search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>::num_seeds;

  size_t result_buffer_allocation_size;
  rmm::device_uvector<INDEX_T> result_indices;  // results_indices_buffer
  rmm::device_uvector<float> result_distances;  // result_distances_buffer
  rmm::device_uvector<INDEX_T> parent_node_list;
  rmm::device_uvector<uint32_t> topk_hint;
  rmm::device_scalar<uint32_t> terminate_flag;  // dev_terminate_flag, host_terminate_flag.;
  rmm::device_uvector<uint32_t> topk_workspace;

  search(raft::resources const& res,
         search_params params,
         int64_t dim,
         int64_t graph_degree,
         uint32_t topk)
    : search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>(res, params, dim, graph_degree, topk),
      result_indices(0, resource::get_cuda_stream(res)),
      result_distances(0, resource::get_cuda_stream(res)),
      parent_node_list(0, resource::get_cuda_stream(res)),
      topk_hint(0, resource::get_cuda_stream(res)),
      topk_workspace(0, resource::get_cuda_stream(res)),
      terminate_flag(resource::get_cuda_stream(res))
  {
    set_params(res);
  }

  void set_params(raft::resources const& res)
  {
    //
    // Allocate memory for intermediate buffer and workspace.
    //
    result_buffer_size            = itopk_size + (num_parents * graph_degree);
    result_buffer_allocation_size = result_buffer_size + itopk_size;
    result_indices.resize(result_buffer_allocation_size * max_queries,
                          resource::get_cuda_stream(res));
    result_distances.resize(result_buffer_allocation_size * max_queries,
                            resource::get_cuda_stream(res));

    parent_node_list.resize(max_queries * num_parents, resource::get_cuda_stream(res));
    topk_hint.resize(max_queries, resource::get_cuda_stream(res));

    size_t topk_workspace_size = _cuann_find_topk_bufferSize(
      itopk_size, max_queries, result_buffer_size, utils::get_cuda_data_type<DATA_T>());
    RAFT_LOG_DEBUG("# topk_workspace_size: %lu", topk_workspace_size);
    topk_workspace.resize(topk_workspace_size, resource::get_cuda_stream(res));

    hashmap.resize(hashmap_size, resource::get_cuda_stream(res));
  }

  ~search() {}

  void operator()(raft::resources const& res,
                  raft::device_matrix_view<const DATA_T, INDEX_T, layout_stride> dataset,
                  raft::device_matrix_view<const INDEX_T, INDEX_T, row_major> graph,
                  INDEX_T* const topk_indices_ptr,          // [num_queries, topk]
                  DISTANCE_T* const topk_distances_ptr,     // [num_queries, topk]
                  const DATA_T* const queries_ptr,          // [num_queries, dataset_dim]
                  const uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,              // [num_queries, num_seeds]
                  uint32_t* const num_executed_iterations,  // [num_queries,]
                  uint32_t topk)
  {
    // Init hashmap
    cudaStream_t stream      = resource::get_cuda_stream(res);
    const uint32_t hash_size = hashmap::get_size(hash_bitlen);
    set_value_batch(
      hashmap.data(), hash_size, utils::get_max_value<INDEX_T>(), hash_size, num_queries, stream);
    // Init topk_hint
    if (topk_hint.size() > 0) { set_value(topk_hint.data(), 0xffffffffu, num_queries, stream); }

    // Choose initial entry point candidates at random
    random_pickup<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, DISTANCE_T, INDEX_T>(
      dataset.data_handle(),
      dataset.extent(1),
      dataset.extent(0),
      dataset.stride(0),
      queries_ptr,
      num_queries,
      result_buffer_size,
      num_random_samplings,
      rand_xor_mask,
      dev_seed_ptr,
      num_seeds,
      result_indices.data(),
      result_distances.data(),
      result_buffer_allocation_size,
      hashmap.data(),
      hash_bitlen,
      stream);

    unsigned iter = 0;
    while (1) {
      // Make an index list of internal top-k nodes
      _cuann_find_topk(itopk_size,
                       num_queries,
                       result_buffer_size,
                       result_distances.data() + (iter & 0x1) * itopk_size,
                       result_buffer_allocation_size,
                       result_indices.data() + (iter & 0x1) * itopk_size,
                       result_buffer_allocation_size,
                       result_distances.data() + (1 - (iter & 0x1)) * result_buffer_size,
                       result_buffer_allocation_size,
                       result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
                       result_buffer_allocation_size,
                       topk_workspace.data(),
                       true,
                       topk_hint.data(),
                       stream);

      // termination (1)
      if ((iter + 1 == max_iterations)) {
        iter++;
        break;
      }

      if (iter + 1 >= min_iterations) { set_value<uint32_t>(terminate_flag.data(), 1, stream); }

      // pickup parent nodes
      uint32_t _small_hash_bitlen = 0;
      if ((iter + 1) % small_hash_reset_interval == 0) { _small_hash_bitlen = small_hash_bitlen; }
      pickup_next_parents(result_indices.data() + (1 - (iter & 0x1)) * result_buffer_size,
                          result_buffer_allocation_size,
                          itopk_size,
                          num_queries,
                          hashmap.data(),
                          hash_bitlen,
                          _small_hash_bitlen,
                          parent_node_list.data(),
                          num_parents,
                          num_parents,
                          terminate_flag.data(),
                          stream);

      // termination (2)
      if (iter + 1 >= min_iterations && terminate_flag.value(stream)) {
        iter++;
        break;
      }

      // Compute distance to child nodes that are adjacent to the parent node
      compute_distance_to_child_nodes<TEAM_SIZE, MAX_DATASET_DIM>(
        parent_node_list.data(),
        num_parents,
        dataset.data_handle(),
        dataset.extent(1),
        dataset.extent(0),
        dataset.stride(0),
        graph.data_handle(),
        graph.extent(1),
        queries_ptr,
        num_queries,
        hashmap.data(),
        hash_bitlen,
        result_indices.data() + itopk_size,
        result_distances.data() + itopk_size,
        result_buffer_allocation_size,
        stream);

      iter++;
    }  // while ( 1 )

    // Remove parent bit in search results
    remove_parent_bit(num_queries,
                      itopk_size,
                      result_indices.data() + (iter & 0x1) * result_buffer_size,
                      result_buffer_allocation_size,
                      stream);

    // Copy results from working buffer to final buffer
    batched_memcpy(topk_indices_ptr,
                   topk,
                   result_indices.data() + (iter & 0x1) * result_buffer_size,
                   result_buffer_allocation_size,
                   topk,
                   num_queries,
                   stream);
    if (topk_distances_ptr) {
      batched_memcpy(topk_distances_ptr,
                     topk,
                     result_distances.data() + (iter & 0x1) * result_buffer_size,
                     result_buffer_allocation_size,
                     topk,
                     num_queries,
                     stream);
    }

    if (num_executed_iterations) {
      for (std::uint32_t i = 0; i < num_queries; i++) {
        num_executed_iterations[i] = iter;
      }
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
};

}  // namespace multi_kernel_search
}  // namespace raft::neighbors::cagra::detail
