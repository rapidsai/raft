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

#include <vector>

#include "bitonic.hpp"
#include "compute_distance.hpp"
#include "device_common.hpp"
#include "hashmap.hpp"
#include "search_plan.cuh"
#include "topk_for_cagra/topk_core.cuh"  // TODO replace with raft topk if possible
#include "utils.hpp"
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cudart_utils.hpp>  // RAFT_CUDA_TRY_NOT_THROW is used TODO(tfeher): consider moving this to cuda_rt_essentials.hpp

namespace raft::neighbors::experimental::cagra::detail {
namespace multi_cta_search {

// #define _CLK_BREAKDOWN

template <class INDEX_T>
__device__ void pickup_next_parents(INDEX_T* const next_parent_indices,  // [num_parents]
                                    const uint32_t num_parents,
                                    INDEX_T* const itopk_indices,  // [num_itopk]
                                    const size_t num_itopk,
                                    uint32_t* const terminate_flag)
{
  const unsigned warp_id = threadIdx.x / 32;
  if (warp_id > 0) { return; }
  const unsigned lane_id = threadIdx.x % 32;
  for (uint32_t i = lane_id; i < num_parents; i += 32) {
    next_parent_indices[i] = utils::get_max_value<INDEX_T>();
  }
  uint32_t max_itopk = num_itopk;
  if (max_itopk % 32) { max_itopk += 32 - (max_itopk % 32); }
  uint32_t num_new_parents = 0;
  for (uint32_t j = lane_id; j < max_itopk; j += 32) {
    INDEX_T index;
    int new_parent = 0;
    if (j < num_itopk) {
      index = itopk_indices[j];
      if ((index & 0x80000000) == 0) {  // check if most significant bit is set
        new_parent = 1;
      }
    }
    const uint32_t ballot_mask = __ballot_sync(0xffffffff, new_parent);
    if (new_parent) {
      const auto i = __popc(ballot_mask & ((1 << lane_id) - 1)) + num_new_parents;
      if (i < num_parents) {
        next_parent_indices[i] = index;
        itopk_indices[j] |= 0x80000000;  // set most significant bit as used node
      }
    }
    num_new_parents += __popc(ballot_mask);
    if (num_new_parents >= num_parents) { break; }
  }
  if (threadIdx.x == 0 && (num_new_parents == 0)) { *terminate_flag = 1; }
}

template <unsigned MAX_ELEMENTS>
__device__ inline void topk_by_bitonic_sort(float* distances,   // [num_elements]
                                            uint32_t* indices,  // [num_elements]
                                            const uint32_t num_elements,
                                            const uint32_t num_itopk  // num_itopk <= num_elements
)
{
  const unsigned warp_id = threadIdx.x / 32;
  if (warp_id > 0) { return; }
  const unsigned lane_id = threadIdx.x % 32;
  constexpr unsigned N   = (MAX_ELEMENTS + 31) / 32;
  float key[N];
  uint32_t val[N];
  for (unsigned i = 0; i < N; i++) {
    unsigned j = lane_id + (32 * i);
    if (j < num_elements) {
      key[i] = distances[j];
      val[i] = indices[j];
    } else {
      key[i] = utils::get_max_value<float>();
      val[i] = utils::get_max_value<uint32_t>();
    }
  }
  /* Warp Sort */
  bitonic::warp_sort<float, uint32_t, N>(key, val);
  /* Store itopk sorted results */
  for (unsigned i = 0; i < N; i++) {
    unsigned j = (N * lane_id) + i;
    if (j < num_itopk) {
      distances[j] = key[i];
      indices[j]   = val[i];
    }
  }
}

//
// multiple CTAs per single query
//
template <unsigned TEAM_SIZE,
          unsigned BLOCK_SIZE,
          unsigned BLOCK_COUNT,
          unsigned MAX_ELEMENTS,
          unsigned MAX_DATASET_DIM,
          class DATA_T,
          class DISTANCE_T,
          class INDEX_T,
          class LOAD_T>
__launch_bounds__(BLOCK_SIZE, BLOCK_COUNT) __global__ void search_kernel(
  INDEX_T* const result_indices_ptr,       // [num_queries, num_cta_per_query, itopk_size]
  DISTANCE_T* const result_distances_ptr,  // [num_queries, num_cta_per_query, itopk_size]
  const DATA_T* const dataset_ptr,         // [dataset_size, dataset_dim]
  const size_t dataset_dim,
  const size_t dataset_size,
  const DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const INDEX_T* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  uint32_t* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const uint32_t hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t num_parents,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations /* stats */
)
{
  assert(blockDim.x == BLOCK_SIZE);
  assert(dataset_dim <= MAX_DATASET_DIM);

  // const auto num_queries = gridDim.y;
  const auto query_id          = blockIdx.y;
  const auto num_cta_per_query = gridDim.x;
  const auto cta_id            = blockIdx.x;  // local CTA ID

#ifdef _CLK_BREAKDOWN
  uint64_t clk_init                 = 0;
  uint64_t clk_compute_1st_distance = 0;
  uint64_t clk_topk                 = 0;
  uint64_t clk_pickup_parents       = 0;
  uint64_t clk_compute_distance     = 0;
  uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint32_t smem[];

  // Layout of result_buffer
  // +----------------+------------------------------+---------+
  // | internal_top_k | neighbors of parent nodes    | padding |
  // | <itopk_size>   | <num_parents * graph_degree> | upto 32 |
  // +----------------+------------------------------+---------+
  // |<---          result_buffer_size           --->|
  uint32_t result_buffer_size    = itopk_size + (num_parents * graph_degree);
  uint32_t result_buffer_size_32 = result_buffer_size;
  if (result_buffer_size % 32) { result_buffer_size_32 += 32 - (result_buffer_size % 32); }
  assert(result_buffer_size_32 <= MAX_ELEMENTS);

  auto query_buffer          = reinterpret_cast<float*>(smem);
  auto result_indices_buffer = reinterpret_cast<INDEX_T*>(query_buffer + MAX_DATASET_DIM);
  auto result_distances_buffer =
    reinterpret_cast<DISTANCE_T*>(result_indices_buffer + result_buffer_size_32);
  auto parent_indices_buffer =
    reinterpret_cast<uint32_t*>(result_distances_buffer + result_buffer_size_32);
  auto terminate_flag = reinterpret_cast<uint32_t*>(parent_indices_buffer + num_parents);

#if 0
    /* debug */
    for (unsigned i = threadIdx.x; i < result_buffer_size_32; i += BLOCK_SIZE) {
        result_indices_buffer[i] = utils::get_max_value<INDEX_T>();
        result_distances_buffer[i] = utils::get_max_value<DISTANCE_T>();
    }
#endif

  const DATA_T* const query_ptr = queries_ptr + (dataset_dim * query_id);
  for (unsigned i = threadIdx.x; i < MAX_DATASET_DIM; i += BLOCK_SIZE) {
    unsigned j = device::swizzling(i);
    if (i < dataset_dim) {
      query_buffer[j] = static_cast<float>(query_ptr[i]) * device::fragment_scale<DATA_T>();
    } else {
      query_buffer[j] = 0.0;
    }
  }
  if (threadIdx.x == 0) { terminate_flag[0] = 0; }
  uint32_t* local_visited_hashmap_ptr =
    visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * query_id);
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
    hash_bitlen,
    cta_id,
    num_cta_per_query);
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  uint32_t iter = 0;
  while (1) {
    // topk with bitonic sort
    _CLK_START();
    topk_by_bitonic_sort<MAX_ELEMENTS>(result_distances_buffer,
                                       result_indices_buffer,
                                       itopk_size + (num_parents * graph_degree),
                                       itopk_size);
    _CLK_REC(clk_topk);

    if (iter + 1 == max_iteration) {
      __syncthreads();
      break;
    }

    // pick up next parents
    _CLK_START();
    pickup_next_parents<INDEX_T>(
      parent_indices_buffer, num_parents, result_indices_buffer, itopk_size, terminate_flag);
    _CLK_REC(clk_pickup_parents);

    __syncthreads();
    if (*terminate_flag && iter >= min_iteration) { break; }

    // compute the norms between child nodes and query node
    _CLK_START();
    // constexpr unsigned max_n_frags = 16;
    constexpr unsigned max_n_frags = 0;
    device::
      compute_distance_to_child_nodes<TEAM_SIZE, BLOCK_SIZE, MAX_DATASET_DIM, max_n_frags, LOAD_T>(
        result_indices_buffer + itopk_size,
        result_distances_buffer + itopk_size,
        query_buffer,
        dataset_ptr,
        dataset_dim,
        knn_graph,
        graph_degree,
        local_visited_hashmap_ptr,
        hash_bitlen,
        parent_indices_buffer,
        num_parents);
    _CLK_REC(clk_compute_distance);
    __syncthreads();

    iter++;
  }

  for (uint32_t i = threadIdx.x; i < itopk_size; i += BLOCK_SIZE) {
    uint32_t j = i + (itopk_size * (cta_id + (num_cta_per_query * query_id)));
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[i]; }
    result_indices_ptr[j] = result_indices_buffer[i] & ~0x80000000;  // clear most significant bit
  }

  if (threadIdx.x == 0 && cta_id == 0 && num_executed_iterations != nullptr) {
    num_executed_iterations[query_id] = iter + 1;
  }

#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == BLOCK_SIZE - 1) && (blockIdx.x == 0) &&
      ((query_id * 3) % gridDim.y < 3)) {
    RAFT_LOG_DEBUG(
      "query, %d, thread, %d"
      ", init, %d"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", pickup_parents, %lu"
      ", distance, %lu"
      "\n",
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_pickup_parents,
      clk_compute_distance);
  }
#endif
}

#define SET_MC_KERNEL_3(BLOCK_SIZE, BLOCK_COUNT, MAX_ELEMENTS, LOAD_T) \
  kernel = search_kernel<TEAM_SIZE,                                    \
                         BLOCK_SIZE,                                   \
                         BLOCK_COUNT,                                  \
                         MAX_ELEMENTS,                                 \
                         MAX_DATASET_DIM,                              \
                         DATA_T,                                       \
                         DISTANCE_T,                                   \
                         INDEX_T,                                      \
                         LOAD_T>;

#define SET_MC_KERNEL_2(BLOCK_SIZE, BLOCK_COUNT, MAX_ELEMENTS)                    \
  if (load_bit_length == 128) {                                                   \
    SET_MC_KERNEL_3(BLOCK_SIZE, BLOCK_COUNT, MAX_ELEMENTS, device::LOAD_128BIT_T) \
  } else if (load_bit_length == 64) {                                             \
    SET_MC_KERNEL_3(BLOCK_SIZE, BLOCK_COUNT, MAX_ELEMENTS, device::LOAD_64BIT_T)  \
  }

#define SET_MC_KERNEL_1(MAX_ELEMENTS)         \
  /* if ( block_size == 32 ) {                \
      SET_MC_KERNEL_2( 32, 32, MAX_ELEMENTS ) \
  } else */                                   \
  if (block_size == 64) {                     \
    SET_MC_KERNEL_2(64, 16, MAX_ELEMENTS)     \
  } else if (block_size == 128) {             \
    SET_MC_KERNEL_2(128, 8, MAX_ELEMENTS)     \
  } else if (block_size == 256) {             \
    SET_MC_KERNEL_2(256, 4, MAX_ELEMENTS)     \
  } else if (block_size == 512) {             \
    SET_MC_KERNEL_2(512, 2, MAX_ELEMENTS)     \
  } else {                                    \
    SET_MC_KERNEL_2(1024, 1, MAX_ELEMENTS)    \
  }

#define SET_MC_KERNEL                                                       \
  typedef void (*search_kernel_t)(INDEX_T* const result_indices_ptr,        \
                                  DISTANCE_T* const result_distances_ptr,   \
                                  const DATA_T* const dataset_ptr,          \
                                  const size_t dataset_dim,                 \
                                  const size_t dataset_size,                \
                                  const DATA_T* const queries_ptr,          \
                                  const INDEX_T* const knn_graph,           \
                                  const uint32_t graph_degree,              \
                                  const unsigned num_distilation,           \
                                  const uint64_t rand_xor_mask,             \
                                  const INDEX_T* seed_ptr,                  \
                                  const uint32_t num_seeds,                 \
                                  uint32_t* const visited_hashmap_ptr,      \
                                  const uint32_t hash_bitlen,               \
                                  const uint32_t itopk_size,                \
                                  const uint32_t num_parents,               \
                                  const uint32_t min_iteration,             \
                                  const uint32_t max_iteration,             \
                                  uint32_t* const num_executed_iterations); \
  search_kernel_t kernel;                                                   \
  if (result_buffer_size <= 64) {                                           \
    SET_MC_KERNEL_1(64)                                                     \
  } else if (result_buffer_size <= 128) {                                   \
    SET_MC_KERNEL_1(128)                                                    \
  } else if (result_buffer_size <= 256) {                                   \
    SET_MC_KERNEL_1(256)                                                    \
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

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          typename DATA_T,
          typename INDEX_T,
          typename DISTANCE_T>

struct search : public search_plan_impl<DATA_T, INDEX_T, DISTANCE_T> {
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

  uint32_t num_cta_per_query;
  rmm::device_uvector<uint32_t> intermediate_indices;
  rmm::device_uvector<float> intermediate_distances;
  size_t topk_workspace_size;
  rmm::device_uvector<uint32_t> topk_workspace;

  search(raft::device_resources const& res,
         search_params params,
         int64_t dim,
         int64_t graph_degree,
         uint32_t topk)
    : search_plan_impl<DATA_T, INDEX_T, DISTANCE_T>(res, params, dim, graph_degree, topk),
      intermediate_indices(0, res.get_stream()),
      intermediate_distances(0, res.get_stream()),
      topk_workspace(0, res.get_stream())

  {
    set_params(res);
  }

  void set_params(raft::device_resources const& res)
  {
    this->itopk_size   = 32;
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
    uint32_t block_size               = thread_block_size;
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

  ~search() {}

  void operator()(raft::device_resources const& res,
                  raft::device_matrix_view<const DATA_T, INDEX_T, row_major> dataset,
                  raft::device_matrix_view<const INDEX_T, INDEX_T, row_major> graph,
                  INDEX_T* const topk_indices_ptr,       // [num_queries, topk]
                  DISTANCE_T* const topk_distances_ptr,  // [num_queries, topk]
                  const DATA_T* const queries_ptr,       // [num_queries, dataset_dim]
                  const uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,              // [num_queries, num_seeds]
                  uint32_t* const num_executed_iterations,  // [num_queries,]
                  uint32_t topk)
  {
    cudaStream_t stream = res.get_stream();
    uint32_t block_size = thread_block_size;

    SET_MC_KERNEL;
    RAFT_CUDA_TRY(
      cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    // Initialize hash table
    const uint32_t hash_size = hashmap::get_size(hash_bitlen);
    set_value_batch(
      hashmap.data(), hash_size, utils::get_max_value<uint32_t>(), hash_size, num_queries, stream);

    dim3 block_dims(block_size, 1, 1);
    dim3 grid_dims(num_cta_per_query, num_queries, 1);
    RAFT_LOG_DEBUG("Launching kernel with %u threads, (%u, %u) blocks %lu smem",
                   block_size,
                   num_cta_per_query,
                   num_queries,
                   smem_size);
    kernel<<<grid_dims, block_dims, smem_size, stream>>>(intermediate_indices.data(),
                                                         intermediate_distances.data(),
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
                                                         hash_bitlen,
                                                         itopk_size,
                                                         num_parents,
                                                         min_iterations,
                                                         max_iterations,
                                                         num_executed_iterations);
    RAFT_CUDA_TRY(cudaPeekAtLastError());

    // Select the top-k results from the intermediate results
    const uint32_t num_intermediate_results = num_cta_per_query * itopk_size;
    _cuann_find_topk(topk,
                     num_queries,
                     num_intermediate_results,
                     intermediate_distances.data(),
                     num_intermediate_results,
                     intermediate_indices.data(),
                     num_intermediate_results,
                     topk_distances_ptr,
                     topk,
                     topk_indices_ptr,
                     topk,
                     topk_workspace.data(),
                     true,
                     NULL,
                     stream);
  }
};

}  // namespace multi_cta_search
}  // namespace raft::neighbors::experimental::cagra::detail
