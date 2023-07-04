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

#include <raft/util/raft_explicit.hpp>  // RAFT_EXPLICIT
namespace raft::neighbors::experimental::cagra::detail {
namespace single_cta_search {

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

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
                     const DATA_T* const dataset_ptr,         // [dataset_size, dataset_dim]
                     const std::size_t dataset_dim,
                     const std::size_t dataset_size,
                     const std::size_t dataset_ld,
                     const DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
                     const INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
                     const std::uint32_t graph_degree,
                     const unsigned num_distilation,
                     const uint64_t rand_xor_mask,
                     const INDEX_T* seed_ptr,             // [num_queries, num_seeds]
                     const uint32_t num_seeds,
                     INDEX_T* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
                     const std::uint32_t internal_topk,
                     const std::uint32_t num_parents,
                     const std::uint32_t min_iteration,
                     const std::uint32_t max_iteration,
                     std::uint32_t* const num_executed_iterations,  // [num_queries]
                     const std::uint32_t hash_bitlen,
                     const std::uint32_t small_hash_bitlen,
                     const std::uint32_t small_hash_reset_interval) RAFT_EXPLICIT;

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_single_cta_search_kernel(TEAM_SIZE,               \
                                             BLOCK_SIZE,              \
                                             BLOCK_COUNT,             \
                                             MAX_ITOPK,               \
                                             MAX_CANDIDATES,          \
                                             TOPK_BY_BITONIC_SORT,    \
                                             MAX_DATASET_DIM,         \
                                             DATA_T,                  \
                                             DISTANCE_T,              \
                                             INDEX_T,                 \
                                             LOAD_T)                  \
  extern template __global__ void search_kernel<TEAM_SIZE,            \
                                                BLOCK_SIZE,           \
                                                BLOCK_COUNT,          \
                                                MAX_ITOPK,            \
                                                MAX_CANDIDATES,       \
                                                TOPK_BY_BITONIC_SORT, \
                                                MAX_DATASET_DIM,      \
                                                DATA_T,               \
                                                DISTANCE_T,           \
                                                INDEX_T,              \
                                                LOAD_T>(              \
    INDEX_T* const result_indices_ptr,                                \
    DISTANCE_T* const result_distances_ptr,                           \
    const std::uint32_t top_k,                                        \
    const DATA_T* const dataset_ptr,                                  \
    const std::size_t dataset_dim,                                    \
    const std::size_t dataset_size,                                   \
    const std::size_t dataset_ld,                                     \
    const DATA_T* const queries_ptr,                                  \
    const INDEX_T* const knn_graph,                                   \
    const std::uint32_t graph_degree,                                 \
    const unsigned num_distilation,                                   \
    const uint64_t rand_xor_mask,                                     \
    const INDEX_T* seed_ptr,                                          \
    const uint32_t num_seeds,                                         \
    INDEX_T* const visited_hashmap_ptr,                               \
    const std::uint32_t internal_topk,                                \
    const std::uint32_t num_parents,                                  \
    const std::uint32_t min_iteration,                                \
    const std::uint32_t max_iteration,                                \
    std::uint32_t* const num_executed_iterations,                     \
    const std::uint32_t hash_bitlen,                                  \
    const std::uint32_t small_hash_bitlen,                            \
    const std::uint32_t small_hash_reset_interval);

// search_single_cta_float_uint32_dim1024_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 64, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 128, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 256, 1, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 32, 0, 1024, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 32, 0, 1024, float, float, uint32_t, uint4);

// search_single_cta_float_uint32_dim128_t8.cu
instantiate_single_cta_search_kernel(8, 64, 16, 64, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 64, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 128, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 256, 1, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 32, 0, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 32, 0, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 32, 0, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 32, 0, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 32, 0, 128, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 32, 0, 128, float, float, uint32_t, uint4);

// search_single_cta_float_uint32_dim256_t16.cu
instantiate_single_cta_search_kernel(16, 64, 16, 64, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 64, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 128, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 256, 1, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 32, 0, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 32, 0, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 32, 0, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 32, 0, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 32, 0, 256, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 32, 0, 256, float, float, uint32_t, uint4);

// search_single_cta_float_uint32_dim512_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 64, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 128, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 256, 1, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 32, 0, 512, float, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 32, 0, 512, float, float, uint32_t, uint4);

// search_single_cta_int8_uint32_dim1024_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 64, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 128, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 128, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 128, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 256, 1, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 32, 0, 1024, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 32, 0, 1024, int8_t, float, uint32_t, uint4);

// search_single_cta_int8_uint32_dim128_t8.cu
instantiate_single_cta_search_kernel(8, 64, 16, 64, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 64, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 128, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 256, 1, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 32, 0, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 32, 0, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 32, 0, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 32, 0, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 32, 0, 128, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 32, 0, 128, int8_t, float, uint32_t, uint4);

// search_single_cta_int8_uint32_dim256_t16.cu
instantiate_single_cta_search_kernel(16, 64, 16, 64, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 64, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 128, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 256, 1, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 32, 0, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 32, 0, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 32, 0, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 32, 0, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 32, 0, 256, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 32, 0, 256, int8_t, float, uint32_t, uint4);

// search_single_cta_int8_uint32_dim512_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 64, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 128, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 256, 1, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 32, 0, 512, int8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 32, 0, 512, int8_t, float, uint32_t, uint4);

// search_single_cta_uint8_uint32_dim1024_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 64, 16, 128, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 64, 16, 256, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 64, 16, 512, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 64, 16, 128, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 64, 16, 256, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 64, 16, 512, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 128, 8, 128, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 128, 8, 256, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 128, 8, 512, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 128, 8, 128, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 128, 8, 256, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 128, 8, 512, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 256, 4, 128, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 256, 4, 256, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 256, 4, 512, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 256, 4, 128, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 256, 4, 256, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 256, 4, 512, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 512, 2, 128, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 512, 2, 256, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 512, 2, 512, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 512, 2, 128, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 512, 2, 256, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 512, 2, 512, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 128, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 64, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 64, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 128, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 128, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 64, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 128, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 256, 1, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 32, 0, 1024, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 32, 0, 1024, uint8_t, float, uint32_t, uint4);

// search_single_cta_uint8_uint32_dim128_t8.cu
instantiate_single_cta_search_kernel(8, 64, 16, 64, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 64, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 128, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 256, 1, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 32, 0, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 32, 0, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 32, 0, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 32, 0, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 32, 0, 128, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 32, 0, 128, uint8_t, float, uint32_t, uint4);

// search_single_cta_uint8_uint32_dim256_t16.cu
instantiate_single_cta_search_kernel(16, 64, 16, 64, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 64, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  16, 1024, 1, 128, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  16, 1024, 1, 256, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  16, 1024, 1, 512, 128, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  16, 1024, 1, 128, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  16, 1024, 1, 256, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  16, 1024, 1, 512, 256, 1, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 32, 0, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 32, 0, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 32, 0, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 32, 0, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 32, 0, 256, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 32, 0, 256, uint8_t, float, uint32_t, uint4);

// search_single_cta_uint8_uint32_dim512_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 64, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 128, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 128, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 128, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 256, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(
  32, 1024, 1, 512, 256, 1, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 32, 0, 512, uint8_t, float, uint32_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 32, 0, 512, uint8_t, float, uint32_t, uint4);

// search_single_cta_float_uint64_dim1024_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 64, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 128, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 256, 1, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 32, 0, 1024, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 32, 0, 1024, float, float, uint64_t, uint4);

// search_single_cta_float_uint64_dim128_t8.cu
instantiate_single_cta_search_kernel(8, 64, 16, 64, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 64, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 128, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 256, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 64, 16, 512, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 64, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 128, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 256, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 128, 8, 512, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 64, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 128, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 64, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 128, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 64, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 128, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 64, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 128, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 256, 1, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 256, 32, 0, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 256, 4, 512, 32, 0, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 256, 32, 0, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 512, 2, 512, 32, 0, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 256, 32, 0, 128, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(8, 1024, 1, 512, 32, 0, 128, float, float, uint64_t, uint4);

// search_single_cta_float_uint64_dim256_t16.cu
instantiate_single_cta_search_kernel(16, 64, 16, 64, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 64, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 128, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 256, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 64, 16, 512, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 64, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 128, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 256, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 128, 8, 512, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 64, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 128, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 64, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 128, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 64, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 128, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 64, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 128, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 256, 1, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 256, 32, 0, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 256, 4, 512, 32, 0, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 256, 32, 0, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 512, 2, 512, 32, 0, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 256, 32, 0, 256, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(16, 1024, 1, 512, 32, 0, 256, float, float, uint64_t, uint4);

// search_single_cta_float_uint64_dim512_t32.cu
instantiate_single_cta_search_kernel(32, 64, 16, 64, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 64, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 128, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 256, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 64, 16, 512, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 64, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 128, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 256, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 128, 8, 512, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 64, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 128, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 64, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 128, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 64, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 128, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 64, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 128, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 256, 1, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 256, 32, 0, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 256, 4, 512, 32, 0, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 256, 32, 0, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 512, 2, 512, 32, 0, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 256, 32, 0, 512, float, float, uint64_t, uint4);
instantiate_single_cta_search_kernel(32, 1024, 1, 512, 32, 0, 512, float, float, uint64_t, uint4);

#undef instantiate_single_cta_search_kernel

}  // namespace single_cta_search
}  // namespace raft::neighbors::experimental::cagra::detail
