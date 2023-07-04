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
namespace multi_cta_search {

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

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
  const size_t dataset_ld,
  const DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const INDEX_T* const knn_graph,   // [dataset_size, graph_degree]
  const uint32_t graph_degree,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const INDEX_T* seed_ptr,             // [num_queries, num_seeds]
  const uint32_t num_seeds,
  INDEX_T* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const uint32_t hash_bitlen,
  const uint32_t itopk_size,
  const uint32_t num_parents,
  const uint32_t min_iteration,
  const uint32_t max_iteration,
  uint32_t* const num_executed_iterations /* stats */
  ) RAFT_EXPLICIT;

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_multi_cta_search_kernel(TEAM_SIZE,                                          \
                                            BLOCK_SIZE,                                         \
                                            BLOCK_COUNT,                                        \
                                            MAX_ELEMENTS,                                       \
                                            MAX_DATASET_DIM,                                    \
                                            DATA_T,                                             \
                                            DISTANCE_T,                                         \
                                            INDEX_T,                                            \
                                            LOAD_T)                                             \
  extern template __global__ void search_kernel<TEAM_SIZE,                                      \
                                                BLOCK_SIZE,                                     \
                                                BLOCK_COUNT,                                    \
                                                MAX_ELEMENTS,                                   \
                                                MAX_DATASET_DIM,                                \
                                                DATA_T,                                         \
                                                DISTANCE_T,                                     \
                                                INDEX_T,                                        \
                                                LOAD_T>(INDEX_T* const result_indices_ptr,      \
                                                        DISTANCE_T* const result_distances_ptr, \
                                                        const DATA_T* const dataset_ptr,        \
                                                        const size_t dataset_dim,               \
                                                        const size_t dataset_size,              \
                                                        const size_t dataset_ld,                \
                                                        const DATA_T* const queries_ptr,        \
                                                        const INDEX_T* const knn_graph,         \
                                                        const uint32_t graph_degree,            \
                                                        const unsigned num_distilation,         \
                                                        const uint64_t rand_xor_mask,           \
                                                        const INDEX_T* seed_ptr,                \
                                                        const uint32_t num_seeds,               \
                                                        INDEX_T* const visited_hashmap_ptr,     \
                                                        const uint32_t hash_bitlen,             \
                                                        const uint32_t itopk_size,              \
                                                        const uint32_t num_parents,             \
                                                        const uint32_t min_iteration,           \
                                                        const uint32_t max_iteration,           \
                                                        uint32_t* const num_executed_iterations);

// search_multi_cta_float_uint32_dim1024_t32.cu
instantiate_multi_cta_search_kernel(32, 64, 16, 64, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 128, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 256, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 64, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 128, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 256, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 64, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 128, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 256, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 64, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 128, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 256, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 64, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 128, 1024, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 256, 1024, float, float, uint32_t, uint4);

// search_multi_cta_float_uint32_dim128_t8.cu
instantiate_multi_cta_search_kernel(8, 64, 16, 64, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 64, 16, 128, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 64, 16, 256, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 64, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 128, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 256, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 64, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 128, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 256, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 64, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 128, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 256, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 64, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 128, 128, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 256, 128, float, float, uint32_t, uint4);

// search_multi_cta_float_uint32_dim256_t16.cu
instantiate_multi_cta_search_kernel(16, 64, 16, 64, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 64, 16, 128, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 64, 16, 256, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 64, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 128, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 256, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 64, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 128, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 256, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 64, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 128, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 256, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 64, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 128, 256, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 256, 256, float, float, uint32_t, uint4);

// search_multi_cta_float_uint32_dim512_t32.cu
instantiate_multi_cta_search_kernel(32, 64, 16, 64, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 128, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 256, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 64, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 128, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 256, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 64, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 128, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 256, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 64, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 128, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 256, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 64, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 128, 512, float, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 256, 512, float, float, uint32_t, uint4);

// search_multi_cta_int8_uint32_dim1024_t32.cu
instantiate_multi_cta_search_kernel(32, 64, 16, 64, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 128, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 256, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 64, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 128, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 256, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 64, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 128, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 256, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 64, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 128, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 256, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 64, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 128, 1024, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 256, 1024, int8_t, float, uint32_t, uint4);

// search_multi_cta_int8_uint32_dim128_t8.cu
instantiate_multi_cta_search_kernel(8, 64, 16, 64, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 64, 16, 128, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 64, 16, 256, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 64, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 128, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 256, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 64, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 128, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 256, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 64, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 128, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 256, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 64, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 128, 128, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 256, 128, int8_t, float, uint32_t, uint4);

// search_multi_cta_int8_uint32_dim256_t16.cu
instantiate_multi_cta_search_kernel(16, 64, 16, 64, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 64, 16, 128, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 64, 16, 256, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 64, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 128, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 256, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 64, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 128, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 256, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 64, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 128, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 256, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 64, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 128, 256, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 256, 256, int8_t, float, uint32_t, uint4);

// search_multi_cta_int8_uint32_dim512_t32.cu
instantiate_multi_cta_search_kernel(32, 64, 16, 64, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 128, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 256, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 64, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 128, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 256, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 64, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 128, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 256, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 64, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 128, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 256, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 64, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 128, 512, int8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 256, 512, int8_t, float, uint32_t, uint4);

// search_multi_cta_uint8_uint32_dim1024_t32.cu
instantiate_multi_cta_search_kernel(32, 64, 16, 64, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 128, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 256, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 64, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 128, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 256, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 64, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 128, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 256, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 64, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 128, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 256, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 64, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 128, 1024, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 256, 1024, uint8_t, float, uint32_t, uint4);

// search_multi_cta_uint8_uint32_dim128_t8.cu
instantiate_multi_cta_search_kernel(8, 64, 16, 64, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 64, 16, 128, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 64, 16, 256, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 64, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 128, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 128, 8, 256, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 64, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 128, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 256, 4, 256, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 64, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 128, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 512, 2, 256, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 64, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 128, 128, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(8, 1024, 1, 256, 128, uint8_t, float, uint32_t, uint4);

// search_multi_cta_uint8_uint32_dim256_t16.cu
instantiate_multi_cta_search_kernel(16, 64, 16, 64, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 64, 16, 128, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 64, 16, 256, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 64, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 128, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 128, 8, 256, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 64, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 128, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 256, 4, 256, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 64, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 128, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 512, 2, 256, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 64, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 128, 256, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(16, 1024, 1, 256, 256, uint8_t, float, uint32_t, uint4);

// search_multi_cta_uint8_uint32_dim512_t32.cu
instantiate_multi_cta_search_kernel(32, 64, 16, 64, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 128, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 64, 16, 256, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 64, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 128, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 128, 8, 256, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 64, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 128, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 256, 4, 256, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 64, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 128, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 512, 2, 256, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 64, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 128, 512, uint8_t, float, uint32_t, uint4);
instantiate_multi_cta_search_kernel(32, 1024, 1, 256, 512, uint8_t, float, uint32_t, uint4);

#undef instantiate_multi_cta_search_kernel
}  // namespace multi_cta_search
}  // namespace raft::neighbors::experimental::cagra::detail
