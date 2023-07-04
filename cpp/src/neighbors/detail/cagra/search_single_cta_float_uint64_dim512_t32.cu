
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

/*
 * NOTE: this file is generated by search_single_cta_00_generate.py
 *
 * Make changes there and run in this directory:
 *
 * > python search_single_cta_00_generate.py
 *
 */

#include <raft/neighbors/detail/cagra/search_single_cta_kernel-inl.cuh>

namespace raft::neighbors::experimental::cagra::detail::single_cta_search {

#define instantiate_single_cta_search_kernel(TEAM_SIZE,                                        \
                                             BLOCK_SIZE,                                       \
                                             BLOCK_COUNT,                                      \
                                             MAX_ITOPK,                                        \
                                             MAX_CANDIDATES,                                   \
                                             TOPK_BY_BITONIC_SORT,                             \
                                             MAX_DATASET_DIM,                                  \
                                             DATA_T,                                           \
                                             DISTANCE_T,                                       \
                                             INDEX_T,                                          \
                                             LOAD_T)                                           \
  template __global__ void search_kernel<TEAM_SIZE,                                            \
                                         BLOCK_SIZE,                                           \
                                         BLOCK_COUNT,                                          \
                                         MAX_ITOPK,                                            \
                                         MAX_CANDIDATES,                                       \
                                         TOPK_BY_BITONIC_SORT,                                 \
                                         MAX_DATASET_DIM,                                      \
                                         DATA_T,                                               \
                                         DISTANCE_T,                                           \
                                         INDEX_T,                                              \
                                         LOAD_T>(INDEX_T* const result_indices_ptr,            \
                                                 DISTANCE_T* const result_distances_ptr,       \
                                                 const std::uint32_t top_k,                    \
                                                 const DATA_T* const dataset_ptr,              \
                                                 const std::size_t dataset_dim,                \
                                                 const std::size_t dataset_size,               \
                                                 const size_t dataset_ld,                      \
                                                 const DATA_T* const queries_ptr,              \
                                                 const INDEX_T* const knn_graph,               \
                                                 const std::uint32_t graph_degree,             \
                                                 const unsigned num_distilation,               \
                                                 const uint64_t rand_xor_mask,                 \
                                                 const INDEX_T* seed_ptr,                      \
                                                 const uint32_t num_seeds,                     \
                                                 INDEX_T* const visited_hashmap_ptr,           \
                                                 const std::uint32_t internal_topk,            \
                                                 const std::uint32_t num_parents,              \
                                                 const std::uint32_t min_iteration,            \
                                                 const std::uint32_t max_iteration,            \
                                                 std::uint32_t* const num_executed_iterations, \
                                                 const std::uint32_t hash_bitlen,              \
                                                 const std::uint32_t small_hash_bitlen,        \
                                                 const std::uint32_t small_hash_reset_interval);

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

}  // namespace raft::neighbors::experimental::cagra::detail::single_cta_search
