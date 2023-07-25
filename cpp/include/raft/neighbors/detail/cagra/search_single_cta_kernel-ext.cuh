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
namespace raft::neighbors::cagra::detail {
namespace single_cta_search {

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          typename DATA_T,
          typename INDEX_T,
          typename DISTANCE_T>
void select_and_run(  // raft::resources const& res,
  raft::device_matrix_view<const DATA_T, INDEX_T, layout_stride> dataset,
  raft::device_matrix_view<const INDEX_T, INDEX_T, row_major> graph,
  INDEX_T* const topk_indices_ptr,          // [num_queries, topk]
  DISTANCE_T* const topk_distances_ptr,     // [num_queries, topk]
  const DATA_T* const queries_ptr,          // [num_queries, dataset_dim]
  const uint32_t num_queries,
  const INDEX_T* dev_seed_ptr,              // [num_queries, num_seeds]
  uint32_t* const num_executed_iterations,  // [num_queries,]
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,
  uint32_t smem_size,
  int64_t hash_bitlen,
  INDEX_T* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_random_samplings,
  uint64_t rand_xor_mask,
  uint32_t num_seeds,
  size_t itopk_size,
  size_t num_parents,
  size_t min_iterations,
  size_t max_iterations,
  cudaStream_t stream) RAFT_EXPLICIT;

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_single_cta_select_and_run(                                                  \
  TEAM_SIZE, MAX_DATASET_DIM, DATA_T, INDEX_T, DISTANCE_T)                                      \
  extern template void select_and_run<TEAM_SIZE, MAX_DATASET_DIM, DATA_T, INDEX_T, DISTANCE_T>( \
    raft::device_matrix_view<const DATA_T, INDEX_T, layout_stride> dataset,                     \
    raft::device_matrix_view<const INDEX_T, INDEX_T, row_major> graph,                          \
    INDEX_T* const topk_indices_ptr,                                                            \
    DISTANCE_T* const topk_distances_ptr,                                                       \
    const DATA_T* const queries_ptr,                                                            \
    const uint32_t num_queries,                                                                 \
    const INDEX_T* dev_seed_ptr,                                                                \
    uint32_t* const num_executed_iterations,                                                    \
    uint32_t topk,                                                                              \
    uint32_t num_itopk_candidates,                                                              \
    uint32_t block_size,                                                                        \
    uint32_t smem_size,                                                                         \
    int64_t hash_bitlen,                                                                        \
    INDEX_T* hashmap_ptr,                                                                       \
    size_t small_hash_bitlen,                                                                   \
    size_t small_hash_reset_interval,                                                           \
    uint32_t num_random_samplings,                                                              \
    uint64_t rand_xor_mask,                                                                     \
    uint32_t num_seeds,                                                                         \
    size_t itopk_size,                                                                          \
    size_t num_parents,                                                                         \
    size_t min_iterations,                                                                      \
    size_t max_iterations,                                                                      \
    cudaStream_t stream);

instantiate_single_cta_select_and_run(32, 1024, float, uint32_t, float);
instantiate_single_cta_select_and_run(8, 128, float, uint32_t, float);
instantiate_single_cta_select_and_run(16, 256, float, uint32_t, float);
instantiate_single_cta_select_and_run(32, 512, float, uint32_t, float);
instantiate_single_cta_select_and_run(32, 1024, int8_t, uint32_t, float);
instantiate_single_cta_select_and_run(8, 128, int8_t, uint32_t, float);
instantiate_single_cta_select_and_run(16, 256, int8_t, uint32_t, float);
instantiate_single_cta_select_and_run(32, 512, int8_t, uint32_t, float);
instantiate_single_cta_select_and_run(32, 1024, uint8_t, uint32_t, float);
instantiate_single_cta_select_and_run(8, 128, uint8_t, uint32_t, float);
instantiate_single_cta_select_and_run(16, 256, uint8_t, uint32_t, float);
instantiate_single_cta_select_and_run(32, 512, uint8_t, uint32_t, float);

#undef instantiate_single_cta_select_and_run

}  // namespace single_cta_search
}  // namespace raft::neighbors::cagra::detail
