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

#include <raft/neighbors/sample_filter_types.hpp>
#include <raft/util/raft_explicit.hpp>  // RAFT_EXPLICIT

#include <cuda_fp16.h>

namespace raft::neighbors::cagra::detail {
namespace single_cta_search {

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

template <unsigned TEAM_SIZE,
          unsigned MAX_DATASET_DIM,
          typename DATASET_DESCRIPTOR_T,
          typename SAMPLE_FILTER_T>
void select_and_run(  // raft::resources const& res,
  DATASET_DESCRIPTOR_T dataset_desc,
  raft::device_matrix_view<const typename DATASET_DESCRIPTOR_T::INDEX_T, int64_t, row_major> graph,
  typename DATASET_DESCRIPTOR_T::INDEX_T* const topk_indices_ptr,       // [num_queries, topk]
  typename DATASET_DESCRIPTOR_T::DISTANCE_T* const topk_distances_ptr,  // [num_queries, topk]
  const typename DATASET_DESCRIPTOR_T::DATA_T* const queries_ptr,  // [num_queries, dataset_dim]
  const uint32_t num_queries,
  const typename DATASET_DESCRIPTOR_T::INDEX_T* dev_seed_ptr,  // [num_queries, num_seeds]
  uint32_t* const num_executed_iterations,                     // [num_queries,]
  uint32_t topk,
  uint32_t num_itopk_candidates,
  uint32_t block_size,
  uint32_t smem_size,
  int64_t hash_bitlen,
  typename DATASET_DESCRIPTOR_T::INDEX_T* hashmap_ptr,
  size_t small_hash_bitlen,
  size_t small_hash_reset_interval,
  uint32_t num_random_samplings,
  uint64_t rand_xor_mask,
  uint32_t num_seeds,
  size_t itopk_size,
  size_t search_width,
  size_t min_iterations,
  size_t max_iterations,
  SAMPLE_FILTER_T sample_filter,
  raft::distance::DistanceType metric,
  cudaStream_t stream) RAFT_EXPLICIT;

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_single_cta_select_and_run(                                                  \
  TEAM_SIZE, MAX_DATASET_DIM, DATA_T, INDEX_T, DISTANCE_T, SAMPLE_FILTER_T)                     \
  extern template void select_and_run<                                                          \
    TEAM_SIZE,                                                                                  \
    MAX_DATASET_DIM,                                                                            \
    raft::neighbors::cagra::detail::standard_dataset_descriptor_t<DATA_T, INDEX_T, DISTANCE_T>, \
    SAMPLE_FILTER_T>(                                                                           \
    raft::neighbors::cagra::detail::standard_dataset_descriptor_t<DATA_T, INDEX_T, DISTANCE_T>  \
      dataset,                                                                                  \
    raft::device_matrix_view<const INDEX_T, int64_t, row_major> graph,                          \
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
    size_t search_width,                                                                        \
    size_t min_iterations,                                                                      \
    size_t max_iterations,                                                                      \
    SAMPLE_FILTER_T sample_filter,                                                              \
    raft::distance::DistanceType metric,                                                        \
    cudaStream_t stream);

instantiate_single_cta_select_and_run(
  32, 1024, float, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  8, 128, float, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  16, 256, float, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  32, 512, float, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  32, 1024, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  8, 128, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  16, 256, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  32, 512, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  32, 1024, int8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  8, 128, int8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  16, 256, int8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  32, 512, int8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  32, 1024, uint8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  8, 128, uint8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  16, 256, uint8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_single_cta_select_and_run(
  32, 512, uint8_t, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);

#undef instantiate_single_cta_select_and_run

#define instantiate_q_single_cta_select_and_run(TEAM_SIZE,                                      \
                                                MAX_DATASET_DIM,                                \
                                                CODE_BOOK_T,                                    \
                                                PQ_BITS,                                        \
                                                PQ_CODE_BOOK_DIM,                               \
                                                DATA_T,                                         \
                                                INDEX_T,                                        \
                                                DISTANCE_T,                                     \
                                                SAMPLE_FILTER_T)                                \
  extern template void                                                                          \
  select_and_run<TEAM_SIZE,                                                                     \
                 MAX_DATASET_DIM,                                                               \
                 raft::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<DATA_T,           \
                                                                              CODE_BOOK_T,      \
                                                                              PQ_BITS,          \
                                                                              PQ_CODE_BOOK_DIM, \
                                                                              DISTANCE_T,       \
                                                                              INDEX_T>,         \
                 SAMPLE_FILTER_T>(                                                              \
    raft::neighbors::cagra::detail::cagra_q_dataset_descriptor_t<DATA_T,                        \
                                                                 CODE_BOOK_T,                   \
                                                                 PQ_BITS,                       \
                                                                 PQ_CODE_BOOK_DIM,              \
                                                                 DISTANCE_T,                    \
                                                                 INDEX_T> dataset,              \
    raft::device_matrix_view<const INDEX_T, int64_t, row_major> graph,                          \
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
    size_t search_width,                                                                        \
    size_t min_iterations,                                                                      \
    size_t max_iterations,                                                                      \
    SAMPLE_FILTER_T sample_filter,                                                              \
    raft::distance::DistanceType metric,                                                        \
    cudaStream_t stream);

instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 2, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  16, 256, half, 8, 2, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 512, half, 8, 2, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        2,
                                        half,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 4, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  16, 256, half, 8, 4, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 512, half, 8, 4, half, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        4,
                                        half,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);

instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 2, float, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        2,
                                        float,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        2,
                                        float,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        2,
                                        float,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 4, float, uint32_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        4,
                                        float,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        4,
                                        float,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        4,
                                        float,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);

instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 2, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  16, 256, half, 8, 2, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 512, half, 8, 2, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 1024, half, 8, 2, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 4, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  16, 256, half, 8, 4, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 512, half, 8, 4, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 1024, half, 8, 4, half, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);

instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 2, float, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  16, 256, half, 8, 2, float, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 512, half, 8, 2, float, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        2,
                                        float,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 4, float, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  16, 256, half, 8, 4, float, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  32, 512, half, 8, 4, float, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        4,
                                        float,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);

instantiate_q_single_cta_select_and_run(8,
                                        128,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(8,
                                        128,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);

instantiate_q_single_cta_select_and_run(8,
                                        128,
                                        half,
                                        8,
                                        2,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        2,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        2,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        2,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(8,
                                        128,
                                        half,
                                        8,
                                        4,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        4,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        4,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        4,
                                        int8_t,
                                        uint32_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);

instantiate_q_single_cta_select_and_run(8,
                                        128,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        2,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(8,
                                        128,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        4,
                                        uint8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);

instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 2, int8_t, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        2,
                                        int8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        2,
                                        int8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        2,
                                        int8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(
  8, 128, half, 8, 4, int8_t, int64_t, float, raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(16,
                                        256,
                                        half,
                                        8,
                                        4,
                                        int8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        512,
                                        half,
                                        8,
                                        4,
                                        int8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);
instantiate_q_single_cta_select_and_run(32,
                                        1024,
                                        half,
                                        8,
                                        4,
                                        int8_t,
                                        int64_t,
                                        float,
                                        raft::neighbors::filtering::none_cagra_sample_filter);

#undef instantiate_q_single_cta_select_and_run

}  // namespace single_cta_search
}  // namespace raft::neighbors::cagra::detail
