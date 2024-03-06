/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cuda_fp16.h>                       // __half
#include <raft/core/detail/macros.hpp>       // RAFT_WEAK_FUNCTION
#include <raft/distance/distance_types.hpp>  // raft::distance::DistanceType
#include <raft/neighbors/detail/ivf_pq_compute_similarity.cuh>
#include <raft/neighbors/detail/ivf_pq_fp_8bit.cuh>  // raft::neighbors::ivf_pq::detail::fp_8bit
#include <raft/neighbors/sample_filter.cuh>          // none_ivf_sample_filter
#include <raft/neighbors/sample_filter_types.hpp>    // none_ivf_sample_filter
#include <rmm/cuda_stream_view.hpp>                  // rmm::cuda_stream_view

#define instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(                 \
  OutT, LutT, IvfSampleFilterT)                                                             \
  extern template auto                                                                      \
  raft::neighbors::ivf_pq::detail::compute_similarity_select<OutT, LutT, IvfSampleFilterT>( \
    const cudaDeviceProp& dev_props,                                                        \
    bool manage_local_topk,                                                                 \
    int locality_hint,                                                                      \
    double preferred_shmem_carveout,                                                        \
    uint32_t pq_bits,                                                                       \
    uint32_t pq_dim,                                                                        \
    uint32_t precomp_data_count,                                                            \
    uint32_t n_queries,                                                                     \
    uint32_t n_probes,                                                                      \
    uint32_t topk)                                                                          \
    ->raft::neighbors::ivf_pq::detail::selected<OutT, LutT, IvfSampleFilterT>;              \
                                                                                            \
  extern template void                                                                      \
  raft::neighbors::ivf_pq::detail::compute_similarity_run<OutT, LutT, IvfSampleFilterT>(    \
    raft::neighbors::ivf_pq::detail::selected<OutT, LutT, IvfSampleFilterT> s,              \
    rmm::cuda_stream_view stream,                                                           \
    uint32_t dim,                                                                           \
    uint32_t n_probes,                                                                      \
    uint32_t pq_dim,                                                                        \
    uint32_t n_queries,                                                                     \
    uint32_t queries_offset,                                                                \
    raft::distance::DistanceType metric,                                                    \
    raft::neighbors::ivf_pq::codebook_gen codebook_kind,                                    \
    uint32_t topk,                                                                          \
    uint32_t max_samples,                                                                   \
    const float* cluster_centers,                                                           \
    const float* pq_centers,                                                                \
    const uint8_t* const* pq_dataset,                                                       \
    const uint32_t* cluster_labels,                                                         \
    const uint32_t* _chunk_indices,                                                         \
    const float* queries,                                                                   \
    const uint32_t* index_list,                                                             \
    float* query_kths,                                                                      \
    IvfSampleFilterT sample_filter,                                                         \
    LutT* lut_scores,                                                                       \
    OutT* _out_scores,                                                                      \
    uint32_t* _out_indices);

#define COMMA ,
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  half,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  half,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  float,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::none_ivf_sample_filter>);

instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  half,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  half,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  float,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    uint32_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA uint32_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  half,
  half,
  raft::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  half,
  raft::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  float,
  raft::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA false>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select(
  float,
  raft::neighbors::ivf_pq::detail::fp_8bit<5u COMMA true>,
  raft::neighbors::filtering::ivf_to_sample_filter<
    int64_t COMMA raft::neighbors::filtering::bitset_filter<uint32_t COMMA int64_t>>);
#undef COMMA

#undef instantiate_raft_neighbors_ivf_pq_detail_compute_similarity_select
