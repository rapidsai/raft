/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/neighbors/ivf_flat_types.hpp>       // raft::neighbors::ivf_flat::index
#include <raft/neighbors/sample_filter_types.hpp>  // none_ivf_sample_filter
#include <raft/util/raft_explicit.hpp>             // RAFT_EXPLICIT

#include <rmm/cuda_stream_view.hpp>  // rmm:cuda_stream_view

#include <cuda_fp16.h>

#include <cstdint>  // uintX_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors::ivf_flat::detail {

auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k) -> bool;

template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void ivfflat_interleaved_scan(const raft::neighbors::ivf_flat::index<T, IdxT>& index,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const uint32_t queries_offset,
                              const raft::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const uint32_t max_samples,
                              const uint32_t* chunk_indices,
                              const bool select_min,
                              IvfSampleFilterT sample_filter,
                              uint32_t* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream) RAFT_EXPLICIT;

}  // namespace raft::neighbors::ivf_flat::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(                    \
  T, AccT, IdxT, IvfSampleFilterT)                                                              \
  extern template void                                                                          \
  raft::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<T, AccT, IdxT, IvfSampleFilterT>( \
    const raft::neighbors::ivf_flat::index<T, IdxT>& index,                                     \
    const T* queries,                                                                           \
    const uint32_t* coarse_query_results,                                                       \
    const uint32_t n_queries,                                                                   \
    const uint32_t queries_offset,                                                              \
    const raft::distance::DistanceType metric,                                                  \
    const uint32_t n_probes,                                                                    \
    const uint32_t k,                                                                           \
    const uint32_t max_samples,                                                                 \
    const uint32_t* chunk_indices,                                                              \
    const bool select_min,                                                                      \
    IvfSampleFilterT sample_filter,                                                             \
    uint32_t* neighbors,                                                                        \
    float* distances,                                                                           \
    uint32_t& grid_dim_x,                                                                       \
    rmm::cuda_stream_view stream)

instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(
  float, float, int64_t, raft::neighbors::filtering::none_ivf_sample_filter);
instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(
  half, half, int64_t, raft::neighbors::filtering::none_ivf_sample_filter);
instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(
  int8_t, int32_t, int64_t, raft::neighbors::filtering::none_ivf_sample_filter);
instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(
  uint8_t, uint32_t, int64_t, raft::neighbors::filtering::none_ivf_sample_filter);

#undef instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan
