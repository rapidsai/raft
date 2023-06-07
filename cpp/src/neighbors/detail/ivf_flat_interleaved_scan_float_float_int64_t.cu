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

#include <raft/neighbors/detail/ivf_flat_interleaved_scan-inl.cuh>

#define instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(T, AccT, IdxT)  \
  template void raft::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<T, AccT, IdxT>( \
    const raft::neighbors::ivf_flat::index<T, IdxT>& index,                                 \
    const T* queries,                                                                       \
    const uint32_t* coarse_query_results,                                                   \
    const uint32_t n_queries,                                                               \
    const raft::distance::DistanceType metric,                                              \
    const uint32_t n_probes,                                                                \
    const uint32_t k,                                                                       \
    const bool select_min,                                                                  \
    IdxT* neighbors,                                                                        \
    float* distances,                                                                       \
    uint32_t& grid_dim_x,                                                                   \
    rmm::cuda_stream_view stream)

instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(float, float, int64_t);

#undef instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan
