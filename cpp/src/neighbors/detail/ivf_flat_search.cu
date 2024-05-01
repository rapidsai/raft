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

#include <raft/neighbors/detail/ivf_flat_search-inl.cuh>
#include <raft/neighbors/sample_filter_types.hpp>

#include <rmm/resource_ref.hpp>

#define instantiate_raft_neighbors_ivf_flat_detail_search(T, IdxT, IvfSampleFilterT)  \
  template void raft::neighbors::ivf_flat::detail::search<T, IdxT, IvfSampleFilterT>( \
    raft::resources const& handle,                                                    \
    const search_params& params,                                                      \
    const raft::neighbors::ivf_flat::index<T, IdxT>& index,                           \
    const T* queries,                                                                 \
    uint32_t n_queries,                                                               \
    uint32_t k,                                                                       \
    IdxT* neighbors,                                                                  \
    float* distances,                                                                 \
    rmm::device_async_resource_ref mr,                                                \
    IvfSampleFilterT sample_filter)

instantiate_raft_neighbors_ivf_flat_detail_search(
  float, int64_t, raft::neighbors::filtering::none_ivf_sample_filter);
instantiate_raft_neighbors_ivf_flat_detail_search(
  int8_t, int64_t, raft::neighbors::filtering::none_ivf_sample_filter);
instantiate_raft_neighbors_ivf_flat_detail_search(
  uint8_t, int64_t, raft::neighbors::filtering::none_ivf_sample_filter);

#undef instantiate_raft_neighbors_ivf_flat_detail_search
