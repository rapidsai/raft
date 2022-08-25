/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "../ivf_pq_types.hpp"
#include "ann_utils.cuh"
#include "ivf_pq_legacy.cuh"

#include <raft/common/device_loads_stores.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/pow2_utils.cuh>
#include <raft/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <optional>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

/** See raft::spatial::knn::ivf_pq::search docs */
template <typename T, typename IdxT>
inline void search(const handle_t& handle,
                   const search_params& params,
                   const index<IdxT>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   IdxT* neighbors,
                   float* distances,
                   rmm::mr::device_memory_resource* mr = nullptr)
{
  static_assert(std::is_same_v<IdxT, uint64_t>,
                "Only uint64_t index output is supported at this time.");
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());

  auto pool_guard = raft::get_pool_memory_resource(mr, n_queries * n_probes * k * 16);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::search: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  auto& index_mut = const_cast<ivf_pq::index<IdxT>&>(index);

  // set search parameters
  ivf_pq::detail::cuannIvfPqSetSearchParameters(index_mut, n_probes, k);
  ivf_pq::detail::cuannIvfPqSetSearchTuningParameters(index_mut,
                                                      params.internal_distance_dtype,
                                                      params.smem_lut_dtype,
                                                      params.preferred_thread_block_size);
  // Maximum number of query vectors to search at the same time.
  uint32_t batch_size = std::min<uint32_t>(n_queries, 32768);
  size_t max_ws_size  = (size_t)2 * 1024 * 1024 * 1024;  // 2 GiB
  // Allocate memory for index
  size_t ivf_pq_search_workspace_size;
  ivf_pq::detail::cuannIvfPqSearch_bufferSize(
    handle, index_mut, batch_size, max_ws_size, &ivf_pq_search_workspace_size);

  // finally, search!
  ivf_pq::detail::cuannIvfPqSearch(handle, index_mut, queries, n_queries, neighbors, distances, mr);
}

}  // namespace raft::spatial::knn::ivf_pq::detail
