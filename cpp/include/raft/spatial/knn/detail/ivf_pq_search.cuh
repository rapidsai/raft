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
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported element type.");
  static_assert(std::is_same_v<IdxT, uint64_t>,
                "Only uint64_t index output is supported at this time.");
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(
    params.internal_distance_dtype == CUDA_R_16F || params.internal_distance_dtype == CUDA_R_32F,
    "internal_distance_dtype must be either CUDA_R_16F or CUDA_R_32F");
  RAFT_EXPECTS(params.smem_lut_dtype == CUDA_R_16F || params.smem_lut_dtype == CUDA_R_32F ||
                 params.smem_lut_dtype == CUDA_R_8U,
               "smem_lut_dtype must be CUDA_R_16F, CUDA_R_32F or CUDA_R_8U");
  RAFT_EXPECTS(
    params.preferred_thread_block_size == 256 || params.preferred_thread_block_size == 512 ||
      params.preferred_thread_block_size == 1024 || params.preferred_thread_block_size == 0,
    "preferred_thread_block_size must be 0, 256, 512 or 1024, but %u is given.",
    params.preferred_thread_block_size);
  RAFT_EXPECTS(k > 0, "parameter `k` in top-k must be positive.");
  RAFT_EXPECTS(
    k <= index.size(),
    "parameter `k` (%u) in top-k must not be larger that the total size of the index (%u)",
    k,
    index.size());
  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());
  {
    uint32_t n_samples_worst_case = index.size();
    if (n_probes < index.n_lists()) {
      n_samples_worst_case = index.size() - index.inclusiveSumSortedClusterSize()(
                                              std::max<uint32_t>(index.numClustersSize0(),
                                                                 index.n_lists() - 1 - n_probes) -
                                              index.numClustersSize0());
    }
    if (k > n_samples_worst_case) {
      RAFT_LOG_WARN(
        "n_probes is too small to get top-k results reliably (n_probes: %u, k: %u, "
        "n_samples_worst_case: %u).",
        n_probes,
        k,
        n_samples_worst_case);
    }
  }

  auto pool_guard = raft::get_pool_memory_resource(mr, n_queries * n_probes * k * 16);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::search: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  auto& index_mut = const_cast<ivf_pq::index<IdxT>&>(index);

  // Maximum number of query vectors to search at the same time.
  uint32_t batch_size = std::min<uint32_t>(n_queries, 32768);
  auto max_queries    = min(max(batch_size, 1), 4096);
  auto max_batch_size = max_queries;
  {
    // TODO: copied from {legacy}; figure this out.
    // Adjust max_batch_size to improve GPU occupancy of topk kernel.
    uint32_t numCta_total    = getMultiProcessorCount() * 2;
    uint32_t numCta_perBatch = numCta_total / max_batch_size;
    float utilization        = (float)numCta_perBatch * max_batch_size / numCta_total;
    if (numCta_perBatch > 1 || (numCta_perBatch == 1 && utilization < 0.6)) {
      uint32_t numCta_perBatch_1 = numCta_perBatch + 1;
      uint32_t maxBatchSize_1    = numCta_total / numCta_perBatch_1;
      float utilization_1        = (float)numCta_perBatch_1 * maxBatchSize_1 / numCta_total;
      if (utilization < utilization_1) { max_batch_size = maxBatchSize_1; }
    }
  }

  // finally, search!
  ivf_pq::detail::cuannIvfPqSearch(handle,
                                   params,
                                   index_mut,
                                   k,
                                   queries,
                                   n_queries,
                                   neighbors,
                                   distances,
                                   mr,
                                   max_queries,
                                   max_batch_size);
}

}  // namespace raft::spatial::knn::ivf_pq::detail
