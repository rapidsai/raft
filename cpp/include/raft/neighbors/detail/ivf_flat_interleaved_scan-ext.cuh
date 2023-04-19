/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cstdint>                                // uintX_t
#include <raft/neighbors/ivf_flat_types.hpp>      // index
#include <raft/spatial/knn/detail/ann_utils.cuh>  // TODO: consider remove
#include <raft/util/raft_explicit.hpp>            // RAFT_EXPLICIT
#include <rmm/cuda_stream_view.hpp>               // rmm:cuda_stream_view

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors::ivf_flat::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

/**
 * @brief Configure and launch an appropriate template instance of the interleaved scan kernel.
 *
 * @tparam T value type
 * @tparam AccT accumulated type
 * @tparam IdxT type of the indices
 *
 * @param index previously built ivf-flat index
 * @param[in] queries device pointer to the query vectors [batch_size, dim]
 * @param[in] coarse_query_results device pointer to the cluster (list) ids [batch_size, n_probes]
 * @param n_queries batch size
 * @param metric type of the measured distance
 * @param n_probes number of nearest clusters to query
 * @param k number of nearest neighbors.
 *            NB: the maximum value of `k` is limited statically by `kMaxCapacity`.
 * @param select_min whether to select nearest (true) or furthest (false) points w.r.t. the given
 * metric.
 * @param[out] neighbors device pointer to the result indices for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[out] distances device pointer to the result distances for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[inout] grid_dim_x number of blocks launched across all n_probes clusters;
 *               (one block processes one or more probes, hence: 1 <= grid_dim_x <= n_probes)
 * @param stream
 */
template <typename T, typename AccT, typename IdxT>
void ivfflat_interleaved_scan(const raft::neighbors::ivf_flat::index<T, IdxT>& index,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const raft::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const bool select_min,
                              IdxT* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream) RAFT_EXPLICIT;

}  // namespace raft::neighbors::ivf_flat::detail

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(T, AccT, IdxT)         \
  extern template void raft::neighbors::ivf_flat::detail::ivfflat_interleaved_scan<T, AccT, IdxT>( \
    const raft::neighbors::ivf_flat::index<T, IdxT>& index,                                        \
    const T* queries,                                                                              \
    const uint32_t* coarse_query_results,                                                          \
    const uint32_t n_queries,                                                                      \
    const raft::distance::DistanceType metric,                                                     \
    const uint32_t n_probes,                                                                       \
    const uint32_t k,                                                                              \
    const bool select_min,                                                                         \
    IdxT* neighbors,                                                                               \
    float* distances,                                                                              \
    uint32_t& grid_dim_x,                                                                          \
    rmm::cuda_stream_view stream)

instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(float, float, int64_t);
instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(int8_t, int32_t, int64_t);
instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan(uint8_t, uint32_t, int64_t);

#undef instantiate_raft_neighbors_ivf_flat_detail_ivfflat_interleaved_scan
