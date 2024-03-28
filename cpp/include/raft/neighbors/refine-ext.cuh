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

#include <raft/core/device_mdspan.hpp>       // raft::device_matrix_view
#include <raft/core/host_mdspan.hpp>         // // raft::host_matrix_view
#include <raft/core/resources.hpp>           // raft::resources
#include <raft/distance/distance_types.hpp>  // raft::distance::DistanceType
#include <raft/util/raft_explicit.hpp>       // RAFT_EXPLICIT

#include <cstdint>  // int64_t

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors {

template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine(raft::resources const& handle,
            raft::device_matrix_view<const data_t, matrix_idx, row_major> dataset,
            raft::device_matrix_view<const data_t, matrix_idx, row_major> queries,
            raft::device_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,
            raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,
            raft::device_matrix_view<distance_t, matrix_idx, row_major> distances,
            raft::distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
  RAFT_EXPLICIT;

template <typename idx_t, typename data_t, typename distance_t, typename matrix_idx>
void refine(raft::resources const& handle,
            raft::host_matrix_view<const data_t, matrix_idx, row_major> dataset,
            raft::host_matrix_view<const data_t, matrix_idx, row_major> queries,
            raft::host_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,
            raft::host_matrix_view<idx_t, matrix_idx, row_major> indices,
            raft::host_matrix_view<distance_t, matrix_idx, row_major> distances,
            raft::distance::DistanceType metric = distance::DistanceType::L2Unexpanded)
  RAFT_EXPLICIT;

}  // namespace raft::neighbors

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_refine_d(idx_t, data_t, distance_t, matrix_idx)     \
  extern template void raft::neighbors::refine<idx_t, data_t, distance_t, matrix_idx>( \
    raft::resources const& handle,                                                     \
    raft::device_matrix_view<const data_t, matrix_idx, row_major> dataset,             \
    raft::device_matrix_view<const data_t, matrix_idx, row_major> queries,             \
    raft::device_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,  \
    raft::device_matrix_view<idx_t, matrix_idx, row_major> indices,                    \
    raft::device_matrix_view<distance_t, matrix_idx, row_major> distances,             \
    raft::distance::DistanceType metric);

#define instantiate_raft_neighbors_refine_h(idx_t, data_t, distance_t, matrix_idx)     \
  extern template void raft::neighbors::refine<idx_t, data_t, distance_t, matrix_idx>( \
    raft::resources const& handle,                                                     \
    raft::host_matrix_view<const data_t, matrix_idx, row_major> dataset,               \
    raft::host_matrix_view<const data_t, matrix_idx, row_major> queries,               \
    raft::host_matrix_view<const idx_t, matrix_idx, row_major> neighbor_candidates,    \
    raft::host_matrix_view<idx_t, matrix_idx, row_major> indices,                      \
    raft::host_matrix_view<distance_t, matrix_idx, row_major> distances,               \
    raft::distance::DistanceType metric);

instantiate_raft_neighbors_refine_d(int64_t, float, float, int64_t);
instantiate_raft_neighbors_refine_d(int64_t, int8_t, float, int64_t);
instantiate_raft_neighbors_refine_d(int64_t, uint8_t, float, int64_t);

instantiate_raft_neighbors_refine_h(int64_t, float, float, int64_t);
instantiate_raft_neighbors_refine_h(uint32_t, float, float, int64_t);
instantiate_raft_neighbors_refine_h(int64_t, int8_t, float, int64_t);
instantiate_raft_neighbors_refine_h(int64_t, uint8_t, float, int64_t);

#undef instantiate_raft_neighbors_refine_d
#undef instantiate_raft_neighbors_refine_h
