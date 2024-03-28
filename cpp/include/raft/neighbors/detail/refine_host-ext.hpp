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

#include <raft/core/detail/macros.hpp>       // _RAFT_HAS_CUDA
#include <raft/core/host_mdspan.hpp>         // raft::host_matrix_view
#include <raft/distance/distance_types.hpp>  // raft::distance::DistanceType
#include <raft/util/raft_explicit.hpp>       // RAFT_EXPLICIT

#include <cstdint>  // int64_t

#if defined(_RAFT_HAS_CUDA)
#include <cuda_fp16.h>
#endif

#ifdef RAFT_EXPLICIT_INSTANTIATE_ONLY

namespace raft::neighbors::detail {

template <typename IdxT, typename DataT, typename DistanceT, typename ExtentsT>
[[gnu::optimize(3), gnu::optimize("tree-vectorize")]] void refine_host(
  raft::host_matrix_view<const DataT, ExtentsT, row_major> dataset,
  raft::host_matrix_view<const DataT, ExtentsT, row_major> queries,
  raft::host_matrix_view<const IdxT, ExtentsT, row_major> neighbor_candidates,
  raft::host_matrix_view<IdxT, ExtentsT, row_major> indices,
  raft::host_matrix_view<DistanceT, ExtentsT, row_major> distances,
  distance::DistanceType metric = distance::DistanceType::L2Unexpanded) RAFT_EXPLICIT;

}

#endif  // RAFT_EXPLICIT_INSTANTIATE_ONLY

#define instantiate_raft_neighbors_refine(IdxT, DataT, DistanceT, ExtentsT)                    \
  extern template void raft::neighbors::detail::refine_host<IdxT, DataT, DistanceT, ExtentsT>( \
    raft::host_matrix_view<const DataT, ExtentsT, row_major> dataset,                          \
    raft::host_matrix_view<const DataT, ExtentsT, row_major> queries,                          \
    raft::host_matrix_view<const IdxT, ExtentsT, row_major> neighbor_candidates,               \
    raft::host_matrix_view<IdxT, ExtentsT, row_major> indices,                                 \
    raft::host_matrix_view<DistanceT, ExtentsT, row_major> distances,                          \
    distance::DistanceType metric);

instantiate_raft_neighbors_refine(int64_t, float, float, int64_t);
instantiate_raft_neighbors_refine(uint32_t, float, float, int64_t);
instantiate_raft_neighbors_refine(int64_t, int8_t, float, int64_t);
instantiate_raft_neighbors_refine(int64_t, uint8_t, float, int64_t);

#if defined(_RAFT_HAS_CUDA)
instantiate_raft_neighbors_refine(int64_t, half, float, int64_t);
#endif

#undef instantiate_raft_neighbors_refine
