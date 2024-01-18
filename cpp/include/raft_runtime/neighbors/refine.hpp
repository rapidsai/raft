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

#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/distance/distance_types.hpp>

namespace raft::runtime::neighbors {

#define RAFT_INST_REFINE(IDX_T, DATA_T)                                                      \
  void refine(raft::resources const& handle,                                                 \
              raft::device_matrix_view<const DATA_T, int64_t, row_major> dataset,            \
              raft::device_matrix_view<const DATA_T, int64_t, row_major> queries,            \
              raft::device_matrix_view<const IDX_T, int64_t, row_major> neighbor_candidates, \
              raft::device_matrix_view<IDX_T, int64_t, row_major> indices,                   \
              raft::device_matrix_view<float, int64_t, row_major> distances,                 \
              raft::distance::DistanceType metric);                                          \
                                                                                             \
  void refine(raft::resources const& handle,                                                 \
              raft::host_matrix_view<const DATA_T, int64_t, row_major> dataset,              \
              raft::host_matrix_view<const DATA_T, int64_t, row_major> queries,              \
              raft::host_matrix_view<const IDX_T, int64_t, row_major> neighbor_candidates,   \
              raft::host_matrix_view<IDX_T, int64_t, row_major> indices,                     \
              raft::host_matrix_view<float, int64_t, row_major> distances,                   \
              raft::distance::DistanceType metric);

RAFT_INST_REFINE(int64_t, float);
RAFT_INST_REFINE(int64_t, uint8_t);
RAFT_INST_REFINE(int64_t, int8_t);

#undef RAFT_INST_REFINE

}  // namespace raft::runtime::neighbors
