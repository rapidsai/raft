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

#include <raft/neighbors/refine.cuh>

namespace raft::neighbors {

#ifdef RAFT_INST
#undef RAFT_INST
#endif

#define RAFT_INST(T, IdxT)                                                        \
  extern template void refine<IdxT, T, float, int64_t>(                           \
    raft::device_resources const& handle,                                         \
    raft::device_matrix_view<const T, int64_t, row_major> dataset,                \
    raft::device_matrix_view<const T, int64_t, row_major> queries,                \
    raft::device_matrix_view<const IdxT, int64_t, row_major> neighbor_candidates, \
    raft::device_matrix_view<IdxT, int64_t, row_major> indices,                   \
    raft::device_matrix_view<float, int64_t, row_major> distances,                \
    distance::DistanceType metric);                                               \
                                                                                  \
  extern template void refine<IdxT, T, float, int64_t>(                           \
    raft::device_resources const& handle,                                         \
    raft::host_matrix_view<const T, int64_t, row_major> dataset,                  \
    raft::host_matrix_view<const T, int64_t, row_major> queries,                  \
    raft::host_matrix_view<const IdxT, int64_t, row_major> neighbor_candidates,   \
    raft::host_matrix_view<IdxT, int64_t, row_major> indices,                     \
    raft::host_matrix_view<float, int64_t, row_major> distances,                  \
    distance::DistanceType metric);

RAFT_INST(float, int64_t);
RAFT_INST(uint8_t, int64_t);
RAFT_INST(int8_t, int64_t);

#undef RAFT_INST
}  // namespace raft::neighbors
