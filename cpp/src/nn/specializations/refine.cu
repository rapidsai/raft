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

#include <raft/neighbors/refine.cuh>

namespace raft::neighbors {

#define RAFT_INST_REFINE(IDX_T, DATA_T)                                                 \
  void refine(raft::handle_t const& handle,                                             \
              raft::device_matrix_view<DATA_T, uint64_t, row_major> dataset,            \
              raft::device_matrix_view<DATA_T, uint64_t, row_major> queries,            \
              raft::device_matrix_view<IDX_T, uint64_t, row_major> neighbor_candidates, \
              raft::device_matrix_view<IDX_T, uint64_t, row_major> indices,             \
              raft::device_matrix_view<float, uint64_t, row_major> distances,           \
              distance::DistanceType metric)                                            \
  {                                                                                     \
    detail::refine_device<IDX_T, DATA_T, float, uint64_t>(                              \
      handle, dataset, queries, neighbor_candidates, indices, distances, metric);       \
  }

RAFT_INST_REFINE(uint64_t, float);
RAFT_INST_REFINE(uint64_t, uint8_t);
RAFT_INST_REFINE(uint64_t, int8_t);

#undef RAFT_INST_REFINE

}  // namespace raft::neighbors
