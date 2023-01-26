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

#include <raft/neighbors/refine.cuh>

namespace raft::runtime::neighbors {

void refine(raft::device_resources const& handle,
            raft::device_matrix_view<const int8_t, uint64_t, row_major> dataset,
            raft::device_matrix_view<const int8_t, uint64_t, row_major> queries,
            raft::device_matrix_view<const uint64_t, uint64_t, row_major> neighbor_candidates,
            raft::device_matrix_view<uint64_t, uint64_t, row_major> indices,
            raft::device_matrix_view<float, uint64_t, row_major> distances,
            distance::DistanceType metric)
{
  raft::neighbors::refine<uint64_t, int8_t, float, uint64_t>(
    handle, dataset, queries, neighbor_candidates, indices, distances, metric);
}

}  // namespace raft::runtime::neighbors
