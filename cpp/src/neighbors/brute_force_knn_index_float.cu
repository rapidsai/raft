
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cstdint>
#include <raft/neighbors/brute_force-inl.cuh>

template void raft::neighbors::brute_force::search<float, int>(
  raft::resources const& res,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances,
  std::optional<raft::device_vector_view<const float, int64_t>> query_norms);

template void raft::neighbors::brute_force::search<float, int64_t>(
  raft::resources const& res,
  const raft::neighbors::brute_force::index<float>& idx,
  raft::device_matrix_view<const float, int64_t, row_major> queries,
  raft::device_matrix_view<int64_t, int64_t, row_major> neighbors,
  raft::device_matrix_view<float, int64_t, row_major> distances,
  std::optional<raft::device_vector_view<const float, int64_t>> query_norms);

template raft::neighbors::brute_force::index<float> raft::neighbors::brute_force::build<float>(
  raft::resources const& res,
  raft::device_matrix_view<const float, int64_t, row_major> dataset,
  raft::distance::DistanceType metric,
  float metric_arg);
