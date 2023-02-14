/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

#include <raft/neighbors/ball_cover.cuh>
#include <raft/neighbors/ball_cover_types.hpp>

// Ignore upstream specializations to avoid unnecessary recompiling
static_assert(RAFT_DISTANCE_COMPILED, "Requires distance specializations");
#include <raft/distance/specializations.cuh>
#include <raft/neighbors/specializations/detail/ball_cover_lowdim.hpp>
#include <raft/neighbors/specializations/fused_l2_knn.cuh>
#include <raft/neighbors/specializations/knn.cuh>

#include <cstdint>

namespace raft::neighbors::ball_cover {
template void knn_query<std::int64_t, float, std::uint32_t>(
  raft::device_resources const& handle,
  const BallCoverIndex<std::int64_t, float, std::uint32_t, std::uint32_t>& index,
  std::uint32_t k,
  const float* query,
  std::uint32_t n_query_pts,
  std::int64_t* inds,
  float* dists,
  bool perform_post_filtering,
  float weight);

};  // namespace raft::neighbors::ball_cover
