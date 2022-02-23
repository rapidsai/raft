/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <raft/spatial/knn/ball_cover.cuh>
#include <raft/spatial/knn/ball_cover_common.h>
#include <raft/spatial/knn/specializations/detail/ball_cover_lowdim.hpp>

#include <cstdint>

namespace raft {
namespace spatial {
namespace knn {
extern template class BallCoverIndex<int, float, std::uint32_t>;
extern template class BallCoverIndex<std::int64_t, float, std::uint32_t>;

extern template void rbc_build_index<std::int64_t, float, std::uint32_t>(
  const raft::handle_t& handle, BallCoverIndex<std::int64_t, float, std::uint32_t>& index);

extern template void rbc_knn_query<std::int64_t, float, std::uint32_t>(
  const raft::handle_t& handle,
  BallCoverIndex<std::int64_t, float, std::uint32_t>& index,
  std::uint32_t k,
  const float* query,
  std::uint32_t n_query_pts,
  std::int64_t* inds,
  float* dists,
  bool perform_post_filtering,
  float weight);

extern template void rbc_all_knn_query<std::int64_t, float, std::uint32_t>(
  const raft::handle_t& handle,
  BallCoverIndex<std::int64_t, float, std::uint32_t>& index,
  std::uint32_t k,
  std::int64_t* inds,
  float* dists,
  bool perform_post_filtering,
  float weight);
};  // namespace knn
};  // namespace spatial
};  // namespace raft
