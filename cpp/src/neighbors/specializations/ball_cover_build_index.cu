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
#include <raft/neighbors/specializations.cuh>

#include <cstdint>

namespace raft::neighbors::ball_cover {
template class BallCoverIndex<int, float, std::uint32_t, std::uint32_t>;
template class BallCoverIndex<std::int64_t, float, std::uint32_t, std::uint32_t>;

template void build_index<std::int64_t, float, std::uint32_t, std::uint32_t>(
  raft::device_resources const& handle,
  BallCoverIndex<std::int64_t, float, std::uint32_t, std::uint32_t>& index);

};  // namespace raft::neighbors::ball_cover
