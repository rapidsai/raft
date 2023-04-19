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
#pragma once

// TODO: consider putting this back in coalesced reduction
namespace raft::linalg::detail {

template <int warpSize, int rpb>
struct ReductionThinPolicy {
  static constexpr int LogicalWarpSize = warpSize;
  static constexpr int RowsPerBlock    = rpb;
  static constexpr int ThreadsPerBlock = LogicalWarpSize * RowsPerBlock;
};

template <int tpb, int bpr>
struct ReductionThickPolicy {
  static constexpr int ThreadsPerBlock = tpb;
  static constexpr int BlocksPerRow    = bpr;
  static constexpr int BlockStride     = tpb * bpr;
};

}  // namespace raft::linalg::detail
