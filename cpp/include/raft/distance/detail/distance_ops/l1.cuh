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

namespace raft::distance::detail::ops {

// Describes the computation the l1 distance
struct l1_distance_op {
  // Do not load norms of data, the computation of L1 distance does not use them.
  static constexpr bool use_norms = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy, typename DataT>
  constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  template <typename AccT, typename DataT>
  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    acc += raft::abs(x - y);
  };

  template <typename Policy, typename AccT, typename DataT, typename IdxT>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 DataT* regxn,
                 DataT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    return;
  };
};

}  // namespace raft::distance::detail::ops
