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

// Describes the computation the l2 unexpanded distance

template <bool sqrt>
struct l2_unexp_generic_distance_op {
  // Do not load norms of data, the computation of L1 distance does not use them.
  static constexpr bool use_norms = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  template <typename AccT, typename DataT>
  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    const auto diff = x - y;
    acc += diff * diff;
  };

  template <typename Policy, typename AccT, typename DataT, typename IdxT>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 DataT* regxn,
                 DataT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    if constexpr (sqrt) {
#pragma unroll
      for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < Policy::AccColsPerTh; ++j) {
          acc[i][j] = raft::sqrt(acc[i][j]);
        }
      }
    }
  };
};


// Define distance ops with and without square root computation.
using l2_unexp_distance_op = l2_unexp_generic_distance_op<false>;
using l2_unexp_sqrt_distance_op = l2_unexp_generic_distance_op<true>;


}  // namespace raft::distance::detail::ops
