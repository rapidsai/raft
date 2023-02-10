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
#include <raft/util/cuda_utils.cuh>

namespace raft::distance::detail::ops {

// Describes the computation the minkowski distance

template <typename DataT_struct>
struct minkowski_distance_op {
  DataT_struct p;

  minkowski_distance_op(DataT_struct p_) noexcept : p(p_) { }

  // Load norms of input data
  static constexpr bool use_norms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = true;

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
    const auto diff = raft::abs(x - y);
    acc += raft::pow(diff, p);
  };

  template <typename Policy, typename AccT, typename DataT, typename IdxT>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 DataT* regxn,
                 DataT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    const auto one_over_p = 1.0f / p;
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        acc[i][j] = raft::pow(acc[i][j], one_over_p);
      }
    }
  }
};

}  // namespace raft::distance::detail::ops
