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

/**
 * @brief the Hamming Unexpanded distance matrix calculation
 *  It computes the following equation:
 *
 *    c_ij = sum_k (x_ik != y_kj) / k
 */
template <typename IdxT_struct>
struct hamming_distance_op {
  IdxT_struct k;

  hamming_distance_op(IdxT_struct k_) noexcept : k(k_) {}

  // Load norms of input data
  static constexpr bool use_norms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

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
    acc += (x != y);
  };

  template <typename Policy, typename AccT, typename DataT, typename IdxT>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 DataT* regxn,
                 DataT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    const DataT one_over_k = DataT(1.0) / k;
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        acc[i][j] *= one_over_k;
      }
    }
  }
};

}  // namespace raft::distance::detail::ops
