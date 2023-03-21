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
#include <raft/core/operators.hpp>            // raft::log
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace raft::distance::detail::ops {

// Describes the computation the jensen_shannon distance

/**
 * @brief the Jensen Shannon distance matrix calculation
 *
 * It computes the following equation:
 *
 * c_ij = sqrt(0.5 * sum( -x_i * (log(0.5 * (x_i + y_i)) - log(x_i))
 *       + (-y_i * (log(0.5 * (x_i + y_i)) - log(y_i)))))
 */
template <typename DataType, typename AccType, typename IdxType>
struct jensen_shannon_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  // Load norms of input data
  static constexpr bool use_norms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = true;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    const DataT m     = 0.5f * (x + y);
    const bool m_zero = (m == 0);
    const auto logM   = (!m_zero) * raft::log(m + m_zero);

    const bool x_zero = (x == 0);
    const bool y_zero = (y == 0);
    acc += (-x * (logM - raft::log(x + x_zero))) + (-y * (logM - raft::log(y + y_zero)));
  };

  template <typename Policy>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 DataT* regxn,
                 DataT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        acc[i][j] = raft::sqrt(0.5 * acc[i][j]);
      }
    }
  }
};

}  // namespace raft::distance::detail::ops
