/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace raft::distance::detail::ops {

/**
 * @brief the Russell Rao distance matrix calculation
 *
 * It computes the following equation:
 *
 *  c_ij = (k - (sum_k x_ik * y_kj)) / k
 */
template <typename DataType, typename AccType, typename IdxType>
struct russel_rao_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  IdxT k;
  const float one_over_k;

  russel_rao_distance_op(IdxT k_) noexcept : k(k_), one_over_k(1.0f / k_) {}

  // Load norms of input data
  static constexpr bool use_norms = false;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize;
  }

  DI void core(AccT& acc, DataT& x, DataT& y) const { acc += x * y; };

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
        acc[i][j] = (k - acc[i][j]) * one_over_k;
      }
    }
  }
};

}  // namespace raft::distance::detail::ops
