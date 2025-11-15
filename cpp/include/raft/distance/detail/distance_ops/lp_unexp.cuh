/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <raft/core/operators.hpp>            // raft::pow, raft::abs
#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace raft::distance::detail::ops {

/**
 * @brief the unexpanded Lp (Minkowski) distance matrix calculation
 *
 * It computes the following equation:
 *
 *   c_ij = (sum_k |x_ik - y_jk|^p)^(1/p)
 */
template <typename DataType, typename AccType, typename IdxType>
struct lp_unexp_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  DataT p;

  lp_unexp_distance_op(DataT p_) noexcept : p(p_) {}

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
    const auto diff = raft::abs(x - y);
    acc += raft::pow(diff, p);
  };

  template <typename Policy>
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
