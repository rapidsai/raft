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

#include <raft/util/cuda_dev_essentials.cuh>  // DI

namespace raft::distance::detail::ops {

// Epilogue operator for CUTLASS based kernel
template <typename DataT, typename AccT>
struct cosine_cutlass_op {
  __device__ cosine_cutlass_op() noexcept {}
  __device__ AccT operator()(DataT& aNorm, const DataT& bNorm, DataT& accVal) const noexcept
  {
    return static_cast<AccT>(1.0) - static_cast<AccT>(accVal / (aNorm * bNorm));
  }
  __device__ AccT operator()(DataT aData) const noexcept { return aData; }
};

/**
 * @brief the expanded cosine distance matrix calculation
 *
 * It computes the following equation:
 *
 * d(x, y) = 1 - (x ⋅ y) / ( ||x||_2 ||y||_2)
 */
template <typename DataType, typename AccType, typename IdxType>
struct cosine_distance_op {
  using DataT = DataType;
  using AccT  = AccType;
  using IdxT  = IdxType;

  // Load norms of input data
  static constexpr bool use_norms = true;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy>
  static constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize + ((Policy::Mblk + Policy::Nblk) * sizeof(DataT));
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
        acc[i][j] = 1.0 - (acc[i][j] / (regxn[i] * regyn[j]));
      }
    }
  }

  constexpr cosine_cutlass_op<DataT, AccT> get_cutlass_op() const
  {
    return cosine_cutlass_op<DataT, AccT>();
  }
};

}  // namespace raft::distance::detail::ops
