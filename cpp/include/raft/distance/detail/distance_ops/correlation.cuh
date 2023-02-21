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

// Describes the computation the correlation distance


template <typename DataT_struct, typename IdxT_struct>
struct correlation_distance_op {
  const DataT_struct* x2n;
  const DataT_struct* y2n;
  IdxT_struct m;
  IdxT_struct n;
  IdxT_struct k;

  correlation_distance_op(
    bool is_row_major,
    const DataT_struct* x2n_,
    const DataT_struct* y2n_,
    IdxT_struct m_,
    IdxT_struct n_,
    IdxT_struct k_
  ) noexcept
  : x2n(x2n_),
    y2n(y2n_),
    m(m_),
    n(n_),
    k(k_)
  {
    // The distance op is typically created before the row-major/col-major
    // swapping has been done. So we do it here.
    if (!is_row_major) {
      std::swap<const DataT_struct*>(x2n, y2n);
      std::swap(m, n);
    }
  }


  // Load norms of input data
  static constexpr bool use_norms = true;
  // Whether the core function requires so many instructions that it makes sense
  // to reduce loop unrolling, etc. We do this to keep compile times in check.
  static constexpr bool expensive_inner_loop = false;

  // Size of shared memory. This is normally decided by the kernel policy, but
  // some ops such as correlation_distance_op use more.
  template <typename Policy, typename DataT>
  constexpr size_t shared_mem_size()
  {
    return Policy::SmemSize + (2 * (Policy::Mblk + Policy::Nblk) * sizeof(DataT));
  }

  template <typename AccT, typename DataT>
  DI void core(AccT& acc, DataT& x, DataT& y) const
  {
    acc += x * y;
  };

  template <typename Policy, typename AccT, typename DataT, typename IdxT>
  DI void epilog(AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                 DataT* regxn,
                 DataT* regyn,
                 IdxT gridStrideX,
                 IdxT gridStrideY) const
  {
    // Note how we can sneakily get a pointer to shared memory here, to store
    // more data. If the implementation of PairwiseDistanceMatKernel ever
    // changes, this will be where we find the bugs.
    extern __shared__ char smem[];

    DataT regx2n[Policy::AccRowsPerTh], regy2n[Policy::AccColsPerTh];

    DataT* sx2Norm =
      (DataT*)(&smem[Policy::SmemSize + (Policy::Mblk + Policy::Nblk) * sizeof(DataT)]);
    DataT* sy2Norm = (&sx2Norm[Policy::Mblk]);

    // Load x & y norms required by this threadblock in shmem buffer
    if (gridStrideX == blockIdx.x * Policy::Nblk) {
      for (int i = threadIdx.x; i < Policy::Mblk; i += Policy::Nthreads) {
        auto idx   = gridStrideY + i;
        sx2Norm[i] = idx < m ? x2n[idx] : 0;
      }
    }

    for (int i = threadIdx.x; i < Policy::Nblk; i += Policy::Nthreads) {
      auto idx   = gridStrideX + i;
      sy2Norm[i] = idx < n ? y2n[idx] : 0;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
      regx2n[i] = sx2Norm[i * Policy::AccThRows + (threadIdx.x / Policy::AccThCols)];
    }
#pragma unroll
    for (int i = 0; i < Policy::AccColsPerTh; ++i) {
      regy2n[i] = sy2Norm[i * Policy::AccThCols + (threadIdx.x % Policy::AccThCols)];
    }

#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        auto numer   = k * acc[i][j] - (regxn[i] * regyn[j]);
        auto Q_denom = k * regx2n[i] - (regxn[i] * regxn[i]);
        auto R_denom = k * regy2n[j] - (regyn[j] * regyn[j]);

        acc[i][j] = 1 - (numer / raft::sqrt(Q_denom * R_denom));
      }
    }
  }
};

}  // namespace raft::distance::detail::ops
