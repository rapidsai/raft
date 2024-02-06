/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/linalg/contractions.cuh>
#include <raft/util/device_utils.cuh>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

template <typename DataT,
          typename IdxT,
          typename Policy,
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, Policy>>
struct EpsUnexpL2SqNeighborhood : public BaseClass {
 private:
  typedef Policy P;

  bool* adj;
  DataT eps;
  IdxT* vd;

  char* smem;  // for final reductions

  DataT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  DI EpsUnexpL2SqNeighborhood(bool* _adj,
                              IdxT* _vd,
                              const DataT* _x,
                              const DataT* _y,
                              IdxT _m,
                              IdxT _n,
                              IdxT _k,
                              DataT _eps,
                              char* _smem)
    : BaseClass(_x, _y, _m, _n, _k, _smem), adj(_adj), eps(_eps), vd(_vd), smem(_smem)
  {
  }

  DI void run()
  {
    prolog();
    loop();
    epilog();
  }

 private:
  DI void prolog()
  {
    this->ldgXY(IdxT(blockIdx.x) * P::Mblk, IdxT(blockIdx.y) * P::Nblk, 0);
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }
    this->stsXY();
    __syncthreads();
    this->switch_write_buffer();
  }

  DI void loop()
  {
    for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
      this->ldgXY(IdxT(blockIdx.x) * P::Mblk, IdxT(blockIdx.y) * P::Nblk, kidx);
      accumulate();  // on the previous k-block
      this->stsXY();
      __syncthreads();
      this->switch_write_buffer();
      this->switch_read_buffer();
    }
    accumulate();  // last iteration
  }

  DI void epilog()
  {
    IdxT startx = blockIdx.x * P::Mblk + this->accrowid;
    IdxT starty = blockIdx.y * P::Nblk + this->acccolid;
    auto lid    = raft::laneId();
    IdxT sums[P::AccRowsPerTh];
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      sums[i]  = 0;
      auto xid = startx + i * P::AccThRows;
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto yid      = starty + j * P::AccThCols;
        auto is_neigh = acc[i][j] <= eps;
        ///@todo: fix uncoalesced writes using shared mem
        if (xid < this->m && yid < this->n) {
          adj[xid * this->n + yid] = is_neigh;
          sums[i] += is_neigh;
        }
      }
    }
    // perform reduction of adjacency values to compute vertex degrees
    if (vd != nullptr) { updateVertexDegree(sums); }
  }

  DI void accumulate()
  {
#pragma unroll
    for (int ki = 0; ki < P::Kblk; ki += P::Veclen) {
      this->ldsXY(ki);
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < P::Veclen; ++v) {
            auto diff = this->regx[i][v] - this->regy[j][v];
            acc[i][j] += diff * diff;
          }
        }
      }
    }
  }

  DI void updateVertexDegree(IdxT (&sums)[P::AccRowsPerTh])
  {
    __syncthreads();  // so that we can safely reuse smem
    int gid       = this->accrowid;
    int lid       = this->acccolid;
    auto cidx     = IdxT(blockIdx.x) * P::Mblk + gid;
    IdxT totalSum = 0;
    // update the individual vertex degrees
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      // P::AccThCols neighboring threads need to reduce
      // -> we have P::Nblk/P::AccThCols individual reductions
      auto cid = cidx + i * P::AccThRows;
      sums[i]  = raft::logicalWarpReduce<P::AccThCols>(sums[i], raft::add_op());
      if (lid == 0 && cid < this->m) {
        atomicUpdate(cid, sums[i]);
        totalSum += sums[i];
      }
      __syncthreads();  // for safe smem reuse
    }
    // update the total edge count
    totalSum = raft::blockReduce<IdxT>(totalSum, smem);
    if (threadIdx.x == 0) { atomicUpdate(this->m, totalSum); }
  }

  DI void atomicUpdate(IdxT addrId, IdxT val)
  {
    if (sizeof(IdxT) == 4) {
      raft::myAtomicAdd<unsigned>((unsigned*)(vd + addrId), val);
    } else if (sizeof(IdxT) == 8) {
      raft::myAtomicAdd<unsigned long long>((unsigned long long*)(vd + addrId), val);
    }
  }
};  // struct EpsUnexpL2SqNeighborhood

template <typename DataT, typename IdxT, typename Policy>
__launch_bounds__(Policy::Nthreads, 2) RAFT_KERNEL epsUnexpL2SqNeighKernel(
  bool* adj, IdxT* vd, const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k, DataT eps)
{
  extern __shared__ char smem[];
  EpsUnexpL2SqNeighborhood<DataT, IdxT, Policy> obj(adj, vd, x, y, m, n, k, eps, smem);
  obj.run();
}

template <typename DataT, typename IdxT, int VecLen>
void epsUnexpL2SqNeighImpl(bool* adj,
                           IdxT* vd,
                           const DataT* x,
                           const DataT* y,
                           IdxT m,
                           IdxT n,
                           IdxT k,
                           DataT eps,
                           cudaStream_t stream)
{
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(raft::ceildiv<int>(m, Policy::Mblk), raft::ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);
  epsUnexpL2SqNeighKernel<DataT, IdxT, Policy>
    <<<grid, blk, Policy::SmemSize, stream>>>(adj, vd, x, y, m, n, k, eps);
  RAFT_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Computes epsilon neighborhood for the L2-Squared distance metric
 *
 * @tparam DataT   IO and math type
 * @tparam IdxT    Index type
 *
 * @param[out] adj    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  x      first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  y      second matrix [row-major] [on device] [dim = n x k]
 * @param[in]  eps    defines epsilon neighborhood radius (should be passed as
 *                    squared as we compute L2-squared distance in this method)
 * @param[in]  fop    device lambda to do any other custom functions
 * @param[in]  stream cuda stream
 */
template <typename DataT, typename IdxT>
void epsUnexpL2SqNeighborhood(bool* adj,
                              IdxT* vd,
                              const DataT* x,
                              const DataT* y,
                              IdxT m,
                              IdxT n,
                              IdxT k,
                              DataT eps,
                              cudaStream_t stream)
{
  if (vd != nullptr) { RAFT_CUDA_TRY(cudaMemsetAsync(vd, 0, (m + 1) * sizeof(IdxT), stream)); }

  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 16 / sizeof(DataT)>(adj, vd, x, y, m, n, k, eps, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 8 / sizeof(DataT)>(adj, vd, x, y, m, n, k, eps, stream);
  } else {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 1>(adj, vd, x, y, m, n, k, eps, stream);
  }
}
}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
