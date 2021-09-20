/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cub/cub.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <limits>
#include <raft/distance/pairwise_distance_base.cuh>
#include "processing.hpp"

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT>
DI void loadAllWarpQShmem(myWarpSelect &heapArr, Pair *shDumpKV, const IdxT m,
                          const unsigned int numOfNN) {
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
    const auto rowId =
      (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
    if (rowId < m) {
#pragma unroll
      for (int j = 0; j < heapArr[i]->kNumWarpQRegisters; ++j) {
        const int idx = j * warpSize + lid;
        if (idx < numOfNN) {
          Pair KVPair = shDumpKV[rowId * numOfNN + idx];
          heapArr[i]->warpV[j] = KVPair.key;
          heapArr[i]->warpK[j] = KVPair.value;
        }
      }
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect>
DI void loadWarpQShmem(myWarpSelect &heapArr, Pair *shDumpKV, const int rowId,
                       const unsigned int numOfNN) {
  const int lid = raft::laneId();
#pragma unroll
  for (int j = 0; j < heapArr->kNumWarpQRegisters; ++j) {
    const int idx = j * warpSize + lid;
    if (idx < numOfNN) {
      Pair KVPair = shDumpKV[rowId * numOfNN + idx];
      heapArr->warpV[j] = KVPair.key;
      heapArr->warpK[j] = KVPair.value;
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT>
DI void storeWarpQShmem(myWarpSelect &heapArr, Pair *shDumpKV, const IdxT rowId,
                        const unsigned int numOfNN) {
  const int lid = raft::laneId();

#pragma unroll
  for (int j = 0; j < heapArr->kNumWarpQRegisters; ++j) {
    const int idx = j * warpSize + lid;
    if (idx < numOfNN) {
      Pair otherKV = Pair(heapArr->warpV[j], heapArr->warpK[j]);
      shDumpKV[rowId * numOfNN + idx] = otherKV;
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT,
          typename OutT>
DI void storeWarpQGmem(myWarpSelect &heapArr, OutT *out_dists, IdxT *out_inds,
                       const IdxT m, const unsigned int numOfNN,
                       const IdxT starty) {
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
    const auto gmemRowId = starty + i * Policy::AccThRows;
    if (gmemRowId < m) {
#pragma unroll
      for (int j = 0; j < heapArr[i]->kNumWarpQRegisters; ++j) {
        const auto idx = j * warpSize + lid;
        if (idx < numOfNN) {
          out_dists[gmemRowId * numOfNN + idx] = heapArr[i]->warpK[j];
          out_inds[gmemRowId * numOfNN + idx] = (IdxT)heapArr[i]->warpV[j];
        }
      }
    }
  }
}

template <typename Policy, typename Pair, typename myWarpSelect, typename IdxT,
          typename OutT>
DI void loadPrevTopKsGmemWarpQ(myWarpSelect &heapArr, OutT *out_dists,
                               IdxT *out_inds, const IdxT m,
                               const unsigned int numOfNN, const IdxT starty) {
  const int lid = raft::laneId();
#pragma unroll
  for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
    const auto gmemRowId = starty + i * Policy::AccThRows;
    if (gmemRowId < m) {
#pragma unroll
      for (int j = 0; j < heapArr[i]->kNumWarpQRegisters; ++j) {
        const auto idx = j * warpSize + lid;
        if (idx < numOfNN) {
          heapArr[i]->warpK[j] = out_dists[gmemRowId * numOfNN + idx];
          heapArr[i]->warpV[j] = (uint32_t)out_inds[gmemRowId * numOfNN + idx];
        }
      }
      auto constexpr kLaneWarpKTop = heapArr[i]->kNumWarpQRegisters - 1;
      heapArr[i]->warpKTop =
        raft::shfl(heapArr[i]->warpK[kLaneWarpKTop], heapArr[i]->kLane);
    }
  }
}

template <typename Pair, int NumWarpQRegs, typename myWarpSelect>
DI void updateSortedWarpQ(myWarpSelect &heapArr, Pair *allWarpTopKs, int rowId,
                          int finalNumVals, int startId = 0) {
  constexpr uint32_t mask = 0xffffffffu;
  const int lid = raft::laneId();
  // calculate srcLane such that tid 0 -> 31, 1 -> 0,... 31 -> 30.
  // warp around 0 to 31 required for NN > 32
  const auto srcLane = (warpSize + (lid - 1)) & (warpSize - 1);

  for (int k = startId; k < finalNumVals; k++) {
    Pair KVPair = allWarpTopKs[rowId * (256) + k];
#pragma unroll
    for (int i = 0; i < NumWarpQRegs; i++) {
      unsigned activeLanes =
        __ballot_sync(mask, KVPair.value < heapArr->warpK[i]);
      if (activeLanes) {
        Pair tempKV;
        tempKV.value = raft::shfl(heapArr->warpK[i], srcLane);
        tempKV.key = raft::shfl(heapArr->warpV[i], srcLane);
        const auto firstActiveLane = __ffs(activeLanes);
        if (firstActiveLane == (lid + 1)) {
          heapArr->warpK[i] = KVPair.value;
          heapArr->warpV[i] = KVPair.key;
        } else if (activeLanes & ((uint32_t)1 << lid)) {
          heapArr->warpK[i] = tempKV.value;
          heapArr->warpV[i] = tempKV.key;
        }
        if (i == 0 && NumWarpQRegs > 1) {
          if (lid == 0) {
            heapArr->warpK[1] = tempKV.value;
            heapArr->warpV[1] = tempKV.key;
          }
          heapArr->warpK[1] = __shfl_up_sync(mask, heapArr->warpK[1], 1);
          heapArr->warpV[1] = __shfl_up_sync(mask, heapArr->warpV[1], 1);
          break;
        }
      }
    }
  }
}

template <bool useNorms, typename DataT, typename AccT, typename OutT,
          typename IdxT, typename Policy, typename CoreLambda,
          typename FinalLambda, int NumWarpQ, int NumThreadQ,
          bool usePrevTopKs = false, bool isRowMajor = true>
__global__ __launch_bounds__(Policy::Nthreads, 2) void fusedL2kNN(
  const DataT *x, const DataT *y, const DataT *_xn, const DataT *_yn,
  const IdxT m, const IdxT n, const IdxT k, const IdxT lda, const IdxT ldb,
  const IdxT ldd, CoreLambda core_op, FinalLambda fin_op, bool sqrt,
  unsigned int numOfNN, int *mutexes, OutT *out_dists, IdxT *out_inds) {
  extern __shared__ char smem[];

  typedef cub::KeyValuePair<uint32_t, AccT> Pair;
  constexpr auto identity = std::numeric_limits<AccT>::max();
  constexpr auto keyMax = std::numeric_limits<uint32_t>::max();
  constexpr auto Dir = false;
  typedef faiss::gpu::WarpSelect<
    AccT, uint32_t, Dir, faiss::gpu::Comparator<AccT>, NumWarpQ, NumThreadQ, 32>
    myWarpSelect;

  auto rowEpilog_lambda = [m, n, numOfNN, out_dists, out_inds,
                           mutexes] __device__(IdxT gridStrideY) {
    if (gridDim.x == 1) {
      return;
    }

    volatile int *mutex = mutexes;
    Pair *shDumpKV = (Pair *)(&smem[Policy::SmemSize]);
    const int lid = threadIdx.x % warpSize;
    const IdxT starty = gridStrideY + (threadIdx.x / Policy::AccThCols);

    //  0 -> consumer done consuming the buffer.
    // -1 -> consumer started consuming the buffer
    // -2 -> producer done filling the buffer
    // blockIdx.x -> prod started to fill the buffer
    if (blockIdx.x == 0) {
      auto cta_processed = 0;
      myWarpSelect heapArr1(identity, keyMax, numOfNN);
      myWarpSelect heapArr2(identity, keyMax, numOfNN);
      myWarpSelect *heapArr[] = {&heapArr1, &heapArr2};
      __syncthreads();

      loadAllWarpQShmem<Policy, Pair>(heapArr, &shDumpKV[0], m, numOfNN);

      while (cta_processed < gridDim.x - 1) {
        Pair otherKV[Policy::AccRowsPerTh];

        if (threadIdx.x == 0) {
          int32_t old = -3;
          while (old != -1) {
            old = atomicCAS((int *)&mutex[gridStrideY / Policy::Mblk], -2, -1);
          }
          __threadfence();
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId = starty + i * Policy::AccThRows;
          otherKV[i].value = identity;
          otherKV[i].key = keyMax;

          if (lid < numOfNN && rowId < m) {
            otherKV[i].value = out_dists[rowId * numOfNN + lid];
            otherKV[i].key = (uint32_t)out_inds[rowId * numOfNN + lid];
          }
        }
        __threadfence();

        if (threadIdx.x == 0) {
          mutex[gridStrideY / Policy::Mblk] = 0;
          __threadfence();
        }

        // Perform merging of otherKV with topk's across warp.
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId = starty + i * Policy::AccThRows;
          if (rowId < m) {
            heapArr[i]->add(otherKV[i].value, otherKV[i].key);
          }
        }

        cta_processed++;
      }
#pragma unroll
      for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
        const auto rowId = starty + i * Policy::AccThRows;
        if (rowId < m) {
          bool needSort = (heapArr[i]->numVals > 0);
          needSort = __any_sync(0xffffffff, needSort);
          if (needSort) {
            heapArr[i]->reduce();
          }
        }
      }
      storeWarpQGmem<Policy, Pair>(heapArr, out_dists, out_inds, m, numOfNN,
                                   starty);
    } else {
      if (threadIdx.x == 0) {
        int32_t old = -1;
        int32_t blkIdX = (int32_t)blockIdx.x;
        while (old != blkIdX) {
          old = atomicCAS((int *)&mutex[gridStrideY / Policy::Mblk], 0, blkIdX);
        }
        __threadfence();
      }
      __syncthreads();

#pragma unroll
      for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
        const auto rowId = starty + i * Policy::AccThRows;
        const auto shMemRowId =
          (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
        if (rowId < m) {
          for (int idx = lid; idx < numOfNN; idx += warpSize) {
            Pair KVPair = shDumpKV[shMemRowId * numOfNN + idx];
            out_dists[rowId * numOfNN + idx] = KVPair.value;
            out_inds[rowId * numOfNN + idx] = (IdxT)KVPair.key;
          }
        }
      }
      __threadfence();

      if (threadIdx.x == 0) {
        mutex[gridStrideY / Policy::Mblk] = -2;
        __threadfence();
      }
    }
  };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda =
    [numOfNN, sqrt, m, n, ldd, out_dists, out_inds] __device__(
      AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh], DataT * regxn,
      DataT * regyn, IdxT gridStrideX, IdxT gridStrideY) {
      if (sqrt) {
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
          for (int j = 0; j < Policy::AccColsPerTh; ++j) {
            acc[i][j] = raft::mySqrt(acc[i][j]);
          }
        }
      }
      Pair *shDumpKV = (Pair *)(&smem[Policy::SmemSize]);

      constexpr uint32_t mask = 0xffffffffu;
      const IdxT starty = gridStrideY + (threadIdx.x / Policy::AccThCols);
      const IdxT startx = gridStrideX + (threadIdx.x % Policy::AccThCols);
      const int lid = raft::laneId();

      myWarpSelect heapArr1(identity, keyMax, numOfNN);
      myWarpSelect heapArr2(identity, keyMax, numOfNN);
      myWarpSelect *heapArr[] = {&heapArr1, &heapArr2};
      if (usePrevTopKs) {
        if (gridStrideX == blockIdx.x * Policy::Nblk) {
          loadPrevTopKsGmemWarpQ<Policy, Pair>(heapArr, out_dists, out_inds, m,
                                               numOfNN, starty);
        }
      }

      if (gridStrideX > blockIdx.x * Policy::Nblk) {
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId =
            (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
          Pair tempKV = shDumpKV[(rowId * numOfNN) + numOfNN - 1];
          heapArr[i]->warpKTop = tempKV.value;
        }

        // total vals can atmost be 256, (32*8)
        int numValsWarpTopK[Policy::AccRowsPerTh];
        int anyWarpTopKs = 0;
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto rowId = starty + i * Policy::AccThRows;
          numValsWarpTopK[i] = 0;
          if (rowId < m) {
#pragma unroll
            for (int j = 0; j < Policy::AccColsPerTh; ++j) {
              const auto colId = startx + j * Policy::AccThCols;
              if (colId < ldd) {
                if (acc[i][j] < heapArr[i]->warpKTop) {
                  numValsWarpTopK[i]++;
                }
              }
            }
            anyWarpTopKs += numValsWarpTopK[i];
          }
        }
        anyWarpTopKs = __syncthreads_or(anyWarpTopKs > 0);
        if (anyWarpTopKs) {
          Pair *allWarpTopKs = (Pair *)(&smem[0]);
          uint32_t needScanSort[Policy::AccRowsPerTh];

#pragma unroll
          for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
            const auto gmemRowId = starty + i * Policy::AccThRows;
            needScanSort[i] = 0;
            if (gmemRowId < m) {
              int myVals = numValsWarpTopK[i];
              needScanSort[i] = __ballot_sync(mask, myVals > 0);
              if (needScanSort[i]) {
#pragma unroll
                for (unsigned int k = 1; k <= 16; k *= 2) {
                  const unsigned int n =
                    __shfl_up_sync(mask, numValsWarpTopK[i], k);
                  if (lid >= k) {
                    numValsWarpTopK[i] += n;
                  }
                }
              }
              // As each thread will know its total vals to write.
              // we only store its starting location.
              numValsWarpTopK[i] -= myVals;
            }

            if (needScanSort[i]) {
              const auto rowId =
                (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
              if (gmemRowId < m) {
                if (needScanSort[i] & ((uint32_t)1 << lid)) {
#pragma unroll
                  for (int j = 0; j < Policy::AccColsPerTh; ++j) {
                    const auto colId = startx + j * Policy::AccThCols;
                    if (colId < ldd) {
                      if (acc[i][j] < heapArr[i]->warpKTop) {
                        Pair otherKV = {colId, acc[i][j]};
                        allWarpTopKs[rowId * (256) + numValsWarpTopK[i]] =
                          otherKV;
                        numValsWarpTopK[i]++;
                      }
                    }
                  }
                }
                const int finalNumVals = raft::shfl(numValsWarpTopK[i], 31);
                loadWarpQShmem<Policy, Pair>(heapArr[i], &shDumpKV[0], rowId,
                                             numOfNN);
                updateSortedWarpQ<Pair, heapArr[i]->kNumWarpQRegisters>(
                  heapArr[i], &allWarpTopKs[0], rowId, finalNumVals);
              }
            }
          }
          __syncthreads();
#pragma unroll
          for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
            if (needScanSort[i]) {
              const auto rowId =
                (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
              const auto gmemRowId = starty + i * Policy::AccThRows;
              if (gmemRowId < m) {
                storeWarpQShmem<Policy, Pair>(heapArr[i], shDumpKV, rowId,
                                              numOfNN);
              }
            }
          }
        }
      } else {
#pragma unroll
        for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
          const auto gmemRowId = starty + i * Policy::AccThRows;
          const auto shMemRowId =
            (threadIdx.x / Policy::AccThCols) + i * Policy::AccThRows;
          if (gmemRowId < m) {
#pragma unroll
            for (int j = 0; j < Policy::AccColsPerTh; ++j) {
              const auto colId = startx + j * Policy::AccThCols;
              Pair otherKV = {keyMax, identity};
              if (colId < ldd) {
                otherKV.value = acc[i][j];
                otherKV.key = colId;
              }
              heapArr[i]->add(otherKV.value, otherKV.key);
            }

            bool needSort = (heapArr[i]->numVals > 0);
            needSort = __any_sync(mask, needSort);
            if (needSort) {
              heapArr[i]->reduce();
            }
            storeWarpQShmem<Policy, Pair>(heapArr[i], shDumpKV, shMemRowId,
                                          numOfNN);
          }
        }
      }

      if (((gridStrideX + Policy::Nblk * gridDim.x) > n) && gridDim.x == 1) {
        // This is last iteration of grid stride X
        loadAllWarpQShmem<Policy, Pair>(heapArr, &shDumpKV[0], m, numOfNN);
        storeWarpQGmem<Policy, Pair>(heapArr, out_dists, out_inds, m, numOfNN,
                                     starty);
      }
    };

  raft::distance::PairwiseDistances<useNorms, DataT, AccT, OutT, IdxT, Policy,
                                    CoreLambda, decltype(epilog_lambda),
                                    FinalLambda, decltype(rowEpilog_lambda),
                                    isRowMajor, false>
    obj(x, y, m, n, k, lda, ldb, ldd, _xn, _yn, nullptr, smem, core_op,
        epilog_lambda, fin_op, rowEpilog_lambda);
  obj.run();
}

template <typename DataT, typename AccT, typename OutT, typename IdxT,
          int VecLen, bool usePrevTopKs, bool isRowMajor>
void fusedL2kNNImpl(const DataT *x, const DataT *y, IdxT m, IdxT n, IdxT k,
                    IdxT lda, IdxT ldb, IdxT ldd, bool sqrt, OutT *out_dists,
                    IdxT *out_inds, IdxT numOfNN, cudaStream_t stream,
                    void *workspace, size_t &worksize) {
  typedef typename raft::linalg::Policy2x8<DataT, 1>::Policy RowPolicy;
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::ColPolicy ColPolicy;

  typedef typename std::conditional<true, RowPolicy, ColPolicy>::type KPolicy;

  ASSERT(isRowMajor, "Only Row major inputs are allowed");

  dim3 blk(KPolicy::Nthreads);
  // Accumulation operation lambda
  auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) {
    const auto diff = x - y;
    acc += diff * diff;
  };

  auto fin_op = [] __device__(AccT d_val, int g_d_idx) { return d_val; };

  typedef cub::KeyValuePair<uint32_t, AccT> Pair;

  if (isRowMajor) {
    constexpr auto fusedL2kNN32RowMajor =
      fusedL2kNN<false, DataT, AccT, OutT, IdxT, KPolicy, decltype(core_lambda),
                 decltype(fin_op), 32, 2, usePrevTopKs, true>;
    constexpr auto fusedL2kNN64RowMajor =
      fusedL2kNN<false, DataT, AccT, OutT, IdxT, KPolicy, decltype(core_lambda),
                 decltype(fin_op), 64, 3, usePrevTopKs, true>;

    auto fusedL2kNNRowMajor = fusedL2kNN32RowMajor;
    if (numOfNN <= 32) {
      fusedL2kNNRowMajor = fusedL2kNN32RowMajor;
    } else if (numOfNN <= 64) {
      fusedL2kNNRowMajor = fusedL2kNN64RowMajor;
    } else {
      ASSERT(numOfNN <= 64,
             "fusedL2kNN: num of nearest neighbors must be <= 64");
    }

    dim3 grid = raft::distance::launchConfigGenerator<KPolicy>(
      m, n, KPolicy::SmemSize, fusedL2kNNRowMajor);
    if (grid.x > 1) {
      const auto numMutexes = raft::ceildiv<int>(m, KPolicy::Mblk);
      if (workspace == nullptr || worksize < (sizeof(int32_t) * numMutexes)) {
        worksize = sizeof(int32_t) * numMutexes;
        return;
      } else {
        CUDA_CHECK(
          cudaMemsetAsync(workspace, 0, sizeof(int32_t) * numMutexes, stream));
      }
    }

    const auto sharedMemSize =
      KPolicy::SmemSize + (KPolicy::Mblk * numOfNN * sizeof(Pair));

    fusedL2kNNRowMajor<<<grid, blk, sharedMemSize, stream>>>(
      x, y, nullptr, nullptr, m, n, k, lda, ldb, ldd, core_lambda, fin_op, sqrt,
      (uint32_t)numOfNN, (int *)workspace, out_dists, out_inds);
  } else {
  }

  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename AccT, typename OutT, typename IdxT,
          bool usePrevTopKs, bool isRowMajor>
void fusedL2kNN(IdxT m, IdxT n, IdxT k, IdxT lda, IdxT ldb, IdxT ldd,
                const DataT *x, const DataT *y, bool sqrt, OutT *out_dists,
                IdxT *out_inds, IdxT numOfNN, cudaStream_t stream,
                void *workspace, size_t &worksize) {
  size_t bytesA = sizeof(DataT) * lda;
  size_t bytesB = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytesA % 16 == 0 && bytesB % 16 == 0) {
    fusedL2kNNImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), usePrevTopKs,
                   isRowMajor>(x, y, m, n, k, lda, ldb, ldd, sqrt, out_dists,
                               out_inds, numOfNN, stream, workspace, worksize);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    fusedL2kNNImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), usePrevTopKs,
                   isRowMajor>(x, y, m, n, k, lda, ldb, ldd, sqrt, out_dists,
                               out_inds, numOfNN, stream, workspace, worksize);
  } else {
    fusedL2kNNImpl<DataT, AccT, OutT, IdxT, 1, usePrevTopKs, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, sqrt, out_dists, out_inds, numOfNN, stream,
      workspace, worksize);
  }
}

/**
 * Compute the k-nearest neighbors using L2 unexpanded distance.

 * @tparam value_idx
 * @tparam value_t
 * @param[out] out_inds output indices array on device (size n_query_rows * k)
 * @param[out] out_dists output dists array on device (size n_query_rows * k)
 * @param[in] index input index array on device (size n_index_rows * D)
 * @param[in] query input query array on device (size n_query_rows * D)
 * @param[in] n_index_rows number of rows in index array
 * @param[in] n_query_rows number of rows in query array
 * @param[in] k number of closest neighbors to return
 * @param[in] rowMajorIndex are the index arrays in row-major layout?
 * @param[in] rowMajorQuery are the query array in row-major layout?
 * @param[in] stream stream to order kernel launch
 */
template <raft::distance::DistanceType distanceType, typename value_idx,
          typename value_t, bool usePrevTopKs>
void l2_unexpanded_knn(size_t D, value_idx *out_inds, value_t *out_dists,
                       const value_t *index, const value_t *query,
                       size_t n_index_rows, size_t n_query_rows, int k,
                       bool rowMajorIndex, bool rowMajorQuery,
                       cudaStream_t stream, void *workspace, size_t &worksize) {
  // Validate the input data
  ASSERT(k > 0, "l2Knn: k must be > 0");
  ASSERT(D > 0, "l2Knn: D must be > 0");
  ASSERT(n_index_rows > 0, "l2Knn: n_index_rows must be > 0");
  ASSERT(index, "l2Knn: index must be provided (passed null)");
  ASSERT(n_query_rows > 0, "l2Knn: n_query_rows must be > 0");
  ASSERT(query, "l2Knn: query must be provided (passed null)");
  ASSERT(out_dists, "l2Knn: out_dists must be provided (passed null)");
  ASSERT(out_inds, "l2Knn: out_inds must be provided (passed null)");
  // Currently we only support same layout for x & y inputs.
  ASSERT(rowMajorIndex == rowMajorQuery,
         "l2Knn: rowMajorIndex and rowMajorQuery should have same layout");

  bool sqrt = (distanceType == raft::distance::DistanceType::L2SqrtUnexpanded);

  if (rowMajorIndex) {
    value_idx lda = D, ldb = D, ldd = n_index_rows;
    fusedL2kNN<value_t, value_t, value_t, value_idx, usePrevTopKs, true>(
      n_query_rows, n_index_rows, D, lda, ldb, ldd, query, index, sqrt,
      out_dists, out_inds, k, stream, workspace, worksize);
  } else {
    // TODO: Add support for column major layout
  }
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
