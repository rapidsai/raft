/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <raft/distance/pairwise_distance_base.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include "processing.hpp"
#include <limits>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

template <bool useNorms, typename DataT, typename AccT, typename OutT,
          typename IdxT, typename Policy, typename CoreLambda,
          typename FinalLambda, bool isRowMajor = true>
__global__ __launch_bounds__( Policy::Nthreads, 2) void fusedL2kNN(
                      const DataT* x, const DataT* y,
                      const DataT* _xn, const DataT* _yn, IdxT m,
                      IdxT n, IdxT k, IdxT lda, IdxT ldb,
                      IdxT ldd,
                      CoreLambda core_op,
                      FinalLambda fin_op,
                      bool sqrt,
                      unsigned int numOfNN,
                      int *mutexes,
                      OutT *out_dists,
                      IdxT *out_inds) {
  extern __shared__ char smem[];

  typedef cub::KeyValuePair<uint32_t, AccT> Pair;
  constexpr auto identity = std::numeric_limits<AccT>::max();
  constexpr auto keyMax = std::numeric_limits<uint32_t>::max();
  constexpr auto NumWarpQ = 32;
  constexpr auto NumThreadQ = 2;
  constexpr auto Dir = false;
  typedef  faiss::gpu::WarpSelect<AccT, uint32_t, Dir,
            faiss::gpu::Comparator<AccT>, NumWarpQ,
          NumThreadQ, 32> myWarpSelect;

  auto rowEpilog_lambda = [m, numOfNN, out_dists, out_inds, mutexes] __device__(IdxT gridStrideY) {

    if (gridDim.x == 1) {
      return;
    }

    volatile int *mutex = mutexes;
    Pair *shDumpKV = (Pair*)(&smem[Policy::SmemSize]);
    // make the thread distribution to be 8x32 from previous 16x16
    constexpr auto newAccRowsPerTh = Policy::AccRowsPerTh;
    constexpr auto newAccThRows = Policy::AccThRows;
    constexpr auto newAccThCols = Policy::AccThCols;

    const int lid = threadIdx.x % warpSize;
    const auto starty = gridStrideY + (threadIdx.x / newAccThCols);

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
#pragma unroll
      for (int i = 0; i < newAccRowsPerTh; ++i) {
        const auto rowId = (threadIdx.x / newAccThCols) + i * newAccThRows;
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
          auto constexpr kLaneWarpKTop = heapArr[i]->kNumWarpQRegisters - 1;
          heapArr[i]->warpKTop = raft::shfl(heapArr[i]->warpK[kLaneWarpKTop], heapArr[i]->kLane);
        }
      }

      while (cta_processed < gridDim.x-1) {
        Pair otherKV[newAccRowsPerTh];

        if (threadIdx.x == 0) {
          int32_t old = -3;
          while (old != -1) {
            old = atomicCAS((int*)&mutex[gridStrideY / Policy::Mblk], -2, -1);
          }
          __threadfence();
        }
        __syncthreads();

#pragma unroll
        for (int i = 0; i < newAccRowsPerTh; ++i) {
          const auto rowId = starty + i * newAccThRows;
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
        for (int i = 0; i < newAccRowsPerTh; ++i) {
          const auto rowId = starty + i * newAccThRows;
          if (rowId < m) {
            heapArr[i]->add(otherKV[i].value, otherKV[i].key);
          }
        }

        cta_processed++;
      }
#pragma unroll
      for (int i = 0; i < newAccRowsPerTh; ++i) {
        const auto rowId = starty + i * newAccThRows;
        if (rowId < m) {
          bool needSort = (heapArr[i]->numVals > 0);
          needSort = __any_sync(0xffffffff, needSort);
          if (needSort) {
            heapArr[i]->reduce();
          }
#pragma unroll
          for (int j = 0; j < heapArr[i]->kNumWarpQRegisters; ++j) {
            const auto idx = j * warpSize + lid;
            if (idx < numOfNN) {
              out_dists[rowId * numOfNN + idx] = heapArr[i]->warpK[j];
              out_inds[rowId * numOfNN + idx] = (IdxT)heapArr[i]->warpV[j];
            }
          }
        }
      }
    } else {
      if (threadIdx.x == 0) {
        int32_t old = -1;
        int32_t blkIdX = (int32_t)blockIdx.x;
        while (old != blkIdX) {
          old = atomicCAS((int*)&mutex[gridStrideY / Policy::Mblk], 0, blkIdX);
        }
        __threadfence();
      }
      __syncthreads();

#pragma unroll
      for (int i = 0; i < newAccRowsPerTh; ++i) {
        const auto rowId = starty + i * newAccThRows;
        const auto shMemRowId = (threadIdx.x / newAccThCols) + i * newAccThRows;
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
  auto epilog_lambda = [numOfNN, sqrt, m, n, ldd, out_dists, out_inds] __device__(
                        AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                        DataT *regxn, DataT *regyn, IdxT gridStrideX,
                        IdxT gridStrideY) {
    if (sqrt) {
#pragma unroll
      for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < Policy::AccColsPerTh; ++j) {
          acc[i][j] = raft::mySqrt(acc[i][j]);
        }
      }
    }
    Pair *shDumpKV = (Pair*)(&smem[Policy::SmemSize]);

    constexpr auto newAccRowsPerTh = Policy::AccRowsPerTh;
    constexpr auto newAccThRows = Policy::AccThRows;
    constexpr auto newAccColsPerTh = Policy::AccColsPerTh;
    constexpr auto newAccThCols = Policy::AccThCols;
    constexpr uint32_t mask = 0xffffffffu;
    const auto starty = gridStrideY + (threadIdx.x / newAccThCols);
    const auto startx = gridStrideX + (threadIdx.x % newAccThCols);
    const int lid = raft::laneId();

    myWarpSelect heapArr1(identity, keyMax, numOfNN);
    myWarpSelect heapArr2(identity, keyMax, numOfNN);
    myWarpSelect *heapArr[] = {&heapArr1, &heapArr2};

    if (gridStrideX > blockIdx.x * Policy::Nblk) {
#pragma unroll
      for (int i = 0; i < newAccRowsPerTh; ++i) {
        const auto rowId = (threadIdx.x / newAccThCols) + i * newAccThRows;
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
          auto constexpr kLaneWarpKTop = heapArr[i]->kNumWarpQRegisters - 1;
          heapArr[i]->warpKTop = raft::shfl(heapArr[i]->warpK[kLaneWarpKTop], heapArr[i]->kLane);
        }
      }
    }

    int anyWarpTopKs = 1;
    if (gridStrideX > blockIdx.x * Policy::Nblk) {
      // total vals can atmost be 256, (32*8)
      int numValsWarpTopK[newAccRowsPerTh];
      int checkIfAny = 0;
#pragma unroll
      for (int i = 0; i < newAccRowsPerTh; ++i) {
        const auto rowId = starty + i * newAccThRows;
        numValsWarpTopK[i] = 0;
        if (rowId < m) {
#pragma unroll
          for (int j = 0; j < newAccColsPerTh; ++j) {
            const auto colId = startx + j * newAccThCols;
            if (colId < ldd) {
              if (acc[i][j] < heapArr[i]->warpKTop) {
                numValsWarpTopK[i]++;
              }
            }
          }
          int myVals = numValsWarpTopK[i];
#pragma unroll
          for (unsigned int k = 1; k <= 16; k *= 2) {
            const unsigned int n = __shfl_up_sync(mask, numValsWarpTopK[i], k);
            if (lid >= k) {
              numValsWarpTopK[i] += n;
            }
          }
          // As each thread will know its total vals to write.
          // we only store its starting location.
          numValsWarpTopK[i] -= myVals;
        }
        checkIfAny += numValsWarpTopK[i];
      }
      anyWarpTopKs = __syncthreads_or(checkIfAny > 0);
      if (anyWarpTopKs) {
        Pair *allWarpTopKs = (Pair*)(&smem[0]);
        //
        bool needSort = (checkIfAny > 0);
        needSort = __any_sync(mask, needSort);

        if (needSort) {
#pragma unroll
          for (int i = 0; i < newAccRowsPerTh; ++i) {
            const auto rowId = (threadIdx.x / newAccThCols) + i * newAccThRows;
            const auto gmemRowId = starty + i * newAccThRows;
            if (gmemRowId < m) {
#pragma unroll
              for (int j = 0; j < newAccColsPerTh; ++j) {
                const auto colId = startx + j * newAccThCols;
                if (colId < ldd) {
                  if (acc[i][j] < heapArr[i]->warpKTop) {
                    Pair otherKV = {colId, acc[i][j]};
                    allWarpTopKs[rowId * (256) + numValsWarpTopK[i]] = otherKV;
                    numValsWarpTopK[i]++;
                  }
                }
              }
            }
          }
          __syncwarp();
#pragma unroll
          for (int i = 0; i < newAccRowsPerTh; ++i) {
            const auto rowId = (threadIdx.x / newAccThCols) + i * newAccThRows;
            const auto gmemRowId = starty + i * newAccThRows;
            const int finalNumVals = raft::shfl(numValsWarpTopK[i], 31);
            int limit = faiss::gpu::utils::roundDown(finalNumVals, warpSize);
            if (gmemRowId < m) {
              int j = lid;
              for (; j < limit; j += warpSize) {
                Pair otherKV = allWarpTopKs[rowId * (256) + j];
                heapArr[i]->add(otherKV.value, otherKV.key);
              }
              if (j < finalNumVals) {
                Pair otherKV = allWarpTopKs[rowId * (256) + j];
                heapArr[i]->addThreadQ(otherKV.value, otherKV.key);
              }
            }
          }
        }
        __syncthreads();
      }
    } else {
#pragma unroll
      for (int i = 0; i < newAccRowsPerTh; ++i) {
        const auto rowId = starty + i * newAccThRows;
        if (rowId < m) {
  #pragma unroll
          for (int j = 0; j < newAccColsPerTh; ++j) {
            const auto colId = startx + j * newAccThCols;
            Pair otherKV = {keyMax, identity};
            if (colId < ldd) {
              otherKV.value = acc[i][j];
              otherKV.key = colId;
            }
            heapArr[i]->add(otherKV.value, otherKV.key);
          }
        }
      }
    }
#pragma unroll
    for (int i = 0; i < newAccRowsPerTh; ++i) {
      const auto rowId = (threadIdx.x / newAccThCols) + i * newAccThRows;
      const auto gmemRowId = starty + i * newAccThRows;
      if (gmemRowId < m) {
        if (anyWarpTopKs) {
          bool needSort = (heapArr[i]->numVals > 0);
          needSort = __any_sync(mask, needSort);
          if (needSort) {
            heapArr[i]->reduce();
          }
        }
        if (((gridStrideX + Policy::Nblk * gridDim.x) > n) && gridDim.x == 1) {
          // This is last iteration of grid stride X
#pragma unroll
          for (int j = 0; j < heapArr[i]->kNumWarpQRegisters; ++j) {
            const auto idx = j * warpSize + lid;
            if (idx < numOfNN) {
              out_dists[gmemRowId * numOfNN + idx] = heapArr[i]->warpK[j];
              out_inds[gmemRowId * numOfNN + idx] = (IdxT)heapArr[i]->warpV[j];
            }
          }
        } else {
          if (anyWarpTopKs) {
#pragma unroll
            for (int j = 0; j < heapArr[i]->kNumWarpQRegisters; ++j) {
              const int idx = j * warpSize + lid;
              if (idx < numOfNN) {
                Pair otherKV = {heapArr[i]->warpV[j], heapArr[i]->warpK[j]};
                shDumpKV[rowId * numOfNN + idx] = otherKV;
              }
            }
          }
        }
      }
    }
  };

  raft::distance::PairwiseDistances<useNorms, DataT, AccT, OutT, IdxT,
        Policy, CoreLambda, decltype(epilog_lambda), FinalLambda,
        decltype(rowEpilog_lambda), isRowMajor, false>
    obj(x, y, m, n, k, lda, ldb, ldd, _xn, _yn, nullptr, smem, core_op,
        epilog_lambda, fin_op, rowEpilog_lambda);
  obj.run();
}



template <typename DataT, typename AccT, typename OutT, typename IdxT,
          int VecLen, bool isRowMajor>
void fusedL2kNNImpl(const DataT *x, const DataT *y, IdxT m, IdxT n, IdxT k,
                    IdxT lda, IdxT ldb, IdxT ldd, bool sqrt, OutT *out_dists,
                    IdxT *out_inds, IdxT numOfNN, cudaStream_t stream, 
                    std::shared_ptr<deviceAllocator> allocator,
                    void *workspace, size_t worksize) {
  typedef typename raft::linalg::Policy1x1<DataT, 1>::Policy RowPolicy;
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::ColPolicy ColPolicy;

  typedef
    typename std::conditional<true, RowPolicy, ColPolicy>::type KPolicy;

  ASSERT(isRowMajor, "Only Row major inputs are allowed");

/* dim3 grid(raft::ceildiv<int>(n, KPolicy::Nblk),
           raft::ceildiv<int>(m, KPolicy::Mblk));*/
  dim3 blk(KPolicy::Nthreads);
  // Accumulation operation lambda
  auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) {
    const auto diff = x - y;
    acc += diff * diff;
  };

  auto fin_op = [] __device__(AccT d_val, int g_d_idx) {
    return d_val;
  };

  if (isRowMajor) {
    auto fusedL2kNNRowMajor = fusedL2kNN<false, DataT, AccT, OutT, IdxT, KPolicy,
                              decltype(core_lambda), decltype(fin_op), true>;
    dim3 grid = raft::distance::launchConfigGenerator<KPolicy>(m, n, KPolicy::SmemSize, fusedL2kNNRowMajor);
    int *mutexes = (int*)workspace;
    raft::mr::device::buffer<int> d_mutexes(allocator, stream, 0);
    // initialize d_mutexes with 0 to allow producers to fill the buffer.
    if (grid.x > 1) {
      if (workspace == NULL || worksize < (sizeof(int32_t) * raft::ceildiv<int>(m, KPolicy::Mblk))) {
        d_mutexes.resize(raft::ceildiv<int>(m, KPolicy::Mblk));
        mutexes = d_mutexes.data();
      }
      CUDA_CHECK(cudaMemsetAsync(mutexes, 0, sizeof(int32_t) * raft::ceildiv<int>(m, KPolicy::Mblk), stream));
    }

    typedef cub::KeyValuePair<uint32_t, AccT> Pair;
    const auto sharedMemSize = KPolicy::SmemSize + (KPolicy::Mblk * numOfNN * sizeof(Pair));

    //printf("sqrt = %d grid.x = %d grid.y = %d \n", (int)sqrt, (int)grid.x, (int)grid.y);
    fusedL2kNNRowMajor<<<grid, blk, sharedMemSize, stream>>>(
        x, y, nullptr, nullptr, m, n, k, lda, ldb, ldd, core_lambda,
        fin_op, sqrt, (uint32_t)numOfNN, mutexes, out_dists, out_inds);
  } else {
  }

  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename AccT, typename OutT, typename IdxT,
          bool isRowMajor>
void fusedL2kNN(IdxT m, IdxT n, IdxT k, IdxT lda, IdxT ldb, IdxT ldd,
                  const DataT *x, const DataT *y, bool sqrt, OutT *out_dists,
                  IdxT *out_inds, IdxT numOfNN, cudaStream_t stream, 
                  std::shared_ptr<deviceAllocator> allocator,
                  void *workspace, size_t worksize) {
  //printf("D = %ld K = %ld m = %ld n = %ld \n", k, numOfNN, m, n);
  size_t bytesA = sizeof(DataT) * lda;
  size_t bytesB = sizeof(DataT) * ldb;
  if (16 % sizeof(DataT) == 0 && bytesA % 16 == 0 && bytesB % 16 == 0) {
    fusedL2kNNImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), isRowMajor>
            (x, y, m, n, k, lda, ldb, ldd, sqrt, out_dists, out_inds, numOfNN,
              stream, allocator, workspace, worksize);
  } else if (8 % sizeof(DataT) == 0 && bytesA % 8 == 0 && bytesB % 8 == 0) {
    fusedL2kNNImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), isRowMajor>
        (x, y, m, n, k, lda, ldb, ldd, sqrt, out_dists, out_inds, numOfNN,
          stream, allocator, workspace, worksize);
  } else {
    fusedL2kNNImpl<DataT, AccT, OutT, IdxT, 1, isRowMajor>(
      x, y, m, n, k, lda, ldb, ldd, sqrt, out_dists, out_inds, numOfNN,
        stream, allocator, workspace, worksize);
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
        typename value_t>
void l2_unexpanded_knn(size_t D, value_idx *out_inds, value_t *out_dists,
                       const value_t *index, const value_t *query,
                       size_t n_index_rows, size_t n_query_rows, int k,
                       bool rowMajorIndex, bool rowMajorQuery,
                       cudaStream_t stream, 
                       std::shared_ptr<deviceAllocator> allocator,
                       void *workspace, size_t worksize) {
/*
        args.k = k;
        args.dims = D;
        args.vectors = input[i];
        args.vectorsRowMajor = rowMajorIndex;
        args.numVectors = sizes[i];
        args.queries = search_items;
        args.queriesRowMajor = rowMajorQuery;
        args.numQueries = n;
        args.outDistances = out_d_ptr;
        args.outIndices = out_i_ptr;*/
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
  ASSERT(rowMajorIndex == rowMajorQuery, "l2Knn: rowMajorIndex and rowMajorQuery should have same layout");

  bool sqrt = (distanceType == raft::distance::DistanceType::L2SqrtUnexpanded);

  if (rowMajorIndex) {
    value_idx lda = D, ldb = D, ldd = n_index_rows;
    fusedL2kNN<value_t, value_t, value_t, value_idx, true>(n_query_rows,
          n_index_rows, D, lda, ldb, ldd, query, index, sqrt, out_dists, out_inds, k,
          stream, allocator, workspace, worksize);
  } else {
/*    IdxT lda = D, ldb = D, ldd = n_query_rows;
    fusedL2kNN<value_t, value_t, value_t, value_idx, false>(n_index_rows,
      n_query_rows, D, lda, ldb, ldd, index, query, sqrt, out_dists, out_inds, k,
      stream, allocator);*/
  }

} 

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
