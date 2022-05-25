/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include "../ann_common.h"
#include "ann_utils.cuh"
#include "knn_brute_force_faiss.cuh"
#include "topk/warpsort_topk.cuh"
#include <raft/common/device_loads_stores.cuh>

#include "common_faiss.h"
#include "processing.hpp"

#include "processing.hpp"
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/pow2_utils.cuh>

//#include <label/classlabels.cuh>
#include <raft/distance/distance.hpp>
#include <raft/spatial/knn/faiss_mr.hpp>

#include <faiss/gpu/GpuDistance.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/GpuIndexIVFScalarQuantizer.h>
#include <faiss/gpu/GpuResources.h>
#include <faiss/gpu/utils/Limits.cuh>
#include <faiss/gpu/utils/Select.cuh>
#include <faiss/gpu/utils/Tensor.cuh>
#include <faiss/utils/Heap.h>

#include <thrust/iterator/transform_iterator.h>

#include <raft/distance/distance_type.hpp>

#include <iostream>
#include <set>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

// #define DEBUG
/// Init centriods
template <typename T>
void ivfflat_centriod_init(T* dataset, T* centriod, int nlist, int dim, int n)
{
  // srand (time(NULL));
  int nparts = n / nlist;
  int index  = rand() % nparts;
  for (int i = 0; i < nlist; i++) {
    memcpy(centriod + i * dim, dataset + i * nparts + index, sizeof(T) * dim);
  }  // end for
}  // end func ivfflat_centriod_init

// search

template <typename T, int veclen>
__device__ __forceinline__ void queryLoadToShmem(const T* const& query,
                                                 T* queryShared,
                                                 const int loadDim)
{
  T queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, query + loadIndex);
  sts(&queryShared[loadIndex], queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<uint8_t, 8>(const uint8_t* const& query,
                                                             uint8_t* queryShared,
                                                             const int loadDim)
{
  constexpr int veclen = 2;  // 8 uint8_t
  uint32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<uint32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<uint32_t*>(queryShared) + loadIndex, queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<uint8_t, 16>(const uint8_t* const& query,
                                                              uint8_t* queryShared,
                                                              const int loadDim)
{
  constexpr int veclen = 4;  // 16 uint8_t
  uint32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<uint32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<uint32_t*>(queryShared) + loadIndex, queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<int8_t, 8>(const int8_t* const& query,
                                                            int8_t* queryShared,
                                                            const int loadDim)
{
  constexpr int veclen = 2;  // 8 int8_t
  int32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<int32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<int32_t*>(queryShared) + loadIndex, queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<int8_t, 16>(const int8_t* const& query,
                                                             int8_t* queryShared,
                                                             const int loadDim)
{
  constexpr int veclen = 4;  // 16 int8_t
  int32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<int32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<int32_t*>(queryShared) + loadIndex, queryReg);
}

template <int kUnroll,
          int wordsPerVectorBlockDim,
          typename computeLambda,
          int veclen,
          typename T,
          typename AccT>
struct loadAndComputeDist {
  computeLambda computeDist;
  AccT& dist;

  __device__ __forceinline__ loadAndComputeDist(AccT& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }

  template <typename IdxT>
  __device__ __forceinline__ void runLoadShmemCompute(const T* const& data,
                                                      const T* queryShared,
                                                      IdxT loadIndex,
                                                      IdxT baseShmemIndex,
                                                      IdxT iShmemIndex)
  {
    T encV[kUnroll][veclen];
    T queryRegs[kUnroll][veclen];
    constexpr int stride  = kUnroll * veclen;
    const int shmemStride = baseShmemIndex + iShmemIndex * stride;
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      ldg(encV[j], data + (loadIndex + j * wordsPerVectorBlockDim) * veclen);
      const int d = shmemStride + j * veclen;
      lds(queryRegs[j], &queryShared[d]);
#pragma unroll
      for (int k = 0; k < veclen; ++k) {
        computeDist(dist, queryRegs[j][k], encV[j][k]);
      }
    }
  }

  template <typename IdxT>
  __device__ __forceinline__ void runLoadShflAndCompute(const T*& data,
                                                        const T* query,
                                                        IdxT baseLoadIndex,
                                                        const int laneId)
  {
    T encV[kUnroll][veclen];
    T queryReg               = query[baseLoadIndex + laneId];
    constexpr int stride     = kUnroll * veclen;
    constexpr int totalIter  = WarpSize / stride;
    constexpr int gmemStride = stride * wordsPerVectorBlockDim;
#pragma unroll
    for (int i = 0; i < totalIter; ++i, data += gmemStride) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        ldg(encV[j], (data + (laneId + j * wordsPerVectorBlockDim) * veclen));
        T q[veclen];
        const int d = (i * kUnroll + j) * veclen;
#pragma unroll
        for (int k = 0; k < veclen; ++k) {
          q[k] = shfl(queryReg, d + k, WarpSize);
          computeDist(dist, q[k], encV[j][k]);  //@TODO add other metrics
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const T*& data, const T* query, const int laneId, const int dim, const int dimBlocks)
  {
    const int loadDim     = dimBlocks + laneId;
    T queryReg            = loadDim < dim ? query[loadDim] : 0;
    const int loadDataIdx = laneId * veclen;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += wordsPerVectorBlockDim * veclen) {
      T enc[veclen];
      T q[veclen];
      ldg(enc, data + loadDataIdx);
#pragma unroll
      for (int k = 0; k < veclen; k++) {
        q[k] = shfl(queryReg, d + k, WarpSize);
        computeDist(dist, q[k], enc[k]);
      }
    }  // end for d < dim - dimBlocks
  }
};

// This handles uint8_t 8, 16 veclens
template <int kUnroll, int wordsPerVectorBlockDim, typename computeLambda, int uint8_veclen>
struct loadAndComputeDist<kUnroll,
                          wordsPerVectorBlockDim,
                          computeLambda,
                          uint8_veclen,
                          uint8_t,
                          uint32_t> {
  computeLambda computeDist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* queryShared,
                                                      int loadIndex,
                                                      int baseShmemIndex,
                                                      int iShmemIndex)
  {
    constexpr int veclen_int = uint8_veclen / 4;  // converting uint8_t veclens to int
    uint32_t encV[kUnroll][veclen_int];
    uint32_t queryRegs[kUnroll][veclen_int];

    loadIndex = loadIndex * veclen_int;
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      ldg(encV[j],
          reinterpret_cast<unsigned const*>(data) + loadIndex +
            j * wordsPerVectorBlockDim * veclen_int);
      const int d = iShmemIndex * kUnroll + j * veclen_int;
      lds(queryRegs[j], reinterpret_cast<unsigned const*>(queryShared + baseShmemIndex) + d);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        computeDist(dist, queryRegs[j][k], encV[j][k]);
      }
    }
  }
  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int laneId)
  {
    constexpr int veclen_int = uint8_veclen / 4;  // converting uint8_t veclens to int
    uint32_t encV[kUnroll][veclen_int];
    uint32_t queryReg =
      (laneId < 8) ? reinterpret_cast<unsigned const*>(query + baseLoadIndex)[laneId] : 0;
    uint32_t q[kUnroll][veclen_int];
    constexpr int stride = kUnroll * uint8_veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * wordsPerVectorBlockDim) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        ldg(encV[j],
            reinterpret_cast<unsigned const*>(data) +
              (laneId + j * wordsPerVectorBlockDim) * veclen_int);
        const int d = (i * kUnroll + j) * veclen_int;
#pragma unroll
        for (int k = 0; k < veclen_int; ++k) {
          q[j][k] = shfl(queryReg, d + k, WarpSize);
          computeDist(dist, q[j][k], encV[j][k]);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int laneId,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen_int = uint8_veclen / 4;
    const int loadDim        = dimBlocks + laneId * 4;  // Here 4 is for 1 - int
    uint32_t queryReg = loadDim < dim ? reinterpret_cast<uint32_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks;
         d += uint8_veclen, data += wordsPerVectorBlockDim * uint8_veclen) {
      uint32_t enc[veclen_int];
      uint32_t q[veclen_int];
      ldg(enc, reinterpret_cast<uint32_t const*>(data) + laneId * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        q[k] = shfl(queryReg, (d / 4) + k, WarpSize);
        computeDist(dist, q[k], enc[k]);
      }
    }  // end for d < dim - dimBlocks
  }
};

// Keep this specialized uint8 veclen = 4, because compiler is generating suboptimal code while
// using above common template of int2/int4
template <int kUnroll, int wordsPerVectorBlockDim, typename computeLambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, computeLambda, 4, uint8_t, uint32_t> {
  computeLambda computeDist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* queryShared,
                                                      int loadIndex,
                                                      int baseShmemIndex,
                                                      int iShmemIndex)
  {
    uint32_t encV[kUnroll];
    uint32_t queryRegs[kUnroll];

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      encV[j]     = reinterpret_cast<unsigned const*>(data)[loadIndex + j * wordsPerVectorBlockDim];
      const int d = (iShmemIndex * kUnroll + j);
      queryRegs[j] = reinterpret_cast<unsigned const*>(queryShared + baseShmemIndex)[d];
      computeDist(dist, queryRegs[j], encV[j]);
    }
  }
  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int laneId)
  {
    uint32_t encV[kUnroll];
    uint32_t queryReg =
      (laneId < 8) ? reinterpret_cast<unsigned const*>(query + baseLoadIndex)[laneId] : 0;
    uint32_t q[kUnroll];
    constexpr int veclen = 4;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * wordsPerVectorBlockDim) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        encV[j]     = reinterpret_cast<unsigned const*>(data)[laneId + j * wordsPerVectorBlockDim];
        const int d = (i * kUnroll + j);
        q[j]        = shfl(queryReg, d, WarpSize);
        computeDist(dist, q[j], encV[j]);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int laneId,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 4;
    const int loadDim    = dimBlocks + laneId;
    uint32_t queryReg    = loadDim < dim ? reinterpret_cast<unsigned const*>(query)[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += wordsPerVectorBlockDim * veclen) {
      uint32_t enc = reinterpret_cast<unsigned const*>(data)[laneId];
      uint32_t q   = shfl(queryReg, d / veclen, WarpSize);
      computeDist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename computeLambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, computeLambda, 2, uint8_t, uint32_t> {
  computeLambda computeDist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* queryShared,
                                                      int loadIndex,
                                                      int baseShmemIndex,
                                                      int iShmemIndex)
  {
    uint32_t encV[kUnroll];
    uint32_t queryRegs[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      encV[j]     = 0;
      encV[j]     = reinterpret_cast<uint16_t const*>(data)[loadIndex + j * wordsPerVectorBlockDim];
      const int d = (iShmemIndex * kUnroll + j);
      queryRegs[j] = 0;
      queryRegs[j] = reinterpret_cast<uint16_t const*>(queryShared + baseShmemIndex)[d];
      computeDist(dist, queryRegs[j], encV[j]);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int laneId)
  {
    uint32_t encV[kUnroll];
    uint32_t queryReg = 0;
    queryReg = (laneId < 16) ? reinterpret_cast<uint16_t const*>(query + baseLoadIndex)[laneId] : 0;
    uint32_t q[kUnroll];
    constexpr int veclen = 2;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * wordsPerVectorBlockDim) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        encV[j]     = 0;
        encV[j]     = reinterpret_cast<uint16_t const*>(data)[laneId + j * wordsPerVectorBlockDim];
        const int d = (i * kUnroll + j);
        q[j]        = shfl(queryReg, d, WarpSize);
        computeDist(dist, q[j], encV[j]);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int laneId,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 2;
    int loadDim          = dimBlocks + laneId * veclen;
    uint32_t queryReg    = 0;
    queryReg = loadDim < dim ? reinterpret_cast<uint16_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += wordsPerVectorBlockDim * veclen) {
      uint32_t enc = reinterpret_cast<uint16_t const*>(data)[laneId];
      uint32_t q   = shfl(queryReg, d / veclen, WarpSize);
      computeDist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename computeLambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, computeLambda, 1, uint8_t, uint32_t> {
  computeLambda computeDist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* queryShared,
                                                      int loadIndex,
                                                      int baseShmemIndex,
                                                      int iShmemIndex)
  {
    uint32_t encV[kUnroll];
    uint32_t queryRegs[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      encV[j]      = data[loadIndex + j * wordsPerVectorBlockDim];
      const int d  = (iShmemIndex * kUnroll + j);
      queryRegs[j] = queryShared[baseShmemIndex + d];
      computeDist(dist, queryRegs[j], encV[j]);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int laneId)
  {
    uint32_t encV[kUnroll];
    uint32_t queryReg = 0;
    queryReg          = query[baseLoadIndex + laneId];
    uint32_t q[kUnroll];
    constexpr int veclen = 1;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * wordsPerVectorBlockDim) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        encV[j]     = 0;
        encV[j]     = data[laneId + j * wordsPerVectorBlockDim];
        const int d = (i * kUnroll + j);
        q[j]        = shfl(queryReg, d, WarpSize);
        computeDist(dist, q[j], encV[j]);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int laneId,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 1;
    int loadDim          = dimBlocks + laneId;
    uint32_t queryReg    = 0;
    queryReg             = loadDim < dim ? query[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += wordsPerVectorBlockDim * veclen) {
      uint32_t enc = 0;
      enc          = data[laneId];
      uint32_t q   = shfl(queryReg, d, WarpSize);
      computeDist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

// This device function is for int8 veclens 4, 8 and 16
template <int kUnroll, int wordsPerVectorBlockDim, typename computeLambda, int int8_veclen>
struct loadAndComputeDist<kUnroll,
                          wordsPerVectorBlockDim,
                          computeLambda,
                          int8_veclen,
                          int8_t,
                          int32_t> {
  computeLambda computeDist;
  int32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* queryShared,
                                                      int loadIndex,
                                                      int baseShmemIndex,
                                                      int iShmemIndex)
  {
    constexpr int veclen_int = int8_veclen / 4;  // converting int8_t veclens to int
    int32_t encV[kUnroll][veclen_int];
    int32_t queryRegs[kUnroll][veclen_int];

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      ldg(encV[j],
          reinterpret_cast<int32_t const*>(data) +
            (loadIndex + j * wordsPerVectorBlockDim) * veclen_int);
      const int d = iShmemIndex * kUnroll + j * veclen_int;
      lds(queryRegs[j], reinterpret_cast<int32_t const*>(queryShared + baseShmemIndex) + d);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        computeDist(dist, queryRegs[j][k], encV[j][k]);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int laneId)
  {
    constexpr int veclen_int = int8_veclen / 4;  // converting int8_t veclens to int
    int32_t encV[kUnroll][veclen_int];
    int32_t queryReg =
      (laneId < 8) ? reinterpret_cast<int32_t const*>(query + baseLoadIndex)[laneId] : 0;
    int32_t q[kUnroll][veclen_int];
    constexpr int stride = kUnroll * int8_veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * wordsPerVectorBlockDim) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        ldg(encV[j],
            reinterpret_cast<int32_t const*>(data) +
              (laneId + j * wordsPerVectorBlockDim) * veclen_int);
        const int d = (i * kUnroll + j) * veclen_int;
#pragma unroll
        for (int k = 0; k < veclen_int; ++k) {
          q[j][k] = shfl(queryReg, d + k, WarpSize);
          computeDist(dist, q[j][k], encV[j][k]);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int laneId, const int dim, const int dimBlocks)
  {
    constexpr int veclen_int = int8_veclen / 4;
    const int loadDim        = dimBlocks + laneId * 4;  // Here 4 is for 1 - int;
    int32_t queryReg = loadDim < dim ? reinterpret_cast<int32_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks;
         d += int8_veclen, data += wordsPerVectorBlockDim * int8_veclen) {
      int32_t enc[veclen_int];
      int32_t q[veclen_int];
      ldg(enc, reinterpret_cast<int32_t const*>(data) + laneId * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        q[k] = shfl(queryReg, (d / 4) + k, WarpSize);  // Here 4 is for 1 - int;
        computeDist(dist, q[k], enc[k]);
      }
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename computeLambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, computeLambda, 2, int8_t, int32_t> {
  computeLambda computeDist;
  int32_t& dist;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }
  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* queryShared,
                                                      int loadIndex,
                                                      int baseShmemIndex,
                                                      int iShmemIndex)
  {
    int32_t encV[kUnroll];
    int32_t queryRegs[kUnroll];
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      encV[j]     = 0;
      encV[j]     = reinterpret_cast<uint16_t const*>(data)[loadIndex + j * wordsPerVectorBlockDim];
      const int d = (iShmemIndex * kUnroll + j);
      queryRegs[j] = 0;
      queryRegs[j] = reinterpret_cast<uint16_t const*>(queryShared + baseShmemIndex)[d];
      computeDist(dist, queryRegs[j], encV[j]);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int laneId)
  {
    int32_t encV[kUnroll];
    int32_t queryReg = 0;
    queryReg = (laneId < 16) ? reinterpret_cast<uint16_t const*>(query + baseLoadIndex)[laneId] : 0;
    int32_t q[kUnroll];
    constexpr int veclen = 2;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * wordsPerVectorBlockDim) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        encV[j]     = 0;
        encV[j]     = reinterpret_cast<uint16_t const*>(data)[laneId + j * wordsPerVectorBlockDim];
        const int d = (i * kUnroll + j);
        q[j]        = shfl(queryReg, d, WarpSize);
        computeDist(dist, q[j], encV[j]);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int laneId, const int dim, const int dimBlocks)
  {
    constexpr int veclen = 2;
    int loadDim          = dimBlocks + laneId * veclen;
    int32_t queryReg     = 0;
    queryReg = loadDim < dim ? reinterpret_cast<uint16_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += wordsPerVectorBlockDim * veclen) {
      int32_t enc = reinterpret_cast<uint16_t const*>(data + laneId * veclen)[0];
      int32_t q   = shfl(queryReg, d / veclen, WarpSize);
      computeDist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename computeLambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, computeLambda, 1, int8_t, int32_t> {
  computeLambda computeDist;
  int32_t& dist;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, computeLambda op)
    : dist(dist), computeDist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* queryShared,
                                                      int loadIndex,
                                                      int baseShmemIndex,
                                                      int iShmemIndex)
  {
    int32_t encV[kUnroll];
    int32_t queryRegs[kUnroll];

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      encV[j]      = 0;
      encV[j]      = data[loadIndex + j * wordsPerVectorBlockDim];
      const int d  = (iShmemIndex * kUnroll + j);
      queryRegs[j] = 0;
      queryRegs[j] = queryShared[baseShmemIndex + d];
      computeDist(dist, queryRegs[j], encV[j]);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int laneId)
  {
    constexpr int veclen = 1;
    constexpr int stride = kUnroll * veclen;
    int32_t encV[kUnroll];
    int32_t queryReg = 0;
    queryReg         = query[baseLoadIndex + laneId];
    int32_t q[kUnroll];

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * wordsPerVectorBlockDim) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        encV[j]     = 0;
        encV[j]     = data[laneId + j * wordsPerVectorBlockDim];
        const int d = (i * kUnroll + j);
        q[j]        = shfl(queryReg, d, WarpSize);
        computeDist(dist, q[j], encV[j]);
      }
    }
  }
  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int laneId, const int dim, const int dimBlocks)
  {
    constexpr int veclen = 1;
    const int loadDim    = dimBlocks + laneId;
    int32_t queryReg     = 0;
    queryReg             = loadDim < dim ? query[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += wordsPerVectorBlockDim * veclen) {
      int32_t enc = 0;
      enc         = data[laneId];
      int32_t q   = shfl(queryReg, d, WarpSize);
      computeDist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

//#define USE_FAISS 1

template <int CAPACITY, int veclen, typename T, typename value_t, typename distLambda, bool GREATER>
__global__ void interleaved_scan(
  const T* queries,        // Input: Query Vector; [batch_size, dim]
  uint32_t* coarse_index,  // Record the cluster(list) id; [batch_size,nprobe]
  uint32_t* list_index,    // Record the id of vector for each cluster(list); [nrow]
  const T* list_data,      // Record the full value of vector for each cluster(list) interleaved;
                           // [nrow, dim]
  uint32_t* list_lengths,  // The number of vectors in each cluster(list); [nlist]
  uint32_t* list_prefix_interleave,           // The start offset of each cluster(list) for
                                              // list_index; [nlist]
  const raft::distance::DistanceType metric,  // Function to process the different metric
  distLambda computeDist,
  const uint32_t nprobe,
  const uint32_t k,
  const uint32_t dim,
  size_t* neighbors,  // [batch_size, nprobe]
  float* distances    // [batch_size, nprobe]
)
{
#ifdef USE_FAISS
  // temporary use of FAISS blockSelect for development purpose of k <= 32
  // for comparison purpose
  __shared__ float smemK[utils::kNumWarps * 32];
  __shared__ size_t smemV[utils::kNumWarps * 32];

  constexpr auto Dir = GREATER;
  constexpr auto identity =
    Dir ? std::numeric_limits<float>::min() : std::numeric_limits<float>::max();
  constexpr auto keyMax =
    Dir ? std::numeric_limits<size_t>::min() : std::numeric_limits<size_t>::max();

  faiss::gpu::
    BlockSelect<float, size_t, Dir, faiss::gpu::Comparator<float>, 32, 2, utils::kThreadPerBlock>
      queue(identity, keyMax, smemK, smemV, k);

#else
  extern __shared__ __align__(256) uint8_t smem_ext[];
  topk::block_sort<topk::warp_sort_filtered, CAPACITY, !GREATER, float, size_t> queue(k, smem_ext);
#endif

  using align_warp = Pow2<WarpSize>;
  const int laneId = align_warp::mod(threadIdx.x);
  const int warpId = align_warp::div(threadIdx.x);
  int queryId      = blockIdx.y;

  /// Set the address
  auto query                           = queries + queryId * dim;
  constexpr int bytesPerVectorBlockDim = sizeof(T) * WarpSize;
  constexpr int wordsPerVectorBlockDim = bytesPerVectorBlockDim / sizeof(T);

  // int wordsPerVectorBlock = wordsPerVectorBlockDim * dim;
  const int dimBlocks = align_warp::roundDown(dim);

  // This should be multiple of warpSize = 32
  constexpr uint32_t queryShmemSize = 2048;
  __shared__ T queryShared[queryShmemSize];

  int shLoadDim = (dim < queryShmemSize) ? dim : queryShmemSize;
  shLoadDim     = shLoadDim / veclen;

  for (int loadDim = threadIdx.x; loadDim < shLoadDim; loadDim += blockDim.x) {
    queryLoadToShmem<T, veclen>(query, queryShared, loadDim);
  }
  __syncthreads();
  shLoadDim = (dim > queryShmemSize) ? (shLoadDim * veclen) : dimBlocks;

  for (int probeId = blockIdx.x; probeId < nprobe; probeId += gridDim.x) {
    uint32_t listId = coarse_index[queryId * nprobe + probeId];  // The id of cluster(list)

    /**
     * Uses shared memory
     */
    //@TODO The result with dimension
    // The start address of the full value of vector for each cluster(list) interleaved
    auto vecsBase = list_data + size_t(list_prefix_interleave[listId]) * dim;
    // The start address of index of vector for each cluster(list) interleaved
    auto indexBase = list_index + list_prefix_interleave[listId];
    // The number of vectors in each cluster(list); [nlist]
    const uint32_t numVecs = list_lengths[listId];

    // The number of interleaved group to be processed
    const uint32_t numBlocks = ceildiv<uint32_t>(numVecs, WarpSize);

    for (uint32_t block = warpId; block < numBlocks; block += utils::kNumWarps) {
      value_t dist = 0;
      // This is the vector a given lane/thread handles
      const uint32_t vec = block * WarpSize + laneId;
      bool valid         = vec < numVecs;
      size_t idx         = (valid) ? (size_t)indexBase[vec] : (size_t)laneId;
      // This is where this warp begins reading data
      const T* data =
        vecsBase + size_t(block) * wordsPerVectorBlockDim * dim;  // Start position of this block

      if (valid) {
        /// load query from shared mem
        for (int dBase = 0; dBase < shLoadDim; dBase += WarpSize) {  //
          constexpr int kUnroll   = WarpSize / veclen;
          constexpr int stride    = kUnroll * veclen;
          constexpr int totalIter = WarpSize / stride;

          loadAndComputeDist<kUnroll,
                             wordsPerVectorBlockDim,
                             decltype(computeDist),
                             veclen,
                             T,
                             value_t>
            obj(dist, computeDist);
#pragma unroll
          for (int i = 0; i < totalIter; ++i, data += stride * wordsPerVectorBlockDim) {
            obj.runLoadShmemCompute(data, queryShared, laneId, dBase, i);
          }  // end for i < WarpSize / kUnroll
        }    // end for dBase < dimBlocks
      }

      if (dim > queryShmemSize) {
        constexpr int kUnroll = WarpSize / veclen;
        ;
        loadAndComputeDist<kUnroll,
                           wordsPerVectorBlockDim,
                           decltype(computeDist),
                           veclen,
                           T,
                           value_t>
          obj(dist, computeDist);
        for (int dBase = shLoadDim; dBase < dimBlocks; dBase += WarpSize) {  //
          obj.runLoadShflAndCompute(data, query, dBase, laneId);
        }
        // Remainder chunk = dim - dimBlocks
        obj.runLoadShflAndComputeRemainder(data, query, laneId, dim, dimBlocks);
        // end for d < dim - dimBlocks
      } else {
        if (valid) {
          /// Remainder chunk = dim - dimBlocks
          for (int d = 0; d < dim - dimBlocks;
               d += veclen, data += wordsPerVectorBlockDim * veclen) {
            loadAndComputeDist<1, wordsPerVectorBlockDim, decltype(computeDist), veclen, T, value_t>
              obj(dist, computeDist);
            obj.runLoadShmemCompute(data, queryShared, laneId, dimBlocks + d, 0);
          }  // end for d < dim - dimBlocks
        }
      }

      // Enqueue one element per thread
      constexpr float kDummy = GREATER ? lower_bound<float>() : upper_bound<float>();
      float val              = (valid) ? (float)dist : kDummy;
      queue.add(val, idx);
    }  // end for block < numBlocks
  }

  /// Warp_wise topk
#ifdef USE_FAISS
  queue.reduce();
  for (int i = threadIdx.x; i < k; i += utils::kThreadPerBlock) {
    neighbors[queryId * k * gridDim.x + blockIdx.x * k + i] = (size_t)smemV[i];
    distances[queryId * k * gridDim.x + blockIdx.x * k + i] = smemK[i];
  }
#else
  queue.done();
  queue.store(distances + queryId * k * gridDim.x + blockIdx.x * k,
              neighbors + queryId * k * gridDim.x + blockIdx.x * k);
#endif
}  // end kernel

template <typename T>
dim3 launchConfigGenerator(uint32_t numQueries, uint32_t nprobe, int32_t sMemSize, T func)
{
  int devId;
  RAFT_CUDA_TRY(cudaGetDevice(&devId));
  int numSMs;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, devId));
  int numBlocksPerSm = 0;
  dim3 grid;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocksPerSm, func, utils::kThreadPerBlock, sMemSize));

  std::size_t minGridSize = numSMs * numBlocksPerSm;
  std::size_t yChunks     = numQueries;
  std::size_t xChunks     = nprobe;
  // grid.y                  = yChunks > minGridSize ? minGridSize : yChunks;
  grid.y = yChunks;
  grid.x = (minGridSize - grid.y) <= 0 ? 1 : xChunks;
  if (grid.x != 1) {
    std::size_t i = 1;
    while (grid.y * i < minGridSize) {
      i++;
    }
    grid.x = i >= xChunks ? xChunks : i;
  }

  return grid;
}

template <int capacity, int veclen, typename T, typename acc_type>
void launch_interleaved_scan_kernel(
  const T* queries,        // Input: Query Vector; [batch_size, dim]
  uint32_t* coarse_index,  // Record the cluster(list) id; [batch_size,nprobe]
  uint32_t* list_index,    // Record the id of vector for each cluster(list); [nrow]
  void* list_data,         // Record the full value of vector for each cluster(list) interleaved;
                           // [nrow, dim]
  uint32_t* list_lengths,  // The number of vectors in each cluster(list); [nlist]
  uint32_t* list_prefix_interleave,     // The start offset of each cluster(list) for
                                        // list_index; [nlist]
  raft::distance::DistanceType metric,  // Function to process the different metric
  const uint32_t nprobe,
  const uint32_t k,
  const uint32_t dim,
  size_t* neighbors,  // [batch_size, nprobe]
  float* distances,   // [batch_size, nprobe]
  const bool greater,
  const uint32_t batch_size,
  cudaStream_t stream,
  uint32_t& gridDimX)
{
#ifdef USE_FAISS
  int smem_size = 0;
#else
  int smem_size = raft::spatial::knn::detail::topk::calc_smem_size_for_block_wide<acc_type, size_t>(
    utils::kNumWarps, k);
#endif

  // Accumulation inner product lambda
  auto inner_prod_lambda = [] __device__(acc_type & acc, acc_type & x, acc_type & y) {
    if constexpr ((std::is_same<T, int8_t>{}) || (std::is_same<T, uint8_t>{})) {
      if constexpr (veclen == 1) {
        acc += x * y;
      } else {
        acc = dp4a(x, y, acc);
      }
    } else if constexpr (std::is_same<T, float>{}) {
      acc += x * y;
    }
  };

  // Accumulation euclidean L2 lambda
  auto euclidean_lambda = [] __device__(acc_type & acc, acc_type & x, acc_type & y) {
    if constexpr ((std::is_same<T, uint8_t>{})) {
      if constexpr (veclen == 1) {
        const acc_type diff = x - y;
        acc += diff * diff;
      } else {
        const acc_type diff = __vabsdiffu4(x, y);
        acc                 = dp4a(diff, diff, acc);
      }
    } else if constexpr (std::is_same<T, int8_t>{}) {
      if constexpr (veclen == 1) {
        const acc_type diff = x - y;
        acc += diff * diff;
      } else {
        asm("vabsdiff4.u32.s32.s32 %0,%1,%2,%3;" : "=r"(x) : "r"(x), "r"(y), "r"(0));
        acc = dp4a(x, x, acc);
      }
    } else if constexpr ((std::is_same<T, float>{})) {
      const acc_type diff = x - y;
      acc += diff * diff;
    }
  };

  dim3 block_dim(utils::kThreadPerBlock);

  if (greater) {
    if (metric == raft::distance::DistanceType::L2Expanded ||
        metric == raft::distance::DistanceType::L2Unexpanded) {
      constexpr auto interleaved_scan_euclidean_greater =
        interleaved_scan<capacity, veclen, T, acc_type, decltype(euclidean_lambda), true>;
      if (gridDimX == 0) {
        dim3 grid_dim =
          launchConfigGenerator(batch_size, nprobe, smem_size, interleaved_scan_euclidean_greater);
        gridDimX = grid_dim.x;
        return;
      }
      dim3 grid_dim =
        launchConfigGenerator(batch_size, nprobe, smem_size, interleaved_scan_euclidean_greater);
      interleaved_scan_euclidean_greater<<<grid_dim, block_dim, smem_size, stream>>>(
        queries,
        coarse_index,
        list_index,
        (T*)list_data,
        list_lengths,
        list_prefix_interleave,
        metric,
        euclidean_lambda,
        nprobe,
        k,
        dim,
        neighbors,
        distances);
    } else {
      constexpr auto interleaved_scan_inner_prod_greater =
        interleaved_scan<capacity, veclen, T, acc_type, decltype(inner_prod_lambda), true>;
      if (gridDimX == 0) {
        dim3 grid_dim =
          launchConfigGenerator(batch_size, nprobe, smem_size, interleaved_scan_inner_prod_greater);
        gridDimX = grid_dim.x;
        return;
      }
      dim3 grid_dim =
        launchConfigGenerator(batch_size, nprobe, smem_size, interleaved_scan_inner_prod_greater);
      interleaved_scan_inner_prod_greater<<<grid_dim, block_dim, smem_size, stream>>>(
        queries,
        coarse_index,
        list_index,
        (T*)list_data,
        list_lengths,
        list_prefix_interleave,
        metric,
        inner_prod_lambda,
        nprobe,
        k,
        dim,
        neighbors,
        distances);
    }
  } else {
    if (metric == raft::distance::DistanceType::L2Expanded ||
        metric == raft::distance::DistanceType::L2Unexpanded) {
      constexpr auto interleaved_scan_euclidean_ngreater =
        interleaved_scan<capacity, veclen, T, acc_type, decltype(euclidean_lambda), false>;
      if (gridDimX == 0) {
        dim3 grid_dim =
          launchConfigGenerator(batch_size, nprobe, smem_size, interleaved_scan_euclidean_ngreater);
        gridDimX = grid_dim.x;
        return;
      }
      dim3 grid_dim =
        launchConfigGenerator(batch_size, nprobe, smem_size, interleaved_scan_euclidean_ngreater);
      interleaved_scan_euclidean_ngreater<<<grid_dim, block_dim, smem_size, stream>>>(
        queries,
        coarse_index,
        list_index,
        (T*)list_data,
        list_lengths,
        list_prefix_interleave,
        metric,
        euclidean_lambda,
        nprobe,
        k,
        dim,
        neighbors,
        distances);
    } else {
      constexpr auto interleaved_scan_inner_prod_ngreater =
        interleaved_scan<capacity, veclen, T, acc_type, decltype(inner_prod_lambda), false>;
      if (gridDimX == 0) {
        dim3 grid_dim = launchConfigGenerator(
          batch_size, nprobe, smem_size, interleaved_scan_inner_prod_ngreater);
        gridDimX = grid_dim.x;
        return;
      }
      dim3 grid_dim =
        launchConfigGenerator(batch_size, nprobe, smem_size, interleaved_scan_inner_prod_ngreater);
      interleaved_scan_inner_prod_ngreater<<<grid_dim, block_dim, smem_size, stream>>>(
        queries,
        coarse_index,
        list_index,
        (T*)list_data,
        list_lengths,
        list_prefix_interleave,
        metric,
        inner_prod_lambda,
        nprobe,
        k,
        dim,
        neighbors,
        distances);
    }
  }
}

/**
 * Lift the `capacity` and `veclen` parameters to the template level,
 * forward the rest of the arguments unmodified to `launch_interleaved_scan_kernel`.
 */
template <typename T,
          typename AccT,
          int Capacity = topk::kMaxCapacity,
          int Veclen   = std::max<int>(1, 16 / sizeof(T))>
struct select_interleaved_scan_kernel {
  /**
   * Recursively reduce the `Capacity` and `Veclen` parameters until they match the
   * corresponding runtime arguments.
   * By default, this recursive process starts with maximum possible values of the
   * two parameters and ends with both values equal to 1.
   */
  template <typename... Args>
  static inline void run(int capacity, int veclen, Args&&... args)
  {
    if constexpr (Capacity > 1) {
      if (capacity * 2 <= Capacity) {
        return select_interleaved_scan_kernel<T, AccT, Capacity / 2, Veclen>::run(
          capacity, veclen, args...);
      }
    }
    if constexpr (Veclen > 1) {
      if (veclen * 2 <= Veclen) {
        return select_interleaved_scan_kernel<T, AccT, Capacity, Veclen / 2>::run(
          capacity, veclen, args...);
      }
    }
    RAFT_EXPECTS(capacity == Capacity,
                 "Capacity must be power-of-two not bigger than the maximum allowed size.");
    RAFT_EXPECTS(
      veclen == Veclen,
      "Veclen must be power-of-two not bigger than the maximum allowed size for this data type.");
    return launch_interleaved_scan_kernel<Capacity, Veclen, T, AccT>(args...);
  }
};

template <typename T, typename AccT>
void ivfflat_interleaved_scan(const T* queries,                  //[batch_size, dim]
                              uint32_t* coarse_index,            //[batch_size,nprobe]
                              uint32_t* list_index,              // [nrow]
                              void* list_data,                   //[nrow, dim]
                              uint32_t* list_lengths,            // [nlist]
                              uint32_t* list_prefix_interleave,  // [nlist]
                              const raft::distance::DistanceType metric,
                              const uint32_t nprobe,
                              const uint32_t k,
                              const uint32_t batch_size,
                              const uint32_t dim,
                              size_t* neighbors,  // [batch_size, nprobe, k]
                              float* distances,   // [batch_size, nprobe, k]
                              cudaStream_t stream,
                              const bool greater,
                              const int veclen,
                              uint32_t& gridDimX)
{
  const int capacity = raft::spatial::knn::detail::topk::calc_capacity(k);
  select_interleaved_scan_kernel<T, AccT>::run(capacity,
                                               veclen,
                                               queries,
                                               coarse_index,
                                               list_index,
                                               list_data,
                                               list_lengths,
                                               list_prefix_interleave,
                                               metric,
                                               nprobe,
                                               k,
                                               dim,
                                               neighbors,
                                               distances,
                                               greater,
                                               batch_size,
                                               stream,
                                               gridDimX);
}

}  // namespace detail
}  // namespace knn
}  // namespace spatial
}  // namespace raft
