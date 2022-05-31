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

// #define USE_FAISS

#include "../ann_common.h"
#include "ann_utils.cuh"
#include "topk/warpsort_topk.cuh"

#include <raft/common/device_loads_stores.cuh>
#include <raft/core/logger.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.cuh>
#include <raft/pow2_utils.cuh>

#ifdef USE_FAISS
#include <faiss/gpu/utils/Comparators.cuh>
#include <faiss/gpu/utils/Select.cuh>
#endif

#include <rmm/cuda_stream_view.hpp>

namespace raft::spatial::knn::detail {

/**
 * @brief Copy Veclen elements of type T from `query` to `queryShared` at position `loadDim *
 * Veclen`.
 *
 * @param[in] query a pointer to a device global memory
 * @param[out] queryShared a pointer to a device shared memory
 * @param loadDim position at which to start copying elements.
 */
template <typename T, int Veclen>
__device__ __forceinline__ void queryLoadToShmem(const T* const& query,
                                                 T* queryShared,
                                                 const int loadDim)
{
  T queryReg[Veclen];
  const int loadIndex = loadDim * Veclen;
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
          typename Lambda,
          int Veclen,
          typename T,
          typename AccT>
struct loadAndComputeDist {
  Lambda compute_dist;
  AccT& dist;

  __device__ __forceinline__ loadAndComputeDist(AccT& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  template <typename IdxT>
  __device__ __forceinline__ void runLoadShmemCompute(const T* const& data,
                                                      const T* queryShared,
                                                      IdxT loadIndex,
                                                      IdxT baseShmemIndex,
                                                      IdxT iShmemIndex)
  {
    T encV[kUnroll][Veclen];
    T queryRegs[kUnroll][Veclen];
    constexpr int stride  = kUnroll * Veclen;
    const int shmemStride = baseShmemIndex + iShmemIndex * stride;
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      ldg(encV[j], data + (loadIndex + j * wordsPerVectorBlockDim) * Veclen);
      const int d = shmemStride + j * Veclen;
      lds(queryRegs[j], &queryShared[d]);
#pragma unroll
      for (int k = 0; k < Veclen; ++k) {
        compute_dist(dist, queryRegs[j][k], encV[j][k]);
      }
    }
  }

  template <typename IdxT>
  __device__ __forceinline__ void runLoadShflAndCompute(const T*& data,
                                                        const T* query,
                                                        IdxT baseLoadIndex,
                                                        const int laneId)
  {
    T encV[kUnroll][Veclen];
    T queryReg               = query[baseLoadIndex + laneId];
    constexpr int stride     = kUnroll * Veclen;
    constexpr int totalIter  = WarpSize / stride;
    constexpr int gmemStride = stride * wordsPerVectorBlockDim;
#pragma unroll
    for (int i = 0; i < totalIter; ++i, data += gmemStride) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        ldg(encV[j], (data + (laneId + j * wordsPerVectorBlockDim) * Veclen));
        T q[Veclen];
        const int d = (i * kUnroll + j) * Veclen;
#pragma unroll
        for (int k = 0; k < Veclen; ++k) {
          q[k] = shfl(queryReg, d + k, WarpSize);
          compute_dist(dist, q[k], encV[j][k]);  //@TODO add other metrics
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const T*& data, const T* query, const int laneId, const int dim, const int dimBlocks)
  {
    const int loadDim     = dimBlocks + laneId;
    T queryReg            = loadDim < dim ? query[loadDim] : 0;
    const int loadDataIdx = laneId * Veclen;
    for (int d = 0; d < dim - dimBlocks; d += Veclen, data += wordsPerVectorBlockDim * Veclen) {
      T enc[Veclen];
      T q[Veclen];
      ldg(enc, data + loadDataIdx);
#pragma unroll
      for (int k = 0; k < Veclen; k++) {
        q[k] = shfl(queryReg, d + k, WarpSize);
        compute_dist(dist, q[k], enc[k]);
      }
    }  // end for d < dim - dimBlocks
  }
};

// This handles uint8_t 8, 16 Veclens
template <int kUnroll, int wordsPerVectorBlockDim, typename Lambda, int uint8_veclen>
struct loadAndComputeDist<kUnroll,
                          wordsPerVectorBlockDim,
                          Lambda,
                          uint8_veclen,
                          uint8_t,
                          uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
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
        compute_dist(dist, queryRegs[j][k], encV[j][k]);
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
          compute_dist(dist, q[j][k], encV[j][k]);
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
        compute_dist(dist, q[k], enc[k]);
      }
    }  // end for d < dim - dimBlocks
  }
};

// Keep this specialized uint8 Veclen = 4, because compiler is generating suboptimal code while
// using above common template of int2/int4
template <int kUnroll, int wordsPerVectorBlockDim, typename Lambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, Lambda, 4, uint8_t, uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
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
      compute_dist(dist, queryRegs[j], encV[j]);
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
        compute_dist(dist, q[j], encV[j]);
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
      compute_dist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename Lambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, Lambda, 2, uint8_t, uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
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
      compute_dist(dist, queryRegs[j], encV[j]);
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
        compute_dist(dist, q[j], encV[j]);
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
      compute_dist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename Lambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, Lambda, 1, uint8_t, uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
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
      compute_dist(dist, queryRegs[j], encV[j]);
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
        compute_dist(dist, q[j], encV[j]);
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
      compute_dist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

// This device function is for int8 veclens 4, 8 and 16
template <int kUnroll, int wordsPerVectorBlockDim, typename Lambda, int int8_veclen>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, Lambda, int8_veclen, int8_t, int32_t> {
  Lambda compute_dist;
  int32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
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
        compute_dist(dist, queryRegs[j][k], encV[j][k]);
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
          compute_dist(dist, q[j][k], encV[j][k]);
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
        compute_dist(dist, q[k], enc[k]);
      }
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename Lambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, Lambda, 2, int8_t, int32_t> {
  Lambda compute_dist;
  int32_t& dist;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
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
      compute_dist(dist, queryRegs[j], encV[j]);
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
        compute_dist(dist, q[j], encV[j]);
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
      compute_dist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

template <int kUnroll, int wordsPerVectorBlockDim, typename Lambda>
struct loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, Lambda, 1, int8_t, int32_t> {
  Lambda compute_dist;
  int32_t& dist;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
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
      compute_dist(dist, queryRegs[j], encV[j]);
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
        compute_dist(dist, q[j], encV[j]);
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
      compute_dist(dist, q, enc);
    }  // end for d < dim - dimBlocks
  }
};

/**
 * See `ivfflat_interleaved_scan` for parameter docs.
 */
template <int Capacity, int Veclen, bool Greater, typename T, typename AccT, typename Lambda>
__global__ void interleaved_scan_kernel(Lambda compute_dist,
                                        const T* queries,
                                        const uint32_t* coarse_index,
                                        const uint32_t* list_index,
                                        const T* list_data,
                                        const uint32_t* list_lengths,
                                        const uint32_t* list_prefix_interleave,
                                        const uint32_t nprobe,
                                        const uint32_t k,
                                        const uint32_t dim,
                                        size_t* neighbors,
                                        float* distances)
{
#ifdef USE_FAISS
  // temporary use of FAISS blockSelect for development purpose of k <= 32
  // for comparison purpose
  __shared__ float smemK[utils::kNumWarps * 32];
  __shared__ size_t smemV[utils::kNumWarps * 32];

  constexpr auto Dir = Greater;
  constexpr auto identity =
    Dir ? std::numeric_limits<float>::min() : std::numeric_limits<float>::max();
  constexpr auto keyMax =
    Dir ? std::numeric_limits<size_t>::min() : std::numeric_limits<size_t>::max();

  faiss::gpu::
    BlockSelect<float, size_t, Dir, faiss::gpu::Comparator<float>, 32, 2, utils::kThreadPerBlock>
      queue(identity, keyMax, smemK, smemV, k);

#else
  extern __shared__ __align__(256) uint8_t smem_ext[];
  topk::block_sort<topk::warp_sort_filtered, Capacity, !Greater, float, size_t> queue(k, smem_ext);
#endif

  using align_warp = Pow2<WarpSize>;
  const int laneId = align_warp::mod(threadIdx.x);
  const int warpId = align_warp::div(threadIdx.x);
  int queryId      = blockIdx.y;

  /// Set the address
  auto query                           = queries + queryId * dim;
  constexpr int wordsPerVectorBlockDim = WarpSize;

  // How many full warps needed to compute the distance (without remainder)
  const int full_warps_along_dim = align_warp::roundDown(dim);

  // Using shared memory for the query;
  // This allows to save on global memory bandwidth when reading index and query
  // data at the same time.
  // This should be multiple of warpSize = 32
  constexpr uint32_t queryShmemSize = 2048;
  __shared__ T queryShared[queryShmemSize];

  int shLoadDim = (dim < queryShmemSize) ? dim : queryShmemSize;

  // load the query data from global to shared memory
  for (int loadDim = threadIdx.x; loadDim * Veclen < shLoadDim; loadDim += blockDim.x) {
    queryLoadToShmem<T, Veclen>(query, queryShared, loadDim);
  }
  __syncthreads();
  shLoadDim = (dim > queryShmemSize) ? shLoadDim : full_warps_along_dim;

  // Every CUDA block scans one cluster at a time.
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

    // Every warp reads WarpSize vectors and computes the distances to them.
    // Then, the distances and corresponding ids are distributed among the threads,
    // and each thread adds one (id, dist) pair to the filtering queue.
    for (uint32_t block = warpId; block < numBlocks; block += utils::kNumWarps) {
      AccT dist = 0;
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
          constexpr int kUnroll   = WarpSize / Veclen;
          constexpr int stride    = kUnroll * Veclen;
          constexpr int totalIter = WarpSize / stride;

          loadAndComputeDist<kUnroll,
                             wordsPerVectorBlockDim,
                             decltype(compute_dist),
                             Veclen,
                             T,
                             AccT>
            obj(dist, compute_dist);
#pragma unroll
          for (int i = 0; i < totalIter; ++i, data += stride * wordsPerVectorBlockDim) {
            obj.runLoadShmemCompute(data, queryShared, laneId, dBase, i);
          }  // end for i < WarpSize / kUnroll
        }    // end for dBase < full_warps_along_dim
      }

      if (dim > queryShmemSize) {
        constexpr int kUnroll = WarpSize / Veclen;
        ;
        loadAndComputeDist<kUnroll, wordsPerVectorBlockDim, decltype(compute_dist), Veclen, T, AccT>
          obj(dist, compute_dist);
        for (int dBase = shLoadDim; dBase < full_warps_along_dim; dBase += WarpSize) {  //
          obj.runLoadShflAndCompute(data, query, dBase, laneId);
        }
        // Remainder chunk = dim - full_warps_along_dim
        obj.runLoadShflAndComputeRemainder(data, query, laneId, dim, full_warps_along_dim);
        // end for d < dim - full_warps_along_dim
      } else {
        if (valid) {
          /// Remainder chunk = dim - full_warps_along_dim
          for (int d = 0; d < dim - full_warps_along_dim;
               d += Veclen, data += wordsPerVectorBlockDim * Veclen) {
            loadAndComputeDist<1, wordsPerVectorBlockDim, decltype(compute_dist), Veclen, T, AccT>
              obj(dist, compute_dist);
            obj.runLoadShmemCompute(data, queryShared, laneId, full_warps_along_dim + d, 0);
          }  // end for d < dim - full_warps_along_dim
        }
      }

      // Enqueue one element per thread
      constexpr float kDummy = Greater ? lower_bound<float>() : upper_bound<float>();
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

/**
 *  Configure the gridDim.x to maximize GPU occupancy, but reduce the output size
 */
template <typename T>
uint32_t configure_launch_x(uint32_t numQueries, uint32_t nprobe, int32_t sMemSize, T func)
{
  int dev_id;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  int num_sms;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));
  int num_blocks_per_sm = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &num_blocks_per_sm, func, utils::kThreadPerBlock, sMemSize));

  size_t min_grid_size = num_sms * num_blocks_per_sm;
  size_t min_grid_x    = ceildiv<size_t>(min_grid_size, numQueries);
  return min_grid_x > nprobe ? nprobe : static_cast<uint32_t>(min_grid_x);
}

template <int Capacity, int Veclen, bool Greater, typename T, typename AccT, typename Lambda>
void launch_kernel(Lambda lambda,
                   const T* queries,
                   const uint32_t* coarse_index,
                   const uint32_t* list_index,
                   const T* list_data,
                   const uint32_t* list_lengths,
                   const uint32_t* list_prefix_interleave,
                   const uint32_t nprobe,
                   const uint32_t k,
                   const uint32_t dim,
                   size_t* neighbors,
                   float* distances,
                   const uint32_t batch_size,
                   uint32_t& grid_dim_x,
                   rmm::cuda_stream_view stream)
{
  constexpr auto kKernel = interleaved_scan_kernel<Capacity, Veclen, Greater, T, AccT, Lambda>;
#ifdef USE_FAISS
  int smem_size = 0;
#else
  int smem_size = raft::spatial::knn::detail::topk::calc_smem_size_for_block_wide<AccT, size_t>(
    utils::kNumWarps, k);
#endif

  // power-of-two less than cuda limit (for better addr alignment)
  constexpr uint32_t kMaxGridY = 32768;

  if (grid_dim_x == 0) {
    grid_dim_x = configure_launch_x(std::min(kMaxGridY, batch_size), nprobe, smem_size, kKernel);
    return;
  }

  for (uint32_t query_offset = 0; query_offset < batch_size; query_offset += kMaxGridY) {
    uint32_t grid_dim_y = std::min<uint32_t>(kMaxGridY, batch_size - query_offset);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(utils::kThreadPerBlock);
    RAFT_LOG_TRACE(
      "Launching the ivf-flat interleaved_scan_kernel (%d, %d, 1) x (%d, 1, 1), nprobe = %d",
      grid_dim.x,
      grid_dim.y,
      block_dim.x,
      nprobe);
    kKernel<<<grid_dim, block_dim, smem_size, stream>>>(lambda,
                                                        queries,
                                                        coarse_index,
                                                        list_index,
                                                        list_data,
                                                        list_lengths,
                                                        list_prefix_interleave,
                                                        nprobe,
                                                        k,
                                                        dim,
                                                        neighbors,
                                                        distances);
    queries += grid_dim_y * dim;
    neighbors += grid_dim_y * grid_dim_x * k;
    distances += grid_dim_y * grid_dim_x * k;
  }
}

template <int Veclen, typename T, typename AccT>
struct euclidean_dist {
  __device__ inline void operator()(AccT& acc, AccT x, AccT y)
  {
    const AccT diff = x - y;
    acc += diff * diff;
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, uint8_t, uint32_t> {
  __device__ inline void operator()(uint32_t& acc, uint32_t x, uint32_t y)
  {
    if constexpr (Veclen > 1) {
      const uint32_t diff = __vabsdiffu4(x, y);
      acc                 = dp4a(diff, diff, acc);
    } else {
      const uint32_t diff = x - y;
      acc += diff * diff;
    }
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, int8_t, int32_t> {
  __device__ inline void operator()(int32_t& acc, int32_t x, int32_t y)
  {
    if constexpr (Veclen > 1) {
      const int32_t diff = static_cast<int32_t>(__vabsdiffs4(x, y));
      acc                = dp4a(diff, diff, acc);
    } else {
      const int32_t diff = x - y;
      acc += diff * diff;
    }
  }
};

template <int Veclen, typename T, typename AccT>
struct inner_prod_dist {
  __device__ inline void operator()(AccT& acc, AccT x, AccT y)
  {
    if constexpr (Veclen > 1 && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
      acc = dp4a(x, y, acc);
    } else {
      acc += x * y;
    }
  }
};

/** Select the distance computation function and forward the rest of the arguments. */
template <int Capacity, int Veclen, bool Greater, typename T, typename AccT, typename... Args>
void launch_with_fixed_consts(raft::distance::DistanceType metric, Args&&... args)
{
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2Unexpanded) {
    launch_kernel<Capacity, Veclen, Greater, T, AccT, euclidean_dist<Veclen, T, AccT>>({}, args...);
  } else {
    launch_kernel<Capacity, Veclen, Greater, T, AccT, inner_prod_dist<Veclen, T, AccT>>({},
                                                                                        args...);
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
  static inline void run(int capacity, int veclen, bool greater, Args&&... args)
  {
    if constexpr (Capacity > 1) {
      if (capacity * 2 <= Capacity) {
        return select_interleaved_scan_kernel<T, AccT, Capacity / 2, Veclen>::run(
          capacity, veclen, greater, args...);
      }
    }
    if constexpr (Veclen > 1) {
      if (veclen * 2 <= Veclen) {
        return select_interleaved_scan_kernel<T, AccT, Capacity, Veclen / 2>::run(
          capacity, veclen, greater, args...);
      }
    }
    RAFT_EXPECTS(capacity == Capacity,
                 "Capacity must be power-of-two not bigger than the maximum allowed size "
                 "topk::kMaxCapacity (%d).",
                 topk::kMaxCapacity);
    RAFT_EXPECTS(
      veclen == Veclen,
      "Veclen must be power-of-two not bigger than the maximum allowed size for this data type.");
    if (greater) {
      launch_with_fixed_consts<Capacity, Veclen, true, T, AccT>(args...);
    } else {
      launch_with_fixed_consts<Capacity, Veclen, false, T, AccT>(args...);
    }
  }
};

/**
 * @brief Configure and launch an appropriate template instance of the interleaved scan kernel.
 *
 * @tparam T value type
 * @tparam AccT accumulated type
 *
 * @param[in] queries device pointer to the query vectors [batch_size, dim]
 * @param[in] coarse_index device pointer to the cluster (list) ids [batch_size, nprobe]
 * @param[in] list_index device pointer to the row ids in each cluster [nrow]
 * @param[in] list_data device pointer to the data in all clusters interleaved [nrow, dim]
 * @param[in] list_lengths device pointer to the numbers of vectors in each cluster [nlist]
 * @param[in] list_prefix_interleave device pointer to the offsets of each cluster in list_index
 * [nlist]
 * @param[in] metric type of the measured distance
 * @param[in] nprobe number of nearest clusters to query
 * @param[in] k number of nearest neighbors.
 *            NB: the maximum value of `k` is limited statically by `topk::kMaxCapacity`.
 * @param[in] batch_size number of query vectors
 * @param[in] dim dimensionality of search data and query vectors
 * @param[out] neighbors device pointer to the result indices for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[out] distances device pointer to the result distances for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[in] stream
 * @param[in] greater whether to select nearest (false) or furthest (true) points w.r.t. the given
 * metric.
 * @param[in] veclen (optimization parameters) size of the vector for vectorized processing
 * @param[inout] grid_dim_x number of blocks launched across all nprobe clusters;
 *               (one block processes one or more probes, hence: 1 <= grid_dim_x <= nprobe)
 */
template <typename T, typename AccT>
void ivfflat_interleaved_scan(const T* queries,
                              const uint32_t* coarse_index,
                              const uint32_t* list_index,
                              const T* list_data,
                              const uint32_t* list_lengths,
                              const uint32_t* list_prefix_interleave,
                              const raft::distance::DistanceType metric,
                              const uint32_t nprobe,
                              const uint32_t k,
                              const uint32_t batch_size,
                              const uint32_t dim,
                              size_t* neighbors,
                              float* distances,
                              rmm::cuda_stream_view stream,
                              const bool greater,
                              const int veclen,
                              uint32_t& grid_dim_x)
{
  const int capacity = raft::spatial::knn::detail::topk::calc_capacity(k);
  select_interleaved_scan_kernel<T, AccT>::run(capacity,
                                               veclen,
                                               greater,
                                               metric,
                                               queries,
                                               coarse_index,
                                               list_index,
                                               list_data,
                                               list_lengths,
                                               list_prefix_interleave,
                                               nprobe,
                                               k,
                                               dim,
                                               neighbors,
                                               distances,
                                               batch_size,
                                               grid_dim_x,
                                               stream);
}

}  // namespace raft::spatial::knn::detail
