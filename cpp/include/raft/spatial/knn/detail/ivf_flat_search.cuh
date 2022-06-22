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

#include "../ivf_flat_types.hpp"
#include "ann_utils.cuh"
#include "topk/radix_topk.cuh"
#include "topk/warpsort_topk.cuh"

#include <raft/common/device_loads_stores.cuh>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/pow2_utils.cuh>

#ifdef USE_FAISS
#include <faiss/gpu/utils/Comparators.cuh>
#include <faiss/gpu/utils/Select.cuh>
#endif

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <optional>

namespace raft::spatial::knn::detail::ivf_flat {

using raft::spatial::knn::ivf_flat::index;
using raft::spatial::knn::ivf_flat::search_params;

constexpr int kThreadsPerBlock = 128;

/**
 * @brief Copy Veclen elements of type T from `query` to `query_shared` at position `loadDim *
 * Veclen`.
 *
 * @param[in] query a pointer to a device global memory
 * @param[out] query_shared a pointer to a device shared memory
 * @param loadDim position at which to start copying elements.
 */
template <typename T, int Veclen>
__device__ __forceinline__ void queryLoadToShmem(const T* const& query,
                                                 T* query_shared,
                                                 const int loadDim)
{
  T queryReg[Veclen];
  const int loadIndex = loadDim * Veclen;
  ldg(queryReg, query + loadIndex);
  sts(&query_shared[loadIndex], queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<uint8_t, 8>(const uint8_t* const& query,
                                                             uint8_t* query_shared,
                                                             const int loadDim)
{
  constexpr int veclen = 2;  // 8 uint8_t
  uint32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<uint32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<uint32_t*>(query_shared) + loadIndex, queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<uint8_t, 16>(const uint8_t* const& query,
                                                              uint8_t* query_shared,
                                                              const int loadDim)
{
  constexpr int veclen = 4;  // 16 uint8_t
  uint32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<uint32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<uint32_t*>(query_shared) + loadIndex, queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<int8_t, 8>(const int8_t* const& query,
                                                            int8_t* query_shared,
                                                            const int loadDim)
{
  constexpr int veclen = 2;  // 8 int8_t
  int32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<int32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<int32_t*>(query_shared) + loadIndex, queryReg);
}

template <>
__device__ __forceinline__ void queryLoadToShmem<int8_t, 16>(const int8_t* const& query,
                                                             int8_t* query_shared,
                                                             const int loadDim)
{
  constexpr int veclen = 4;  // 16 int8_t
  int32_t queryReg[veclen];
  const int loadIndex = loadDim * veclen;
  ldg(queryReg, reinterpret_cast<int32_t const*>(query) + loadIndex);
  sts(reinterpret_cast<int32_t*>(query_shared) + loadIndex, queryReg);
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
                                                      const T* query_shared,
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
      lds(queryRegs[j], &query_shared[d]);
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
                                                      const uint8_t* query_shared,
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
      lds(queryRegs[j], reinterpret_cast<unsigned const*>(query_shared + baseShmemIndex) + d);
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
                                                      const uint8_t* query_shared,
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
      queryRegs[j] = reinterpret_cast<unsigned const*>(query_shared + baseShmemIndex)[d];
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
                                                      const uint8_t* query_shared,
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
      queryRegs[j] = reinterpret_cast<uint16_t const*>(query_shared + baseShmemIndex)[d];
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
                                                      const uint8_t* query_shared,
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
      queryRegs[j] = query_shared[baseShmemIndex + d];
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
                                                      const int8_t* query_shared,
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
      lds(queryRegs[j], reinterpret_cast<int32_t const*>(query_shared + baseShmemIndex) + d);
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
                                                      const int8_t* query_shared,
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
      queryRegs[j] = reinterpret_cast<uint16_t const*>(query_shared + baseShmemIndex)[d];
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
                                                      const int8_t* query_shared,
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
      queryRegs[j] = query_shared[baseShmemIndex + d];
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
 *
 * query_smem_elems must be multiple of WarpSize * Veclen
 */
template <int Capacity, int Veclen, bool Ascending, typename T, typename AccT, typename Lambda>
__global__ void __launch_bounds__(kThreadsPerBlock)
  interleaved_scan_kernel(Lambda compute_dist,
                          const uint32_t query_smem_elems,
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
  extern __shared__ __align__(256) uint8_t interleaved_scan_kernel_smem[];
  // Using shared memory for the (part of the) query;
  // This allows to save on global memory bandwidth when reading index and query
  // data at the same time.
  // Its size is `query_smem_elems`.
  T* query_shared = reinterpret_cast<T*>(interleaved_scan_kernel_smem);
#ifdef USE_FAISS
  // temporary use of FAISS blockSelect for development purpose of k <= 32
  // for comparison purpose
  __shared__ float smemK[kThreadsPerBlock];
  __shared__ size_t smemV[kThreadsPerBlock];

  constexpr auto Dir = !Ascending;
  constexpr auto identity =
    Dir ? std::numeric_limits<float>::min() : std::numeric_limits<float>::max();
  constexpr auto keyMax =
    Dir ? std::numeric_limits<size_t>::min() : std::numeric_limits<size_t>::max();

  faiss::gpu::
    BlockSelect<float, size_t, Dir, faiss::gpu::Comparator<float>, 32, 2, kThreadsPerBlock>
      queue(identity, keyMax, smemK, smemV, k);

#else
  topk::block_sort<topk::warp_sort_filtered, Capacity, Ascending, float, size_t> queue(
    k, interleaved_scan_kernel_smem + query_smem_elems * sizeof(T));
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

  int shLoadDim = (dim < query_smem_elems) ? dim : query_smem_elems;

  // load the query data from global to shared memory
  for (int loadDim = threadIdx.x; loadDim * Veclen < shLoadDim; loadDim += blockDim.x) {
    queryLoadToShmem<T, Veclen>(query, query_shared, loadDim);
  }
  __syncthreads();
  shLoadDim = (dim > query_smem_elems) ? shLoadDim : full_warps_along_dim;

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

    constexpr uint32_t kNumWarps = kThreadsPerBlock / WarpSize;
    // Every warp reads WarpSize vectors and computes the distances to them.
    // Then, the distances and corresponding ids are distributed among the threads,
    // and each thread adds one (id, dist) pair to the filtering queue.
    for (uint32_t block = warpId; block < numBlocks; block += kNumWarps) {
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
            obj.runLoadShmemCompute(data, query_shared, laneId, dBase, i);
          }  // end for i < WarpSize / kUnroll
        }    // end for dBase < full_warps_along_dim
      }

      if (dim > query_smem_elems) {
        constexpr int kUnroll = WarpSize / Veclen;
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
            obj.runLoadShmemCompute(data, query_shared, laneId, full_warps_along_dim + d, 0);
          }  // end for d < dim - full_warps_along_dim
        }
      }

      // Enqueue one element per thread
      constexpr float kDummy = Ascending ? upper_bound<float>() : lower_bound<float>();
      float val              = (valid) ? (float)dist : kDummy;
      queue.add(val, idx);
    }  // end for block < numBlocks
  }

  /// Warp_wise topk
#ifdef USE_FAISS
  queue.reduce();
  for (int i = threadIdx.x; i < k; i += kThreadsPerBlock) {
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
    &num_blocks_per_sm, func, kThreadsPerBlock, sMemSize));

  size_t min_grid_size = num_sms * num_blocks_per_sm;
  size_t min_grid_x    = ceildiv<size_t>(min_grid_size, numQueries);
  return min_grid_x > nprobe ? nprobe : static_cast<uint32_t>(min_grid_x);
}

template <int Capacity, int Veclen, bool Ascending, typename T, typename AccT, typename Lambda>
void launch_kernel(Lambda lambda,
                   const ivf_flat::index<T>& index,
                   const T* queries,
                   const uint32_t* coarse_index,
                   const uint32_t num_queries,
                   const uint32_t nprobe,
                   const uint32_t k,
                   size_t* neighbors,
                   float* distances,
                   uint32_t& grid_dim_x,
                   rmm::cuda_stream_view stream)
{
  RAFT_EXPECTS(reinterpret_cast<size_t>(queries) % (Veclen * sizeof(T)) == 0,
               "Queries data is not aligned to the vector load size (Veclen).");
  RAFT_EXPECTS(Veclen == index.veclen,
               "Configured Veclen does not match the index interleaving pattern.");
  constexpr auto kKernel   = interleaved_scan_kernel<Capacity, Veclen, Ascending, T, AccT, Lambda>;
  const int max_query_smem = 16384;
  int query_smem_elems =
    std::min<int>(max_query_smem / sizeof(T), Pow2<Veclen * WarpSize>::roundUp(index.dim()));
  int smem_size = query_smem_elems * sizeof(T);
#ifndef USE_FAISS
  constexpr int kSubwarpSize = std::min<int>(Capacity, WarpSize);
  smem_size += raft::spatial::knn::detail::topk::calc_smem_size_for_block_wide<AccT, size_t>(
    kThreadsPerBlock / kSubwarpSize, k);
#endif

  // power-of-two less than cuda limit (for better addr alignment)
  constexpr uint32_t kMaxGridY = 32768;

  if (grid_dim_x == 0) {
    grid_dim_x = configure_launch_x(std::min(kMaxGridY, num_queries), nprobe, smem_size, kKernel);
    return;
  }

  for (uint32_t query_offset = 0; query_offset < num_queries; query_offset += kMaxGridY) {
    uint32_t grid_dim_y = std::min<uint32_t>(kMaxGridY, num_queries - query_offset);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(kThreadsPerBlock);
    RAFT_LOG_TRACE(
      "Launching the ivf-flat interleaved_scan_kernel (%d, %d, 1) x (%d, 1, 1), nprobe = %d, "
      "smem_size = %d",
      grid_dim.x,
      grid_dim.y,
      block_dim.x,
      nprobe,
      smem_size);
    kKernel<<<grid_dim, block_dim, smem_size, stream>>>(lambda,
                                                        query_smem_elems,
                                                        queries,
                                                        coarse_index,
                                                        index.indices.data(),
                                                        index.data.data(),
                                                        index.list_sizes.data(),
                                                        index.list_offsets.data(),
                                                        nprobe,
                                                        k,
                                                        index.dim(),
                                                        neighbors,
                                                        distances);
    queries += grid_dim_y * index.dim();
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
template <int Capacity, int Veclen, bool Ascending, typename T, typename AccT, typename... Args>
void launch_with_fixed_consts(raft::distance::DistanceType metric, Args&&... args)
{
  if (metric == raft::distance::DistanceType::L2Expanded ||
      metric == raft::distance::DistanceType::L2Unexpanded) {
    launch_kernel<Capacity, Veclen, Ascending, T, AccT, euclidean_dist<Veclen, T, AccT>>({},
                                                                                         args...);
  } else {
    launch_kernel<Capacity, Veclen, Ascending, T, AccT, inner_prod_dist<Veclen, T, AccT>>({},
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
  static inline void run(int capacity, int veclen, bool select_min, Args&&... args)
  {
    if constexpr (Capacity > 1) {
      if (capacity * 2 <= Capacity) {
        return select_interleaved_scan_kernel<T, AccT, Capacity / 2, Veclen>::run(
          capacity, veclen, select_min, args...);
      }
    }
    if constexpr (Veclen > 1) {
      if (veclen * 2 <= Veclen) {
        return select_interleaved_scan_kernel<T, AccT, Capacity, Veclen / 2>::run(
          capacity, veclen, select_min, args...);
      }
    }
    RAFT_EXPECTS(capacity == Capacity,
                 "Capacity must be power-of-two not bigger than the maximum allowed size "
                 "topk::kMaxCapacity (%d).",
                 topk::kMaxCapacity);
    RAFT_EXPECTS(
      veclen == Veclen,
      "Veclen must be power-of-two not bigger than the maximum allowed size for this data type.");
    if (select_min) {
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
 * @param index previously built ivf-flat index
 * @param[in] queries device pointer to the query vectors [batch_size, dim]
 * @param[in] coarse_query_results device pointer to the cluster (list) ids [batch_size, n_probes]
 * @param n_queries batch size
 * @param metric type of the measured distance
 * @param n_probes number of nearest clusters to query
 * @param k number of nearest neighbors.
 *            NB: the maximum value of `k` is limited statically by `topk::kMaxCapacity`.
 * @param select_min whether to select nearest (true) or furthest (false) points w.r.t. the given
 * metric.
 * @param[out] neighbors device pointer to the result indices for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[out] distances device pointer to the result distances for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[inout] grid_dim_x number of blocks launched across all n_probes clusters;
 *               (one block processes one or more probes, hence: 1 <= grid_dim_x <= n_probes)
 * @param stream
 */
template <typename T, typename AccT>
void ivfflat_interleaved_scan(const ivf_flat::index<T>& index,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const raft::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const bool select_min,
                              size_t* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream)
{
  const int capacity = raft::spatial::knn::detail::topk::calc_capacity(k);
  select_interleaved_scan_kernel<T, AccT>::run(capacity,
                                               index.veclen,
                                               select_min,
                                               metric,
                                               index,
                                               queries,
                                               coarse_query_results,
                                               n_queries,
                                               n_probes,
                                               k,
                                               neighbors,
                                               distances,
                                               grid_dim_x,
                                               stream);
}

template <typename T, typename AccT>
void search_impl(const handle_t& handle,
                 const index<T>& index,
                 const T* queries,
                 uint32_t n_queries,
                 uint32_t k,
                 uint32_t n_probes,
                 bool select_min,
                 size_t* neighbors,
                 AccT* distances,
                 rmm::cuda_stream_view stream,
                 rmm::mr::device_memory_resource* search_mr)
{
  // The norm of query
  rmm::device_uvector<float> query_norm_dev(n_queries, stream, search_mr);
  // The distance value of cluster(list) and queries
  rmm::device_uvector<float> distance_buffer_dev(n_queries * index.n_lists(), stream, search_mr);
  // The topk distance value of cluster(list) and queries
  rmm::device_uvector<float> coarse_distances_dev(n_queries * n_probes, stream, search_mr);
  // The topk  index of cluster(list) and queries
  rmm::device_uvector<uint32_t> coarse_indices_dev(n_queries * n_probes, stream, search_mr);
  // The topk distance value of candicate vectors from each cluster(list)
  rmm::device_uvector<AccT> refined_distances_dev(n_queries * n_probes * k, stream, search_mr);
  // The topk index of candicate vectors from each cluster(list)
  rmm::device_uvector<size_t> refined_indices_dev(n_queries * n_probes * k, stream, search_mr);

  size_t float_query_size;
  if constexpr (std::is_integral_v<T>) {
    float_query_size = n_queries * index.dim();
  } else {
    float_query_size = 0;
  }
  rmm::device_uvector<float> converted_queries_dev(float_query_size, stream, search_mr);
  float* converted_queries_ptr = converted_queries_dev.data();

  if constexpr (std::is_same_v<T, float>) {
    converted_queries_ptr = const_cast<float*>(queries);
  } else {
    linalg::unaryOp(
      converted_queries_ptr, queries, n_queries * index.dim(), utils::mapping<float>{}, stream);
  }

  float alpha = 1.0f;
  float beta  = 0.0f;

  if (index.metric == raft::distance::DistanceType::L2Expanded) {
    alpha = -2.0f;
    beta  = 1.0f;
    utils::dots_along_rows(
      n_queries, index.dim(), converted_queries_ptr, query_norm_dev.data(), stream);
    utils::outer_add(query_norm_dev.data(),
                     n_queries,
                     index.center_norms->data(),
                     index.n_lists(),
                     distance_buffer_dev.data(),
                     stream);
    RAFT_LOG_TRACE_VEC(index.center_norms->data(), 20);
    RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  } else {
    alpha = 1.0f;
    beta  = 0.0f;
  }

  linalg::gemm(handle,
               true,
               false,
               index.n_lists(),
               n_queries,
               index.dim(),
               &alpha,
               index.centers.data(),
               index.dim(),
               converted_queries_ptr,
               index.dim(),
               &beta,
               distance_buffer_dev.data(),
               index.n_lists(),
               stream);

  RAFT_LOG_TRACE_VEC(distance_buffer_dev.data(), 20);
  if (n_probes <= raft::spatial::knn::detail::topk::kMaxCapacity) {
    topk::warp_sort_topk<AccT, uint32_t>(distance_buffer_dev.data(),
                                         nullptr,
                                         n_queries,
                                         index.n_lists(),
                                         n_probes,
                                         coarse_distances_dev.data(),
                                         coarse_indices_dev.data(),
                                         select_min,
                                         stream,
                                         search_mr);
  } else {
    topk::radix_topk<AccT, uint32_t, 11, 512>(distance_buffer_dev.data(),
                                              nullptr,
                                              n_queries,
                                              index.n_lists(),
                                              n_probes,
                                              coarse_distances_dev.data(),
                                              coarse_indices_dev.data(),
                                              select_min,
                                              stream,
                                              search_mr);
  }
  RAFT_LOG_TRACE_VEC(coarse_indices_dev.data(), 1 * n_probes);
  RAFT_LOG_TRACE_VEC(coarse_distances_dev.data(), 1 * n_probes);

  AccT* distances_dev_ptr = refined_distances_dev.data();
  size_t* indices_dev_ptr = refined_indices_dev.data();

  uint32_t grid_dim_x = 0;
  if (n_probes > 1) {
    // query the gridDimX size to store probes topK output
    ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(index,
                                                                    nullptr,
                                                                    nullptr,
                                                                    n_queries,
                                                                    index.metric,
                                                                    n_probes,
                                                                    k,
                                                                    select_min,
                                                                    nullptr,
                                                                    nullptr,
                                                                    grid_dim_x,
                                                                    stream);
  } else {
    grid_dim_x = 1;
  }

  if (grid_dim_x == 1) {
    distances_dev_ptr = distances;
    indices_dev_ptr   = neighbors;
  }

  ivfflat_interleaved_scan<T, typename utils::config<T>::value_t>(index,
                                                                  queries,
                                                                  coarse_indices_dev.data(),
                                                                  n_queries,
                                                                  index.metric,
                                                                  n_probes,
                                                                  k,
                                                                  select_min,
                                                                  indices_dev_ptr,
                                                                  distances_dev_ptr,
                                                                  grid_dim_x,
                                                                  stream);

  RAFT_LOG_TRACE_VEC(distances_dev_ptr, 2 * k);
  RAFT_LOG_TRACE_VEC(indices_dev_ptr, 2 * k);

  // Merge topk values from different blocks
  if (grid_dim_x > 1) {
    if (k <= raft::spatial::knn::detail::topk::kMaxCapacity) {
      topk::warp_sort_topk<AccT, size_t>(refined_distances_dev.data(),
                                         refined_indices_dev.data(),
                                         n_queries,
                                         k * grid_dim_x,
                                         k,
                                         distances,
                                         neighbors,
                                         select_min,
                                         stream,
                                         search_mr);
    } else {
      topk::radix_topk<AccT, size_t, 11, 512>(refined_distances_dev.data(),
                                              refined_indices_dev.data(),
                                              n_queries,
                                              k * grid_dim_x,
                                              k,
                                              distances,
                                              neighbors,
                                              select_min,
                                              stream,
                                              search_mr);
    }
  }
}

template <typename T>
inline void search(const handle_t& handle,
                   const search_params& params,
                   const index<T>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   size_t* neighbors,
                   float* distances,
                   rmm::cuda_stream_view stream,
                   rmm::mr::device_memory_resource* mr = nullptr)
{
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_flat_handle::search(%u, %u, %zu)", n_queries, k, neighbors);

  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());

  bool select_min;
  switch (index.metric) {
    case raft::distance::DistanceType::InnerProduct:
    case raft::distance::DistanceType::CosineExpanded:
    case raft::distance::DistanceType::CorrelationExpanded:
      // Similarity metrics have the opposite meaning, i.e. nearest neigbours are those with larger
      // similarity (See the same logic at cpp/include/raft/sparse/selection/detail/knn.cuh:362
      // {perform_k_selection})
      select_min = false;
      break;
    default: select_min = true;
  }

  //   // Set memory buffer to be reused across searches
  //   auto cur_memory_resource = rmm::mr::get_current_device_resource();
  //   if (!search_mem_res_.has_value() || search_mem_res_->get_upstream() != cur_memory_resource) {
  //     search_mem_res_.emplace(cur_memory_resource, Pow2<256>::roundUp(n_queries * n_probes * k *
  //     16));
  //   }
  std::optional<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>> pool_res;
  if (mr == nullptr) {
    pool_res.emplace(rmm::mr::get_current_device_resource(),
                     Pow2<256>::roundUp(n_queries * n_probes * k * 16));
    mr = &(pool_res.value());
  }

  return search_impl<T, float>(
    handle, index, queries, n_queries, k, n_probes, select_min, neighbors, distances, stream, mr);
}

}  // namespace raft::spatial::knn::detail::ivf_flat
