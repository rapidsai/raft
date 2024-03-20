/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/logger.hpp>  // RAFT_LOG_TRACE
#include <raft/core/operators.hpp>
#include <raft/distance/distance_types.hpp>
#include <raft/matrix/detail/select_warpsort.cuh>
#include <raft/neighbors/detail/ivf_common.cuh>
#include <raft/neighbors/ivf_flat_types.hpp>
#include <raft/neighbors/sample_filter_types.hpp>
#include <raft/spatial/knn/detail/ann_utils.cuh>
#include <raft/util/cuda_rt_essentials.hpp>  // RAFT_CUDA_TRY
#include <raft/util/device_loads_stores.cuh>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace raft::neighbors::ivf_flat::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

constexpr int kThreadsPerBlock = 128;

auto RAFT_WEAK_FUNCTION is_local_topk_feasible(uint32_t k) -> bool
{
  return k <= matrix::detail::select::warpsort::kMaxCapacity;
}

/**
 * @brief Copy `n` elements per block from one place to another.
 *
 * @param[out] out target pointer (unique per block)
 * @param[in] in source pointer
 * @param n number of elements to copy
 */
template <int VecBytes = 16, typename T>
__device__ inline void copy_vectorized(T* out, const T* in, uint32_t n)
{
  constexpr int VecElems = VecBytes / sizeof(T);  // NOLINT
  using align_bytes      = Pow2<(size_t)VecBytes>;
  if constexpr (VecElems > 1) {
    using align_elems = Pow2<VecElems>;
    if (!align_bytes::areSameAlignOffsets(out, in)) {
      return copy_vectorized<(VecBytes >> 1), T>(out, in, n);
    }
    {  // process unaligned head
      uint32_t head = align_bytes::roundUp(in) - in;
      if (head > 0) {
        copy_vectorized<sizeof(T), T>(out, in, head);
        n -= head;
        in += head;
        out += head;
      }
    }
    {  // process main part vectorized
      using vec_t = typename IOType<T, VecElems>::Type;
      copy_vectorized<sizeof(vec_t), vec_t>(
        reinterpret_cast<vec_t*>(out), reinterpret_cast<const vec_t*>(in), align_elems::div(n));
    }
    {  // process unaligned tail
      uint32_t tail = align_elems::mod(n);
      if (tail > 0) {
        n -= tail;
        copy_vectorized<sizeof(T), T>(out + n, in + n, tail);
      }
    }
  }
  if constexpr (VecElems <= 1) {
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
      out[i] = in[i];
    }
  }
}

/**
 * @brief Load a part of a vector from the index and from query, compute the (part of the) distance
 * between them, and aggregate it using the provided Lambda; one structure per thread, per query,
 * and per index item.
 *
 * @tparam kUnroll elements per loop (normally, kUnroll = WarpSize / Veclen)
 * @tparam Lambda computing the part of the distance for one dimension and aggregating it:
 *                void (AccT& acc, AccT x, AccT y)
 * @tparam Veclen size of the vectorized load
 * @tparam T type of the data in the query and the index
 * @tparam AccT type of the accumulated value (an optimization for 8bit values to be loaded as 32bit
 * values)
 */
template <int kUnroll, typename Lambda, int Veclen, typename T, typename AccT>
struct loadAndComputeDist {
  Lambda compute_dist;
  AccT& dist;

  __device__ __forceinline__ loadAndComputeDist(AccT& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  /**
   * Load parts of vectors from the index and query and accumulates the partial distance.
   * This version assumes the query is stored in shared memory.
   * Every thread here processes exactly kUnroll * Veclen elements independently of others.
   */
  template <typename IdxT>
  __device__ __forceinline__ void runLoadShmemCompute(const T* const& data,
                                                      const T* query_shared,
                                                      IdxT loadIndex,
                                                      IdxT shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      T encV[Veclen];
      ldg(encV, data + (loadIndex + j * kIndexGroupSize) * Veclen);
      T queryRegs[Veclen];
      lds(queryRegs, &query_shared[shmemIndex + j * Veclen]);
#pragma unroll
      for (int k = 0; k < Veclen; ++k) {
        compute_dist(dist, queryRegs[k], encV[k]);
      }
    }
  }

  /**
   * Load parts of vectors from the index and query and accumulates the partial distance.
   * This version assumes the query is stored in the global memory and is different for every
   * thread. One warp loads exactly WarpSize query elements at once and then reshuffles them into
   * corresponding threads (`WarpSize / (kUnroll * Veclen)` elements per thread at once).
   */
  template <typename IdxT>
  __device__ __forceinline__ void runLoadShflAndCompute(const T*& data,
                                                        const T* query,
                                                        IdxT baseLoadIndex,
                                                        const int lane_id)
  {
    T queryReg               = query[baseLoadIndex + lane_id];
    constexpr int stride     = kUnroll * Veclen;
    constexpr int totalIter  = WarpSize / stride;
    constexpr int gmemStride = stride * kIndexGroupSize;
#pragma unroll
    for (int i = 0; i < totalIter; ++i, data += gmemStride) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        T encV[Veclen];
        ldg(encV, data + (lane_id + j * kIndexGroupSize) * Veclen);
        const int d = (i * kUnroll + j) * Veclen;
#pragma unroll
        for (int k = 0; k < Veclen; ++k) {
          compute_dist(dist, shfl(queryReg, d + k, WarpSize), encV[k]);
        }
      }
    }
  }

  /**
   * Load parts of vectors from the index and query and accumulates the partial distance.
   * This version augments `runLoadShflAndCompute` when `dim` is not a multiple of `WarpSize`.
   */
  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const T*& data, const T* query, const int lane_id, const int dim, const int dimBlocks)
  {
    const int loadDim     = dimBlocks + lane_id;
    T queryReg            = loadDim < dim ? query[loadDim] : T{0};
    const int loadDataIdx = lane_id * Veclen;
    for (int d = 0; d < dim - dimBlocks; d += Veclen, data += kIndexGroupSize * Veclen) {
      T enc[Veclen];
      ldg(enc, data + loadDataIdx);
#pragma unroll
      for (int k = 0; k < Veclen; k++) {
        compute_dist(dist, shfl(queryReg, d + k, WarpSize), enc[k]);
      }
    }
  }
};

// This handles uint8_t 8, 16 Veclens
template <int kUnroll, typename Lambda, int uint8_veclen>
struct loadAndComputeDist<kUnroll, Lambda, uint8_veclen, uint8_t, uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
    constexpr int veclen_int = uint8_veclen / 4;  // converting uint8_t veclens to int
    loadIndex                = loadIndex * veclen_int;
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV[veclen_int];
      ldg(encV,
          reinterpret_cast<unsigned const*>(data) + loadIndex + j * kIndexGroupSize * veclen_int);
      uint32_t queryRegs[veclen_int];
      lds(queryRegs, reinterpret_cast<unsigned const*>(query_shared + shmemIndex) + j * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        compute_dist(dist, queryRegs[k], encV[k]);
      }
    }
  }
  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    constexpr int veclen_int = uint8_veclen / 4;  // converting uint8_t veclens to int
    uint32_t queryReg =
      (lane_id < 8) ? reinterpret_cast<unsigned const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int stride = kUnroll * uint8_veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV[veclen_int];
        ldg(encV,
            reinterpret_cast<unsigned const*>(data) + (lane_id + j * kIndexGroupSize) * veclen_int);
        const int d = (i * kUnroll + j) * veclen_int;
#pragma unroll
        for (int k = 0; k < veclen_int; ++k) {
          compute_dist(dist, shfl(queryReg, d + k, WarpSize), encV[k]);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen_int = uint8_veclen / 4;
    const int loadDim        = dimBlocks + lane_id * 4;  // Here 4 is for 1 - int
    uint32_t queryReg = loadDim < dim ? reinterpret_cast<uint32_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks;
         d += uint8_veclen, data += kIndexGroupSize * uint8_veclen) {
      uint32_t enc[veclen_int];
      ldg(enc, reinterpret_cast<uint32_t const*>(data) + lane_id * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        uint32_t q = shfl(queryReg, (d / 4) + k, WarpSize);
        compute_dist(dist, q, enc[k]);
      }
    }
  }
};

// Keep this specialized uint8 Veclen = 4, because compiler is generating suboptimal code while
// using above common template of int2/int4
template <int kUnroll, typename Lambda>
struct loadAndComputeDist<kUnroll, Lambda, 4, uint8_t, uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV      = reinterpret_cast<unsigned const*>(data)[loadIndex + j * kIndexGroupSize];
      uint32_t queryRegs = reinterpret_cast<unsigned const*>(query_shared + shmemIndex)[j];
      compute_dist(dist, queryRegs, encV);
    }
  }
  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    uint32_t queryReg =
      (lane_id < 8) ? reinterpret_cast<unsigned const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int veclen = 4;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV = reinterpret_cast<unsigned const*>(data)[lane_id + j * kIndexGroupSize];
        uint32_t q    = shfl(queryReg, i * kUnroll + j, WarpSize);
        compute_dist(dist, q, encV);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 4;
    const int loadDim    = dimBlocks + lane_id;
    uint32_t queryReg    = loadDim < dim ? reinterpret_cast<unsigned const*>(query)[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      uint32_t enc = reinterpret_cast<unsigned const*>(data)[lane_id];
      uint32_t q   = shfl(queryReg, d / veclen, WarpSize);
      compute_dist(dist, q, enc);
    }
  }
};

template <int kUnroll, typename Lambda>
struct loadAndComputeDist<kUnroll, Lambda, 2, uint8_t, uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV      = reinterpret_cast<uint16_t const*>(data)[loadIndex + j * kIndexGroupSize];
      uint32_t queryRegs = reinterpret_cast<uint16_t const*>(query_shared + shmemIndex)[j];
      compute_dist(dist, queryRegs, encV);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    uint32_t queryReg =
      (lane_id < 16) ? reinterpret_cast<uint16_t const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int veclen = 2;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV = reinterpret_cast<uint16_t const*>(data)[lane_id + j * kIndexGroupSize];
        uint32_t q    = shfl(queryReg, i * kUnroll + j, WarpSize);
        compute_dist(dist, q, encV);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 2;
    int loadDim          = dimBlocks + lane_id * veclen;
    uint32_t queryReg = loadDim < dim ? reinterpret_cast<uint16_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      uint32_t enc = reinterpret_cast<uint16_t const*>(data)[lane_id];
      uint32_t q   = shfl(queryReg, d / veclen, WarpSize);
      compute_dist(dist, q, enc);
    }
  }
};

template <int kUnroll, typename Lambda>
struct loadAndComputeDist<kUnroll, Lambda, 1, uint8_t, uint32_t> {
  Lambda compute_dist;
  uint32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(uint32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const uint8_t* const& data,
                                                      const uint8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      uint32_t encV      = data[loadIndex + j * kIndexGroupSize];
      uint32_t queryRegs = query_shared[shmemIndex + j];
      compute_dist(dist, queryRegs, encV);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const uint8_t*& data,
                                                        const uint8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    uint32_t queryReg    = query[baseLoadIndex + lane_id];
    constexpr int veclen = 1;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        uint32_t encV = data[lane_id + j * kIndexGroupSize];
        uint32_t q    = shfl(queryReg, i * kUnroll + j, WarpSize);
        compute_dist(dist, q, encV);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(const uint8_t*& data,
                                                                 const uint8_t* query,
                                                                 const int lane_id,
                                                                 const int dim,
                                                                 const int dimBlocks)
  {
    constexpr int veclen = 1;
    int loadDim          = dimBlocks + lane_id;
    uint32_t queryReg    = loadDim < dim ? query[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      uint32_t enc = data[lane_id];
      uint32_t q   = shfl(queryReg, d, WarpSize);
      compute_dist(dist, q, enc);
    }
  }
};

// This device function is for int8 veclens 4, 8 and 16
template <int kUnroll, typename Lambda, int int8_veclen>
struct loadAndComputeDist<kUnroll, Lambda, int8_veclen, int8_t, int32_t> {
  Lambda compute_dist;
  int32_t& dist;

  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
    constexpr int veclen_int = int8_veclen / 4;  // converting int8_t veclens to int

#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      int32_t encV[veclen_int];
      ldg(encV,
          reinterpret_cast<int32_t const*>(data) + (loadIndex + j * kIndexGroupSize) * veclen_int);
      int32_t queryRegs[veclen_int];
      lds(queryRegs, reinterpret_cast<int32_t const*>(query_shared + shmemIndex) + j * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        compute_dist(dist, queryRegs[k], encV[k]);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    constexpr int veclen_int = int8_veclen / 4;  // converting int8_t veclens to int

    int32_t queryReg =
      (lane_id < 8) ? reinterpret_cast<int32_t const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int stride = kUnroll * int8_veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        int32_t encV[veclen_int];
        ldg(encV,
            reinterpret_cast<int32_t const*>(data) + (lane_id + j * kIndexGroupSize) * veclen_int);
        const int d = (i * kUnroll + j) * veclen_int;
#pragma unroll
        for (int k = 0; k < veclen_int; ++k) {
          int32_t q = shfl(queryReg, d + k, WarpSize);
          compute_dist(dist, q, encV[k]);
        }
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int lane_id, const int dim, const int dimBlocks)
  {
    constexpr int veclen_int = int8_veclen / 4;
    const int loadDim        = dimBlocks + lane_id * 4;  // Here 4 is for 1 - int;
    int32_t queryReg = loadDim < dim ? reinterpret_cast<int32_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += int8_veclen, data += kIndexGroupSize * int8_veclen) {
      int32_t enc[veclen_int];
      ldg(enc, reinterpret_cast<int32_t const*>(data) + lane_id * veclen_int);
#pragma unroll
      for (int k = 0; k < veclen_int; k++) {
        int32_t q = shfl(queryReg, (d / 4) + k, WarpSize);  // Here 4 is for 1 - int;
        compute_dist(dist, q, enc[k]);
      }
    }
  }
};

template <int kUnroll, typename Lambda>
struct loadAndComputeDist<kUnroll, Lambda, 2, int8_t, int32_t> {
  Lambda compute_dist;
  int32_t& dist;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }
  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      int32_t encV      = reinterpret_cast<uint16_t const*>(data)[loadIndex + j * kIndexGroupSize];
      int32_t queryRegs = reinterpret_cast<uint16_t const*>(query_shared + shmemIndex)[j];
      compute_dist(dist, queryRegs, encV);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    int32_t queryReg =
      (lane_id < 16) ? reinterpret_cast<uint16_t const*>(query + baseLoadIndex)[lane_id] : 0;
    constexpr int veclen = 2;
    constexpr int stride = kUnroll * veclen;

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        int32_t encV = reinterpret_cast<uint16_t const*>(data)[lane_id + j * kIndexGroupSize];
        int32_t q    = shfl(queryReg, i * kUnroll + j, WarpSize);
        compute_dist(dist, q, encV);
      }
    }
  }

  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int lane_id, const int dim, const int dimBlocks)
  {
    constexpr int veclen = 2;
    int loadDim          = dimBlocks + lane_id * veclen;
    int32_t queryReg = loadDim < dim ? reinterpret_cast<uint16_t const*>(query + loadDim)[0] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      int32_t enc = reinterpret_cast<uint16_t const*>(data + lane_id * veclen)[0];
      int32_t q   = shfl(queryReg, d / veclen, WarpSize);
      compute_dist(dist, q, enc);
    }
  }
};

template <int kUnroll, typename Lambda>
struct loadAndComputeDist<kUnroll, Lambda, 1, int8_t, int32_t> {
  Lambda compute_dist;
  int32_t& dist;
  __device__ __forceinline__ loadAndComputeDist(int32_t& dist, Lambda op)
    : dist(dist), compute_dist(op)
  {
  }

  __device__ __forceinline__ void runLoadShmemCompute(const int8_t* const& data,
                                                      const int8_t* query_shared,
                                                      int loadIndex,
                                                      int shmemIndex)
  {
#pragma unroll
    for (int j = 0; j < kUnroll; ++j) {
      compute_dist(dist, query_shared[shmemIndex + j], data[loadIndex + j * kIndexGroupSize]);
    }
  }

  __device__ __forceinline__ void runLoadShflAndCompute(const int8_t*& data,
                                                        const int8_t* query,
                                                        int baseLoadIndex,
                                                        const int lane_id)
  {
    constexpr int veclen = 1;
    constexpr int stride = kUnroll * veclen;
    int32_t queryReg     = query[baseLoadIndex + lane_id];

#pragma unroll
    for (int i = 0; i < WarpSize / stride; ++i, data += stride * kIndexGroupSize) {
#pragma unroll
      for (int j = 0; j < kUnroll; ++j) {
        compute_dist(
          dist, shfl(queryReg, i * kUnroll + j, WarpSize), data[lane_id + j * kIndexGroupSize]);
      }
    }
  }
  __device__ __forceinline__ void runLoadShflAndComputeRemainder(
    const int8_t*& data, const int8_t* query, const int lane_id, const int dim, const int dimBlocks)
  {
    constexpr int veclen = 1;
    const int loadDim    = dimBlocks + lane_id;
    int32_t queryReg     = loadDim < dim ? query[loadDim] : 0;
    for (int d = 0; d < dim - dimBlocks; d += veclen, data += kIndexGroupSize * veclen) {
      compute_dist(dist, shfl(queryReg, d, WarpSize), data[lane_id]);
    }
  }
};

// switch to dummy blocksort when Capacity is 0 this explicit dummy is chosen
// to support access to warpsort constants like ::queue_t::kDummy
template <int Capacity, bool Ascending, typename T, typename IdxT>
struct flat_block_sort {
  using type = matrix::detail::select::warpsort::
    block_sort<matrix::detail::select::warpsort::warp_sort_filtered, Capacity, Ascending, T, IdxT>;
};

template <typename T, bool Ascending, typename IdxT>
struct flat_block_sort<0, Ascending, T, IdxT>
  : ivf::detail::dummy_block_sort_t<T, IdxT, Ascending> {
  using type = ivf::detail::dummy_block_sort_t<T, IdxT, Ascending>;
};

template <int Capacity, bool Ascending, typename T, typename IdxT>
using block_sort_t = typename flat_block_sort<Capacity, Ascending, T, IdxT>::type;

/**
 * Scan clusters for nearest neighbors of the query vectors.
 * See `ivfflat_interleaved_scan` for more information.
 *
 * The clusters are stored in the interleaved index format described in ivf_flat_types.hpp.
 * For each query vector, a set of clusters is probed: the distance to each vector in the cluster is
 * calculated, and the top-k nearest neighbors are selected.
 *
 * @param compute_dist distance function
 * @param query_smem_elems number of dimensions of the query vector to fit in a shared memory of a
 * block; this number must be a multiple of `WarpSize * Veclen`.
 * @param[in] query a pointer to all queries in a row-major contiguous format [gridDim.y, dim]
 * @param[in] coarse_index a pointer to the cluster indices to search through [n_probes]
 * @param[in] list_indices index<T, IdxT>.indices
 * @param[in] list_data index<T, IdxT>.data
 * @param[in] list_sizes index<T, IdxT>.list_sizes
 * @param[in] list_offsets index<T, IdxT>.list_offsets
 * @param n_probes
 * @param k
 * @param dim
 * @param sample_filter
 * @param[out] neighbors
 * @param[out] distances
 */
template <int Capacity,
          int Veclen,
          bool Ascending,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterT,
          typename Lambda,
          typename PostLambda>
RAFT_KERNEL __launch_bounds__(kThreadsPerBlock)
  interleaved_scan_kernel(Lambda compute_dist,
                          PostLambda post_process,
                          const uint32_t query_smem_elems,
                          const T* query,
                          const uint32_t* coarse_index,
                          const T* const* list_data_ptrs,
                          const uint32_t* list_sizes,
                          const uint32_t queries_offset,
                          const uint32_t n_probes,
                          const uint32_t k,
                          const uint32_t max_samples,
                          const uint32_t* chunk_indices,
                          const uint32_t dim,
                          IvfSampleFilterT sample_filter,
                          uint32_t* neighbors,
                          float* distances)
{
  extern __shared__ __align__(256) uint8_t interleaved_scan_kernel_smem[];
  constexpr bool kManageLocalTopK = Capacity > 0;
  // Using shared memory for the (part of the) query;
  // This allows to save on global memory bandwidth when reading index and query
  // data at the same time.
  // Its size is `query_smem_elems`.
  T* query_shared = reinterpret_cast<T*>(interleaved_scan_kernel_smem);
  // Make the query input and output point to this block's shared query
  {
    const int query_id = blockIdx.y;
    query += query_id * dim;
    if constexpr (kManageLocalTopK) {
      neighbors += query_id * k * gridDim.x + blockIdx.x * k;
      distances += query_id * k * gridDim.x + blockIdx.x * k;
    } else {
      distances += query_id * uint64_t(max_samples);
    }
    chunk_indices += (n_probes * query_id);
    coarse_index += query_id * n_probes;
  }

  // Copy a part of the query into shared memory for faster processing
  copy_vectorized(query_shared, query, std::min(dim, query_smem_elems));
  __syncthreads();

  using local_topk_t = block_sort_t<Capacity, Ascending, float, uint32_t>;
  local_topk_t queue(k);
  {
    using align_warp  = Pow2<WarpSize>;
    const int lane_id = align_warp::mod(threadIdx.x);

    // How many full warps needed to compute the distance (without remainder)
    const uint32_t full_warps_along_dim = align_warp::roundDown(dim);

    const uint32_t shm_assisted_dim =
      (dim > query_smem_elems) ? query_smem_elems : full_warps_along_dim;

    // Every CUDA block scans one cluster at a time.
    for (int probe_id = blockIdx.x; probe_id < n_probes; probe_id += gridDim.x) {
      const uint32_t list_id = coarse_index[probe_id];  // The id of cluster(list)

      // The number of vectors in each cluster(list); [nlist]
      const uint32_t list_length = list_sizes[list_id];

      // The number of interleaved groups to be processed
      const uint32_t num_groups =
        align_warp::div(list_length + align_warp::Mask);  // ceildiv by power of 2

      uint32_t sample_offset = 0;
      if (probe_id > 0) { sample_offset = chunk_indices[probe_id - 1]; }
      assert(list_length == chunk_indices[probe_id] - sample_offset);
      assert(sample_offset + list_length <= max_samples);

      constexpr int kUnroll        = WarpSize / Veclen;
      constexpr uint32_t kNumWarps = kThreadsPerBlock / WarpSize;
      // Every warp reads WarpSize vectors and computes the distances to them.
      // Then, the distances and corresponding ids are distributed among the threads,
      // and each thread adds one (id, dist) pair to the filtering queue.
      for (uint32_t group_id = align_warp::div(threadIdx.x); group_id < num_groups;
           group_id += kNumWarps) {
        AccT dist = 0;
        // This is where this warp begins reading data (start position of an interleaved group)
        const T* data = list_data_ptrs[list_id] + (group_id * kIndexGroupSize) * dim;

        // This is the vector a given lane/thread handles
        const uint32_t vec_id = group_id * WarpSize + lane_id;
        const bool valid =
          vec_id < list_length && sample_filter(queries_offset + blockIdx.y, list_id, vec_id);

        // Process first shm_assisted_dim dimensions (always using shared memory)
        if (valid) {
          loadAndComputeDist<kUnroll, decltype(compute_dist), Veclen, T, AccT> lc(dist,
                                                                                  compute_dist);
          for (int pos = 0; pos < shm_assisted_dim;
               pos += WarpSize, data += kIndexGroupSize * WarpSize) {
            lc.runLoadShmemCompute(data, query_shared, lane_id, pos);
          }
        }

        if (dim > query_smem_elems) {
          // The default path - using shfl ops - for dimensions beyond query_smem_elems
          loadAndComputeDist<kUnroll, decltype(compute_dist), Veclen, T, AccT> lc(dist,
                                                                                  compute_dist);
          for (int pos = shm_assisted_dim; pos < full_warps_along_dim; pos += WarpSize) {
            lc.runLoadShflAndCompute(data, query, pos, lane_id);
          }
          lc.runLoadShflAndComputeRemainder(data, query, lane_id, dim, full_warps_along_dim);
        } else {
          // when  shm_assisted_dim == full_warps_along_dim < dim
          if (valid) {
            loadAndComputeDist<1, decltype(compute_dist), Veclen, T, AccT> lc(dist, compute_dist);
            for (int pos = full_warps_along_dim; pos < dim;
                 pos += Veclen, data += kIndexGroupSize * Veclen) {
              lc.runLoadShmemCompute(data, query_shared, lane_id, pos);
            }
          }
        }

        // Enqueue one element per thread
        const float val = valid ? static_cast<float>(dist) : local_topk_t::queue_t::kDummy;
        if constexpr (kManageLocalTopK) {
          queue.add(val, sample_offset + vec_id);
        } else {
          if (vec_id < list_length) distances[sample_offset + vec_id] = val;
        }
      }

      // fill up unused slots for current query
      if constexpr (!kManageLocalTopK) {
        if (probe_id + 1 == n_probes) {
          for (uint32_t i = threadIdx.x + sample_offset + list_length; i < max_samples;
               i += blockDim.x) {
            distances[i] = local_topk_t::queue_t::kDummy;
          }
        }
      }
    }
  }

  // finalize and store selected neighbours
  if constexpr (kManageLocalTopK) {
    __syncthreads();
    queue.done(interleaved_scan_kernel_smem);
    queue.store(distances, neighbors, post_process);
  }
}

/**
 *  Configure the gridDim.x to maximize GPU occupancy, but reduce the output size
 */
template <typename T>
uint32_t configure_launch_x(uint32_t numQueries, uint32_t n_probes, int32_t sMemSize, T func)
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
  return min_grid_x > n_probes ? n_probes : static_cast<uint32_t>(min_grid_x);
}

template <int Capacity,
          int Veclen,
          bool Ascending,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterT,
          typename Lambda,
          typename PostLambda>
void launch_kernel(Lambda lambda,
                   PostLambda post_process,
                   const index<T, IdxT>& index,
                   const T* queries,
                   const uint32_t* coarse_index,
                   const uint32_t num_queries,
                   const uint32_t queries_offset,
                   const uint32_t n_probes,
                   const uint32_t k,
                   const uint32_t max_samples,
                   const uint32_t* chunk_indices,
                   IvfSampleFilterT sample_filter,
                   uint32_t* neighbors,
                   float* distances,
                   uint32_t& grid_dim_x,
                   rmm::cuda_stream_view stream)
{
  RAFT_EXPECTS(Veclen == index.veclen(),
               "Configured Veclen does not match the index interleaving pattern.");
  constexpr auto kKernel   = interleaved_scan_kernel<Capacity,
                                                   Veclen,
                                                   Ascending,
                                                   T,
                                                   AccT,
                                                   IdxT,
                                                   IvfSampleFilterT,
                                                   Lambda,
                                                   PostLambda>;
  const int max_query_smem = 16384;
  int query_smem_elems =
    std::min<int>(max_query_smem / sizeof(T), Pow2<Veclen * WarpSize>::roundUp(index.dim()));
  int smem_size = query_smem_elems * sizeof(T);

  if constexpr (Capacity > 0) {
    constexpr int kSubwarpSize = std::min<int>(Capacity, WarpSize);
    auto block_merge_mem =
      raft::matrix::detail::select::warpsort::calc_smem_size_for_block_wide<float, IdxT>(
        kThreadsPerBlock / kSubwarpSize, k);
    smem_size += std::max<int>(smem_size, block_merge_mem);
  }

  // power-of-two less than cuda limit (for better addr alignment)
  constexpr uint32_t kMaxGridY = 32768;

  if (grid_dim_x == 0) {
    grid_dim_x = configure_launch_x(std::min(kMaxGridY, num_queries), n_probes, smem_size, kKernel);
    return;
  }

  for (uint32_t query_offset = 0; query_offset < num_queries; query_offset += kMaxGridY) {
    uint32_t grid_dim_y = std::min<uint32_t>(kMaxGridY, num_queries - query_offset);
    dim3 grid_dim(grid_dim_x, grid_dim_y, 1);
    dim3 block_dim(kThreadsPerBlock);
    RAFT_LOG_TRACE(
      "Launching the ivf-flat interleaved_scan_kernel (%d, %d, 1) x (%d, 1, 1), n_probes = %d, "
      "smem_size = %d",
      grid_dim.x,
      grid_dim.y,
      block_dim.x,
      n_probes,
      smem_size);
    kKernel<<<grid_dim, block_dim, smem_size, stream>>>(lambda,
                                                        post_process,
                                                        query_smem_elems,
                                                        queries,
                                                        coarse_index,
                                                        index.data_ptrs().data_handle(),
                                                        index.list_sizes().data_handle(),
                                                        queries_offset + query_offset,
                                                        n_probes,
                                                        k,
                                                        max_samples,
                                                        chunk_indices,
                                                        index.dim(),
                                                        sample_filter,
                                                        neighbors,
                                                        distances);
    queries += grid_dim_y * index.dim();
    if constexpr (Capacity > 0) {
      neighbors += grid_dim_y * grid_dim_x * k;
      distances += grid_dim_y * grid_dim_x * k;
    } else {
      distances += grid_dim_y * max_samples;
    }
    chunk_indices += grid_dim_y * n_probes;
    coarse_index += grid_dim_y * n_probes;
  }
}

template <int Veclen, typename T, typename AccT>
struct euclidean_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    const auto diff = x - y;
    acc += diff * diff;
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, uint8_t, uint32_t> {
  __device__ __forceinline__ void operator()(uint32_t& acc, uint32_t x, uint32_t y)
  {
    if constexpr (Veclen > 1) {
      const auto diff = __vabsdiffu4(x, y);
      acc             = dp4a(diff, diff, acc);
    } else {
      const auto diff = __usad(x, y, 0u);
      acc += diff * diff;
    }
  }
};

template <int Veclen>
struct euclidean_dist<Veclen, int8_t, int32_t> {
  __device__ __forceinline__ void operator()(int32_t& acc, int32_t x, int32_t y)
  {
    if constexpr (Veclen > 1) {
      // Note that we enforce here that the unsigned version of dp4a is used, because the difference
      // between two int8 numbers can be greater than 127 and therefore represented as a negative
      // number in int8. Casting from int8 to int32 would yield incorrect results, while casting
      // from uint8 to uint32 is correct.
      const auto diff = __vabsdiffs4(x, y);
      acc             = dp4a(diff, diff, static_cast<uint32_t>(acc));
    } else {
      const auto diff = x - y;
      acc += diff * diff;
    }
  }
};

template <int Veclen, typename T, typename AccT>
struct inner_prod_dist {
  __device__ __forceinline__ void operator()(AccT& acc, AccT x, AccT y)
  {
    if constexpr (Veclen > 1 && (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>)) {
      acc = dp4a(x, y, acc);
    } else {
      acc += x * y;
    }
  }
};

/** Select the distance computation function and forward the rest of the arguments. */
template <int Capacity,
          int Veclen,
          bool Ascending,
          typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterT,
          typename... Args>
void launch_with_fixed_consts(raft::distance::DistanceType metric, Args&&... args)
{
  switch (metric) {
    case raft::distance::DistanceType::L2Expanded:
    case raft::distance::DistanceType::L2Unexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterT,
                           euclidean_dist<Veclen, T, AccT>,
                           raft::identity_op>({}, {}, std::forward<Args>(args)...);
    case raft::distance::DistanceType::L2SqrtExpanded:
    case raft::distance::DistanceType::L2SqrtUnexpanded:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterT,
                           euclidean_dist<Veclen, T, AccT>,
                           raft::sqrt_op>({}, {}, std::forward<Args>(args)...);
    case raft::distance::DistanceType::InnerProduct:
      return launch_kernel<Capacity,
                           Veclen,
                           Ascending,
                           T,
                           AccT,
                           IdxT,
                           IvfSampleFilterT,
                           inner_prod_dist<Veclen, T, AccT>,
                           raft::identity_op>({}, {}, std::forward<Args>(args)...);
    // NB: update the description of `knn::ivf_flat::build` when adding here a new metric.
    default: RAFT_FAIL("The chosen distance metric is not supported (%d)", int(metric));
  }
}

/**
 * Lift the `capacity` and `veclen` parameters to the template level,
 * forward the rest of the arguments unmodified to `launch_interleaved_scan_kernel`.
 */
template <typename T,
          typename AccT,
          typename IdxT,
          typename IvfSampleFilterT,
          int Capacity = matrix::detail::select::warpsort::kMaxCapacity,
          int Veclen   = std::max<int>(1, 16 / sizeof(T))>
struct select_interleaved_scan_kernel {
  /**
   * Recursively reduce the `Capacity` and `Veclen` parameters until they match the
   * corresponding runtime arguments.
   * By default, this recursive process starts with maximum possible values of the
   * two parameters and ends with both values equal to 1.
   */
  template <typename... Args>
  static inline void run(int k_max, int veclen, bool select_min, Args&&... args)
  {
    if constexpr (Capacity > 0) {
      if (k_max == 0 || k_max > Capacity) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterT, 0, Veclen>::run(
          k_max, veclen, select_min, std::forward<Args>(args)...);
      }
    }
    if constexpr (Capacity > 1) {
      if (k_max * 2 <= Capacity) {
        return select_interleaved_scan_kernel<T,
                                              AccT,
                                              IdxT,
                                              IvfSampleFilterT,
                                              Capacity / 2,
                                              Veclen>::run(k_max,
                                                           veclen,
                                                           select_min,
                                                           std::forward<Args>(args)...);
      }
    }
    if constexpr (Veclen > 1) {
      if (veclen % Veclen != 0) {
        return select_interleaved_scan_kernel<T, AccT, IdxT, IvfSampleFilterT, Capacity, 1>::run(
          k_max, 1, select_min, std::forward<Args>(args)...);
      }
    }
    // NB: this is the limitation of the warpsort structures that use a huge number of
    //     registers (used in the main kernel here).
    RAFT_EXPECTS(Capacity == 0 || k_max == Capacity,
                 "Capacity must be either 0 or a power-of-two not bigger than the maximum "
                 "allowed size matrix::detail::select::warpsort::kMaxCapacity (%d).",
                 matrix::detail::select::warpsort::kMaxCapacity);
    RAFT_EXPECTS(
      veclen == Veclen,
      "Veclen must be power-of-two not bigger than the maximum allowed size for this data type.");
    if (select_min) {
      launch_with_fixed_consts<Capacity, Veclen, true, T, AccT, IdxT, IvfSampleFilterT>(
        std::forward<Args>(args)...);
    } else {
      launch_with_fixed_consts<Capacity, Veclen, false, T, AccT, IdxT, IvfSampleFilterT>(
        std::forward<Args>(args)...);
    }
  }
};

/**
 * @brief Configure and launch an appropriate template instance of the interleaved scan kernel.
 *
 * @tparam T value type
 * @tparam AccT accumulated type
 * @tparam IdxT type of the indices
 *
 * @param index previously built ivf-flat index
 * @param[in] queries device pointer to the query vectors [batch_size, dim]
 * @param[in] coarse_query_results device pointer to the cluster (list) ids [batch_size, n_probes]
 * @param n_queries batch size
 * @param[in] queries_offset
 *   An offset of the current query batch. It is used for feeding sample_filter with the
 *   correct query index.
 * @param metric type of the measured distance
 * @param n_probes number of nearest clusters to query
 * @param k number of nearest neighbors.
 *            NB: the maximum value of `k` is limited statically by `kMaxCapacity`.
 * @param select_min whether to select nearest (true) or furthest (false) points w.r.t. the given
 * metric.
 * @param[out] neighbors device pointer to the result indices for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[out] distances device pointer to the result distances for each query and cluster
 * [batch_size, grid_dim_x, k]
 * @param[inout] grid_dim_x number of blocks launched across all n_probes clusters;
 *               (one block processes one or more probes, hence: 1 <= grid_dim_x <= n_probes)
 * @param stream
 * @param sample_filter
 *   A filter that selects samples for a given query. Use an instance of none_ivf_sample_filter to
 *   provide a green light for every sample.
 */
template <typename T, typename AccT, typename IdxT, typename IvfSampleFilterT>
void ivfflat_interleaved_scan(const index<T, IdxT>& index,
                              const T* queries,
                              const uint32_t* coarse_query_results,
                              const uint32_t n_queries,
                              const uint32_t queries_offset,
                              const raft::distance::DistanceType metric,
                              const uint32_t n_probes,
                              const uint32_t k,
                              const uint32_t max_samples,
                              const uint32_t* chunk_indices,
                              const bool select_min,
                              IvfSampleFilterT sample_filter,
                              uint32_t* neighbors,
                              float* distances,
                              uint32_t& grid_dim_x,
                              rmm::cuda_stream_view stream)
{
  const int capacity = bound_by_power_of_two(k);

  auto filter_adapter = raft::neighbors::filtering::ivf_to_sample_filter(
    index.inds_ptrs().data_handle(), sample_filter);
  select_interleaved_scan_kernel<T, AccT, IdxT, decltype(filter_adapter)>::run(capacity,
                                                                               index.veclen(),
                                                                               select_min,
                                                                               metric,
                                                                               index,
                                                                               queries,
                                                                               coarse_query_results,
                                                                               n_queries,
                                                                               queries_offset,
                                                                               n_probes,
                                                                               k,
                                                                               max_samples,
                                                                               chunk_indices,
                                                                               filter_adapter,
                                                                               neighbors,
                                                                               distances,
                                                                               grid_dim_x,
                                                                               stream);
}

}  // namespace raft::neighbors::ivf_flat::detail
