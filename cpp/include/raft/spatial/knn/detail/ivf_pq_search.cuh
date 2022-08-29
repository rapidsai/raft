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

#include "../ivf_pq_types.hpp"
#include "ann_utils.cuh"
#include "topk/warpsort_topk.cuh"

#include <raft/common/device_loads_stores.cuh>
#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/mdarray.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/device_atomics.cuh>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/pow2_utils.cuh>
#include <raft/vectorized.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <cub/cub.cuh>
#include <thrust/sequence.h>

#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <optional>

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

template <unsigned expBitLen>
struct fp_8bit;

template <unsigned expBitLen>
__device__ __host__ fp_8bit<expBitLen> __float2fp_8bit(const float v);
template <unsigned expBitLen>
__device__ __host__ float __fp_8bit2float(const fp_8bit<expBitLen>& v);

template <unsigned expBitLen>
struct fp_8bit {
  uint8_t bitstring;

  __device__ __host__ fp_8bit(const uint8_t bs) { bitstring = bs; }
  __device__ __host__ fp_8bit(const float fp)
  {
    bitstring = __float2fp_8bit<expBitLen>(fp).bitstring;
  }
  __device__ __host__ fp_8bit<expBitLen>& operator=(const float fp)
  {
    bitstring = __float2fp_8bit<expBitLen>(fp).bitstring;
    return *this;
  }

  __device__ __host__ operator float() const { return __fp_8bit2float(*this); }
};

// Since __float_as_uint etc can not be used in host codes,
// these converters are needed for test.
union cvt_fp_32bit {
  float fp;
  uint32_t bs;
};
union cvt_fp_16bit {
  half fp;
  uint16_t bs;
};

// Type converters
template <unsigned expBitLen>
__device__ __host__ fp_8bit<expBitLen> __float2fp_8bit(const float v)
{
  if (v < 1. / (1u << ((1u << (expBitLen - 1)) - 1)))
    return fp_8bit<expBitLen>{static_cast<uint8_t>(0)};
  return fp_8bit<expBitLen>{static_cast<uint8_t>(
    (cvt_fp_32bit{.fp = v}.bs + (((1u << (expBitLen - 1)) - 1) << 23) - 0x3f800000u) >>
    (15 + expBitLen))};
}

template <unsigned expBitLen>
__device__ __host__ float __fp_8bit2float(const fp_8bit<expBitLen>& v)
{
  return cvt_fp_32bit{.bs = ((v.bitstring << (15 + expBitLen)) +
                             (0x3f800000u | (0x00400000u >> (8 - expBitLen))) -
                             (((1u << (expBitLen - 1)) - 1) << 23))}
    .fp;
}

#define NUM_THREADS      1024  // DO NOT CHANGE
#define STATE_BIT_LENGTH 8     // 0: state not used,  8: state used
#define MAX_VEC_LENGTH   8     // 1, 2, 4 or 8

__device__ inline uint16_t convert(uint16_t x)
{
  if (x & 0x8000) {
    return x ^ 0xffff;
  } else {
    return x ^ 0x8000;
  }
}

__device__ inline uint32_t convert(uint32_t x)
{
  if (x & 0x80000000) {
    return x ^ 0xffffffff;
  } else {
    return x ^ 0x80000000;
  }
}

struct u16_vector {
  ushort1 x1;
  ushort2 x2;
  ushort4 x4;
  uint4 x8;
};

struct u32_vector {
  uint1 x1;
  uint2 x2;
  uint4 x4;
  ulonglong4 x8;
};

template <int vecLen>
__device__ inline void load_u16_vector(struct u16_vector& vec, const uint16_t* x, int i)
{
  if (vecLen == 1) {
    vec.x1 = ((ushort1*)(x + i))[0];
  } else if (vecLen == 2) {
    vec.x2 = ((ushort2*)(x + i))[0];
  } else if (vecLen == 4) {
    vec.x4 = ((ushort4*)(x + i))[0];
  } else if (vecLen == 8) {
    vec.x8 = ((uint4*)(x + i))[0];
  }
}

template <int vecLen>
__device__ inline void load_u32_vector(struct u32_vector& vec, const uint32_t* x, int i)
{
  if (vecLen == 1) {
    vec.x1 = ((uint1*)(x + i))[0];
  } else if (vecLen == 2) {
    vec.x2 = ((uint2*)(x + i))[0];
  } else if (vecLen == 4) {
    vec.x4 = ((uint4*)(x + i))[0];
  } else if (vecLen == 8) {
    vec.x8 = ((ulonglong4*)(x + i))[0];
  }
}

template <int vecLen>
__device__ inline uint16_t get_element_from_u16_vector(struct u16_vector& vec, int i)
{
  uint16_t xi;
  if (vecLen == 1) {
    xi = convert(vec.x1.x);
  } else if (vecLen == 2) {
    if (i == 0)
      xi = convert(vec.x2.x);
    else
      xi = convert(vec.x2.y);
  } else if (vecLen == 4) {
    if (i == 0)
      xi = convert(vec.x4.x);
    else if (i == 1)
      xi = convert(vec.x4.y);
    else if (i == 2)
      xi = convert(vec.x4.z);
    else
      xi = convert(vec.x4.w);
  } else if (vecLen == 8) {
    if (i == 0)
      xi = convert((uint16_t)(vec.x8.x & 0xffff));
    else if (i == 1)
      xi = convert((uint16_t)(vec.x8.x >> 16));
    else if (i == 2)
      xi = convert((uint16_t)(vec.x8.y & 0xffff));
    else if (i == 3)
      xi = convert((uint16_t)(vec.x8.y >> 16));
    else if (i == 4)
      xi = convert((uint16_t)(vec.x8.z & 0xffff));
    else if (i == 5)
      xi = convert((uint16_t)(vec.x8.z >> 16));
    else if (i == 6)
      xi = convert((uint16_t)(vec.x8.w & 0xffff));
    else
      xi = convert((uint16_t)(vec.x8.w >> 16));
  }
  return xi;
}

template <int vecLen>
__device__ inline uint32_t get_element_from_u32_vector(struct u32_vector& vec, int i)
{
  uint32_t xi;
  if (vecLen == 1) {
    xi = convert(vec.x1.x);
  } else if (vecLen == 2) {
    if (i == 0)
      xi = convert(vec.x2.x);
    else
      xi = convert(vec.x2.y);
  } else if (vecLen == 4) {
    if (i == 0)
      xi = convert(vec.x4.x);
    else if (i == 1)
      xi = convert(vec.x4.y);
    else if (i == 2)
      xi = convert(vec.x4.z);
    else
      xi = convert(vec.x4.w);
  } else if (vecLen == 8) {
    if (i == 0)
      xi = convert((uint32_t)(vec.x8.x & 0xffffffff));
    else if (i == 1)
      xi = convert((uint32_t)(vec.x8.x >> 32));
    else if (i == 2)
      xi = convert((uint32_t)(vec.x8.y & 0xffffffff));
    else if (i == 3)
      xi = convert((uint32_t)(vec.x8.y >> 32));
    else if (i == 4)
      xi = convert((uint32_t)(vec.x8.z & 0xffffffff));
    else if (i == 5)
      xi = convert((uint32_t)(vec.x8.z >> 32));
    else if (i == 6)
      xi = convert((uint32_t)(vec.x8.w & 0xffffffff));
    else
      xi = convert((uint32_t)(vec.x8.w >> 32));
  }
  return xi;
}

template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(NUM_THREADS, 1024 / NUM_THREADS) __global__
  void kern_topk_cg_11(uint32_t topk,
                       uint32_t size_batch,
                       uint32_t max_len_x,
                       uint32_t* len_x,     // [size_batch,]
                       const uint32_t* _x,  // [size_batch, max_len_x,]
                       uint8_t* _state,     // [size_batch, max_len_x / 8,]
                       uint32_t* _labels,   // [size_batch, topk,]
                       uint32_t* _count     // [size_batch, 5 * 1024,]
  )
{
  __shared__ uint32_t smem[2048 + 6];
  uint32_t* best_index = &(smem[2048]);
  uint32_t* best_csum  = &(smem[2048 + 3]);
  typedef cub::BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  namespace cg        = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  uint32_t i_batch    = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == nullptr) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x * gridDim.x;
  uint32_t thread_id   = threadIdx.x + (blockDim_x * blockIdx.x);

  const uint32_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = nullptr;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = raft::ceildiv(max_len_x, num_threads);
    uint32_t numState_perThread  = raft::ceildiv<uint32_t>(numSample_perThread, stateBitLen);
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 6) { smem[2048 + threadIdx.x] = 0; }

  uint32_t* count = _count + (5 * 1024 * i_batch);
  for (int i = thread_id; i < 5 * 1024; i += num_threads) {
    count[i] = 0;
  }
  cg::sync(grid);

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (21 - 11 * j);
    for (int i = threadIdx.x; i < 2048; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 2048) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 2048) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 2048; i += blockDim_x) {
      if (smem[i] > 0) { atomicAdd(&(count[i + (2048 * j)]), smem[i]); }
    }
    cg::sync(grid);

    constexpr int n_data = 2048 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = count[i + (n_data * threadIdx.x) + (2048 * j)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] >= topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
  }

  {
    uint32_t j = 2;
    for (int i = threadIdx.x; i < 1024; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8) {
        iState = state[thread_id + (num_threads * ii)];
        if (iState == (uint8_t)0xff) continue;
      }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          if ((stateBitLen == 8) && (iState & mask)) continue;
          uint32_t xi = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold);  // 0 <= k
            if (k >= 1024) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 1024) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 1024; i += blockDim_x) {
      if (smem[i] > 0) { atomicAdd(&(count[i + (2048 * j)]), smem[i]); }
    }
    cg::sync(grid);

    constexpr int n_data = 1024 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = count[i + (n_data * threadIdx.x) + (2048 * j)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] >= topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += best_index[j];
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u32_vector vec;
      load_u32_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u32_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[2048] < topk)) {
          if (count_below + atomicAdd(&count[2048], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }
}

template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(NUM_THREADS, 1024 / NUM_THREADS) __global__
  void kern_topk_cta_11(uint32_t topk,
                        uint32_t size_batch,
                        uint32_t max_len_x,
                        uint32_t* len_x,     // [size_batch, max_len_x,]
                        const uint32_t* _x,  // [size_batch, max_len_x,]
                        uint8_t* _state,     // [size_batch, max_len_x / 8,]
                        uint32_t* _labels    // [size_batch, topk,]
  )
{
  __shared__ uint32_t smem[2048 + 3 + 3 + 2];
  uint32_t* best_index = &(smem[2048]);
  uint32_t* best_csum  = &(smem[2048 + 3]);
  uint32_t* count      = &(smem[2048 + 6]);
  typedef cub::BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  uint32_t i_batch = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == nullptr) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x;
  uint32_t thread_id   = threadIdx.x;

  const uint32_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = nullptr;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = raft::ceildiv(max_len_x, num_threads);
    uint32_t numState_perThread  = raft::ceildiv<uint32_t>(numSample_perThread, stateBitLen);
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 8) { smem[2048 + threadIdx.x] = 0; }

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (21 - 11 * j);
    for (int i = threadIdx.x; i < 2048; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 2048) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 2048) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    constexpr int n_data = 2048 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = smem[i + (n_data * threadIdx.x)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] > topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
    if (count_below == topk) break;
  }

  {
    uint32_t j = 2;
    for (int i = threadIdx.x; i < 1024; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8) {
        iState = state[thread_id + (num_threads * ii)];
        if (iState == (uint8_t)0xff) continue;
      }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u32_vector x_vec;
        load_u32_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          if ((stateBitLen == 8) && (iState & mask)) continue;
          uint32_t xi = get_element_from_u32_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold);  // 0 <= k
            if (k >= 1024) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 1024) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    constexpr int n_data = 1024 / blockDim_x;
    uint32_t csum[n_data];
#pragma unroll
    for (int i = 0; i < n_data; i++) {
      csum[i] = smem[i + (n_data * threadIdx.x)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

#pragma unroll
    for (int i = n_data - 1; i >= 0; i--) {
      if (count_below + csum[i] > topk) continue;
      uint32_t index = i + (n_data * threadIdx.x);
      atomicMax(&(best_index[j]), index);
      atomicMax(&(best_csum[j]), csum[i]);
      break;
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += best_index[j];
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u32_vector vec;
      load_u32_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u32_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[1] < topk)) {
          if (count_below + atomicAdd(&count[1], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }
}

template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(NUM_THREADS, 1024 / NUM_THREADS) __global__
  void kern_topk_cg_8(uint32_t topk,
                      uint32_t size_batch,
                      uint32_t max_len_x,
                      uint32_t* len_x,     // [size_batch,]
                      const uint16_t* _x,  // [size_batch, max_len_x,]
                      uint8_t* _state,     // [size_batch, max_len_x / 8,]
                      uint32_t* _labels,   // [size_batch, topk,]
                      uint32_t* _count     // [size_batch, 5 * 1024,]
  )
{
  __shared__ uint32_t smem[256 + 4];
  uint32_t* best_index = &(smem[256]);
  uint32_t* best_csum  = &(smem[256 + 2]);
  typedef cub::BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  namespace cg        = cooperative_groups;
  cg::grid_group grid = cg::this_grid();
  uint32_t i_batch    = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == nullptr) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x * gridDim.x;
  uint32_t thread_id   = threadIdx.x + (blockDim_x * blockIdx.x);

  const uint16_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = nullptr;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = raft::ceildiv(max_len_x, num_threads);
    uint32_t numState_perThread  = raft::ceildiv<uint32_t>(numSample_perThread, stateBitLen);
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 4) { smem[256 + threadIdx.x] = 0; }

  uint32_t* count = _count + (2 * 256 * i_batch);
  for (int i = thread_id; i < 2 * 256; i += num_threads) {
    count[i] = 0;
  }
  cg::sync(grid);

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (8 - 8 * j);
    for (int i = threadIdx.x; i < 256; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u16_vector x_vec;
        load_u16_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u16_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 256) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 256) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    for (int i = threadIdx.x; i < 256; i += blockDim_x) {
      if (smem[i] > 0) { atomicAdd(&(count[i + (256 * j)]), smem[i]); }
    }
    cg::sync(grid);

    uint32_t csum[1];
    csum[0] = 0;
    if (threadIdx.x < 256) { csum[0] = count[threadIdx.x + (256 * j)]; }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

    if (threadIdx.x < 256) {
      if (count_below + csum[0] < topk) {
        uint32_t index = threadIdx.x;
        atomicMax(&(best_index[j]), index);
        atomicMax(&(best_csum[j]), csum[0]);
      }
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u16_vector vec;
      load_u16_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u16_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[256] < topk)) {
          if (count_below + atomicAdd(&count[256], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }
}

//
template <int blockDim_x, int stateBitLen, int vecLen>
__launch_bounds__(NUM_THREADS, 1024 / NUM_THREADS) __global__
  void kern_topk_cta_8(uint32_t topk,
                       uint32_t size_batch,
                       uint32_t max_len_x,
                       uint32_t* len_x,     // [size_batch, max_len_x,]
                       const uint16_t* _x,  // [size_batch, max_len_x,]
                       uint8_t* _state,     // [size_batch, max_len_x / 8,]
                       uint32_t* _labels    // [size_batch, topk,]
  )
{
  __shared__ uint32_t smem[256 + 6];
  uint32_t* best_index = &(smem[256]);
  uint32_t* best_csum  = &(smem[256 + 2]);
  uint32_t* count      = &(smem[256 + 4]);
  typedef cub::BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  uint32_t i_batch = blockIdx.y;
  if (i_batch >= size_batch) return;

  uint32_t nx;
  if (len_x == nullptr) {
    nx = max_len_x;
  } else {
    nx = len_x[i_batch];
  }

  uint32_t num_threads = blockDim_x;
  uint32_t thread_id   = threadIdx.x;

  const uint16_t* x = _x + (max_len_x * i_batch);
  uint8_t* state    = nullptr;
  if (stateBitLen == 8) {
    uint32_t numSample_perThread = raft::ceildiv(max_len_x, num_threads);
    uint32_t numState_perThread  = raft::ceildiv<uint32_t>(numSample_perThread, stateBitLen);
    state                        = _state + (numState_perThread * num_threads * i_batch);
  }
  uint32_t* labels = _labels + (topk * i_batch);
  if (threadIdx.x < 6) { smem[256 + threadIdx.x] = 0; }

  uint32_t count_below = 0;
  uint32_t threshold   = 0;

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() < topk".
  //
  for (int j = 0; j < 2; j += 1) {
    uint32_t shift = (8 - 8 * j);
    for (int i = threadIdx.x; i < 256; i += blockDim_x) {
      smem[i] = 0;
    }
    __syncthreads();

    int ii = 0;
    for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
      uint8_t iState = 0;
      if (stateBitLen == 8 && j > 0) { iState = state[thread_id + (num_threads * ii)]; }
#pragma unroll
      for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
        int iv = i + (num_threads * v);
        if (iv >= nx) break;

        struct u16_vector x_vec;
        load_u16_vector<vecLen>(x_vec, x, iv);
#pragma unroll
        for (int u = 0; u < vecLen; u++) {
          int ivu = iv + u;
          if (ivu >= nx) break;

          uint8_t mask = (uint8_t)0x1 << (v + u);
          uint32_t xi  = get_element_from_u16_vector<vecLen>(x_vec, u);
          if (xi < threshold) {
            if (stateBitLen == 8) {
              labels[atomicAdd(&count[0], 1)] = ivu;
              iState |= mask;
            }
          } else {
            uint32_t k = (xi - threshold) >> shift;  // 0 <= k
            if (k >= 256) {
              if (stateBitLen == 8) { iState |= mask; }
            } else if (k + 1 < 256) {
              atomicAdd(&(smem[k + 1]), 1);
            }
          }
        }
      }
      if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
    }
    __syncthreads();

    uint32_t csum[1];
    if (threadIdx.x < 256) { csum[0] = smem[threadIdx.x]; }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);

    if (threadIdx.x < 256) {
      if (count_below + csum[0] < topk) {
        uint32_t index = threadIdx.x;
        atomicMax(&(best_index[j]), index);
        atomicMax(&(best_csum[j]), csum[0]);
      }
    }
    __syncthreads();

    count_below += best_csum[j];
    threshold += (best_index[j] << shift);
    if (count_below == topk) break;
  }

  //
  // Get labels that satifies "x[i] < threshold".
  //
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u16_vector vec;
      load_u16_vector<vecLen>(vec, x, iv);
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;
        uint32_t xi = get_element_from_u16_vector<vecLen>(vec, u);
        if (xi < threshold) {
          labels[atomicAdd(&count[0], 1)] = ivu;
        } else if ((xi == threshold) && (count_below + count[1] < topk)) {
          if (count_below + atomicAdd(&count[1], 1) < topk) {
            labels[atomicAdd(&count[0], 1)] = ivu;
          }
        }
      }
    }
  }
}

inline size_t _cuann_find_topk_bufferSize(const handle_t& handle,
                                          uint32_t topK,
                                          uint32_t sizeBatch,
                                          uint32_t maxSamples,
                                          cudaDataType_t sampleDtype = CUDA_R_32F)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  static_assert(stateBitLen == 0 || stateBitLen == 8);

  size_t workspaceSize = 0;
  // count
  if (sampleDtype == CUDA_R_16F) {
    workspaceSize += Pow2<128>::roundUp(sizeof(uint32_t) * sizeBatch * 2 * 256);
  } else {
    workspaceSize += Pow2<128>::roundUp(sizeof(uint32_t) * sizeBatch * 5 * 1024);
  }
  // state
  if (stateBitLen == 8) {
    // (*) Each thread has at least one array element for state
    uint32_t numBlocks_perBatch = (getMultiProcessorCount() * 2 + sizeBatch) / sizeBatch;

    uint32_t numThreads_perBatch = numThreads * numBlocks_perBatch;
    uint32_t numSample_perThread = raft::ceildiv(maxSamples, numThreads_perBatch);
    uint32_t numState_perThread  = raft::ceildiv<uint32_t>(numSample_perThread, stateBitLen);
    workspaceSize +=
      Pow2<128>::roundUp(sizeof(uint8_t) * numState_perThread * numThreads_perBatch * sizeBatch);
  }

  size_t workspaceSize2 = 0;
  // offsets
  workspaceSize2 += Pow2<128>::roundUp(sizeof(int) * (sizeBatch + 1));
  // keys_in, keys_out, values_out
  workspaceSize2 += Pow2<128>::roundUp(sizeof(float) * sizeBatch * topK);
  workspaceSize2 += Pow2<128>::roundUp(sizeof(float) * sizeBatch * topK);
  workspaceSize2 += Pow2<128>::roundUp(sizeof(uint32_t) * sizeBatch * topK);
  // cub_ws
  size_t cub_ws_size = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                           cub_ws_size,
                                           (float*)nullptr,
                                           (float*)nullptr,
                                           (uint32_t*)nullptr,
                                           (uint32_t*)nullptr,
                                           sizeBatch * topK,
                                           sizeBatch,
                                           (int*)nullptr,
                                           (int*)nullptr);
  workspaceSize2 += Pow2<128>::roundUp(cub_ws_size);
  workspaceSize = max(workspaceSize, workspaceSize2);

  return workspaceSize;
}

int _get_vecLen(uint32_t maxSamples, int maxVecLen = MAX_VEC_LENGTH)
{
  int vecLen = min(maxVecLen, MAX_VEC_LENGTH);
  while ((maxSamples % vecLen) != 0) {
    vecLen /= 2;
  }
  return vecLen;
}

__global__ void _sort_topk_prep(uint32_t sizeBatch,
                                uint32_t topK,
                                uint32_t maxSamples,
                                const uint32_t* labels,  // [sizeBatch, topK]
                                const float* samples,    // [sizeBatch, maxSamples]
                                int* offsets,            // [sizeBatch + 1]
                                float* outputs           // [sizeBatch, topK]
)
{
  uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid < sizeBatch + 1) { offsets[tid] = tid * topK; }
  if (tid < sizeBatch * topK) {
    uint32_t label  = labels[tid];
    uint32_t iBatch = tid / topK;
    float value     = samples[label + (maxSamples * iBatch)];
    outputs[tid]    = value;
  }
}

inline void _cuann_find_topk(const handle_t& handle,
                             uint32_t topK,
                             uint32_t sizeBatch,
                             uint32_t maxSamples,
                             uint32_t* numSamples,  // [sizeBatch,]
                             const float* samples,  // [sizeBatch, maxSamples,]
                             uint32_t* labels,      // [sizeBatch, topK,]
                             rmm::mr::device_memory_resource* mr,
                             bool sort = false)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  static_assert(stateBitLen == 0 || stateBitLen == 8);
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  RAFT_CUDA_TRY(
    cudaMemsetAsync(labels, 0xff, sizeof(uint32_t) * sizeBatch * topK, handle.get_stream()));
#endif

  // Limit the maximum value of vecLen to 4. In the case of FP32,
  // setting vecLen = 8 in cg_kernel causes too much register usage.
  int vecLen = _get_vecLen(maxSamples, 4);
  void* cg_kernel;
  if (vecLen == 4) {
    cg_kernel = (void*)kern_topk_cg_11<numThreads, stateBitLen, 4>;
  } else if (vecLen == 2) {
    cg_kernel = (void*)kern_topk_cg_11<numThreads, stateBitLen, 2>;
  } else if (vecLen == 1) {
    cg_kernel = (void*)kern_topk_cg_11<numThreads, stateBitLen, 1>;
  }

  int numBlocksPerSm_topk;
  size_t dynamicSMemSize = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocksPerSm_topk, cg_kernel, numThreads, dynamicSMemSize));
  int numBlocks_perBatch = raft::ceildiv<uint32_t>(maxSamples, numThreads * vecLen);
  int numBlocks =
    min(numBlocks_perBatch * sizeBatch, getMultiProcessorCount() * numBlocksPerSm_topk);
  numBlocks_perBatch = max(numBlocks / sizeBatch, 1);
  if (maxSamples <= numThreads * 10) {
    // When number of sample is small, using multiple thread-blocks does not
    // improve performance, in which case cta_kernel is used. Tentatively,
    // "numThreads * 10" is used as the threshold, but this may be better
    // determined by auto-tuning, etc.
    numBlocks_perBatch = 1;
  }
  rmm::device_buffer workspace(
    _cuann_find_topk_bufferSize(handle, topK, sizeBatch, maxSamples, CUDA_R_32F),
    handle.get_stream(),
    mr);
  uint32_t* count = reinterpret_cast<uint32_t*>(workspace.data());
  uint8_t* state  = nullptr;
  if (stateBitLen == 8) {
    state = (uint8_t*)count + Pow2<128>::roundUp(sizeof(uint32_t) * sizeBatch * 5 * 1024);
  }

  dim3 threads(numThreads, 1, 1);
  dim3 blocks(numBlocks_perBatch, sizeBatch, 1);
  if (numBlocks_perBatch <= 1) {
    void (*cta_kernel)(
      uint32_t, uint32_t, uint32_t, uint32_t*, const uint32_t*, uint8_t*, uint32_t*);
    int vecLen = _get_vecLen(maxSamples);
    if (vecLen == 8) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 8>;
    } else if (vecLen == 4) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 4>;
    } else if (vecLen == 2) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 2>;
    } else if (vecLen == 1) {
      cta_kernel = kern_topk_cta_11<numThreads, stateBitLen, 1>;
    } else {
      RAFT_FAIL("Unexpected vecLen (%d)", vecLen);
    }
    cta_kernel<<<blocks, threads, 0, handle.get_stream()>>>(
      topK, sizeBatch, maxSamples, numSamples, (const uint32_t*)samples, state, labels);
  } else {
    void* args[9];
    args[0] = {&(topK)};
    args[1] = {&(sizeBatch)};
    args[2] = {&(maxSamples)};
    args[3] = {&(numSamples)};
    args[4] = {&(samples)};
    args[5] = {&(state)};
    args[6] = {&(labels)};
    args[7] = {&(count)};
    args[8] = {nullptr};
    RAFT_CUDA_TRY(
      cudaLaunchCooperativeKernel((void*)cg_kernel, blocks, threads, args, 0, handle.get_stream()));
  }
  if (!sort) { return; }

  // offsets: [sizeBatch + 1]
  // keys_in, keys_out, values_out: [sizeBatch, topK]
  int* offsets   = reinterpret_cast<int*>(workspace.data());
  float* keys_in = (float*)((uint8_t*)offsets + Pow2<128>::roundUp(sizeof(int) * (sizeBatch + 1)));
  float* keys_out =
    (float*)((uint8_t*)keys_in + Pow2<128>::roundUp(sizeof(float) * sizeBatch * topK));
  uint32_t* values_out =
    (uint32_t*)((uint8_t*)keys_out + Pow2<128>::roundUp(sizeof(float) * sizeBatch * topK));
  void* cub_ws =
    (void*)((uint8_t*)values_out + Pow2<128>::roundUp(sizeof(uint32_t) * sizeBatch * topK));

  dim3 stp_threads(128, 1, 1);
  dim3 stp_blocks(raft::ceildiv(max(sizeBatch + 1, sizeBatch * topK), stp_threads.x), 1, 1);
  _sort_topk_prep<<<stp_blocks, stp_threads, 0, handle.get_stream()>>>(
    sizeBatch, topK, maxSamples, labels, samples, offsets, keys_in);

  size_t cub_ws_size = 0;
  cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                           cub_ws_size,
                                           keys_in,
                                           keys_out,
                                           labels,
                                           values_out,
                                           sizeBatch * topK,
                                           sizeBatch,
                                           offsets,
                                           offsets + 1);

  cub::DeviceSegmentedRadixSort::SortPairs(cub_ws,
                                           cub_ws_size,
                                           keys_in,
                                           keys_out,
                                           labels,
                                           values_out,
                                           sizeBatch * topK,
                                           sizeBatch,
                                           offsets,
                                           offsets + 1,
                                           (int)0,
                                           (int)(sizeof(float) * 8),
                                           handle.get_stream());

  raft::copy(labels, values_out, sizeBatch * topK, handle.get_stream());
}

inline void _cuann_find_topk(const handle_t& handle,
                             uint32_t topK,
                             uint32_t sizeBatch,
                             uint32_t maxSamples,
                             uint32_t* numSamples,  // [sizeBatch,]
                             const half* samples,   // [sizeBatch, maxSamples,]
                             uint32_t* labels,      // [sizeBatch, topK,]
                             rmm::mr::device_memory_resource* mr,
                             bool sort = false)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  static_assert(stateBitLen == 0 || stateBitLen == 8);
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  RAFT_CUDA_TRY(
    cudaMemsetAsync(labels, 0xff, sizeof(uint32_t) * sizeBatch * topK, handle.get_stream()));
#endif

  int vecLen = _get_vecLen(maxSamples);
  void* cg_kernel;
  if (vecLen == 8) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 8>;
  } else if (vecLen == 4) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 4>;
  } else if (vecLen == 2) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 2>;
  } else if (vecLen == 1) {
    cg_kernel = (void*)kern_topk_cg_8<numThreads, stateBitLen, 1>;
  }

  int numBlocksPerSm_topk;
  RAFT_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm_topk, cg_kernel, numThreads, 0));
  int numBlocks_perBatch = raft::ceildiv<uint32_t>(maxSamples, numThreads * vecLen);
  int numBlocks =
    min(numBlocks_perBatch * sizeBatch, getMultiProcessorCount() * numBlocksPerSm_topk);
  numBlocks_perBatch = max(numBlocks / sizeBatch, 1);
  if (maxSamples <= numThreads * 10) {
    // When number of sample is small, using multiple thread-blocks does not
    // improve performance, in which case cta_kernel is used. Tentatively,
    // "numThreads * 10" is used as the threshold, but this may be better
    // determined by auto-tuning, etc.
    numBlocks_perBatch = 1;
  }

  rmm::device_buffer workspace(
    _cuann_find_topk_bufferSize(handle, topK, sizeBatch, maxSamples, CUDA_R_16F),
    handle.get_stream(),
    mr);
  uint32_t* count = reinterpret_cast<uint32_t*>(workspace.data());
  uint8_t* state  = nullptr;
  if (stateBitLen == 8) {
    state = (uint8_t*)count + Pow2<128>::roundUp(sizeof(uint32_t) * sizeBatch * 2 * 256);
  }

  dim3 threads(numThreads, 1, 1);
  dim3 blocks(numBlocks_perBatch, sizeBatch, 1);
  if (numBlocks_perBatch <= 1) {
    void (*cta_kernel)(
      uint32_t, uint32_t, uint32_t, uint32_t*, const uint16_t*, uint8_t*, uint32_t*);
    int vecLen = _get_vecLen(maxSamples);
    if (vecLen == 8) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 8>;
    } else if (vecLen == 4) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 4>;
    } else if (vecLen == 2) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 2>;
    } else if (vecLen == 1) {
      cta_kernel = kern_topk_cta_8<numThreads, stateBitLen, 1>;
    } else {
      RAFT_FAIL("Unexpected vecLen (%d)", vecLen);
    }
    cta_kernel<<<blocks, threads, 0, handle.get_stream()>>>(
      topK, sizeBatch, maxSamples, numSamples, (const uint16_t*)samples, state, labels);
  } else {
    void* args[9];
    args[0] = {&(topK)};
    args[1] = {&(sizeBatch)};
    args[2] = {&(maxSamples)};
    args[3] = {&(numSamples)};
    args[4] = {&(samples)};
    args[5] = {&(state)};
    args[6] = {&(labels)};
    args[7] = {&(count)};
    args[8] = {nullptr};
    RAFT_CUDA_TRY(
      cudaLaunchCooperativeKernel((void*)cg_kernel, blocks, threads, args, 0, handle.get_stream()));
  }
}

__device__ inline uint32_t warp_scan(uint32_t x)
{
  uint32_t y;
  y = __shfl_up_sync(0xffffffff, x, 1);
  if (threadIdx.x % 32 >= 1) x += y;
  y = __shfl_up_sync(0xffffffff, x, 2);
  if (threadIdx.x % 32 >= 2) x += y;
  y = __shfl_up_sync(0xffffffff, x, 4);
  if (threadIdx.x % 32 >= 4) x += y;
  y = __shfl_up_sync(0xffffffff, x, 8);
  if (threadIdx.x % 32 >= 8) x += y;
  y = __shfl_up_sync(0xffffffff, x, 16);
  if (threadIdx.x % 32 >= 16) x += y;
  return x;
}

__device__ inline uint32_t thread_block_scan(uint32_t x, uint32_t* smem)
{
  x = warp_scan(x);
  __syncthreads();
  if (threadIdx.x % 32 == 31) { smem[threadIdx.x / 32] = x; }
  __syncthreads();
  if (threadIdx.x < 32) { smem[threadIdx.x] = warp_scan(smem[threadIdx.x]); }
  __syncthreads();
  if (threadIdx.x / 32 > 0) { x += smem[threadIdx.x / 32 - 1]; }
  __syncthreads();
  return x;
}

__global__ void ivfpq_make_chunk_index_ptr(
  uint32_t numProbes,
  uint32_t sizeBatch,
  const uint32_t* cluster_offsets,        // [n_clusters + 1,]
  const uint32_t* _clusterLabelsToProbe,  // [sizeBatch, numProbes,]
  uint32_t* _chunkIndexPtr,               // [sizeBetch, numProbes,]
  uint32_t* numSamples                    // [sizeBatch,]
)
{
  __shared__ uint32_t smem_temp[32];
  __shared__ uint32_t smem_base[2];

  uint32_t iBatch = blockIdx.x;
  if (iBatch >= sizeBatch) return;
  const uint32_t* clusterLabelsToProbe = _clusterLabelsToProbe + (numProbes * iBatch);
  uint32_t* chunkIndexPtr              = _chunkIndexPtr + (numProbes * iBatch);

  //
  uint32_t j_end = (numProbes + 1024 - 1) / 1024;
  for (uint32_t j = 0; j < j_end; j++) {
    uint32_t i   = threadIdx.x + (1024 * j);
    uint32_t val = 0;
    if (i < numProbes) {
      uint32_t l = clusterLabelsToProbe[i];
      val        = cluster_offsets[l + 1] - cluster_offsets[l];
    }
    val = thread_block_scan(val, smem_temp);

    if (i < numProbes) {
      if (j > 0) { val += smem_base[(j - 1) & 0x1]; }
      chunkIndexPtr[i] = val;
      if (i == numProbes - 1) { numSamples[iBatch] = val; }
    }

    if ((j < j_end - 1) && (threadIdx.x == 1023)) { smem_base[j & 0x1] = val; }
  }
}

__device__ inline void ivfpq_get_id_dataset(uint32_t iSample,
                                            uint32_t numProbes,
                                            const uint32_t* cluster_offsets,  // [n_clusters + 1,]
                                            const uint32_t* cluster_labels,   // [numProbes,]
                                            const uint32_t* chunkIndexPtr,    // [numProbes,]
                                            uint32_t& iChunk,
                                            uint32_t& label,
                                            uint32_t& iDataset)
{
  uint32_t minChunk = 0;
  uint32_t maxChunk = numProbes - 1;
  iChunk            = (minChunk + maxChunk) / 2;
  while (minChunk < maxChunk) {
    if (iSample >= chunkIndexPtr[iChunk]) {
      minChunk = iChunk + 1;
    } else {
      maxChunk = iChunk;
    }
    iChunk = (minChunk + maxChunk) / 2;
  }

  label                   = cluster_labels[iChunk];
  uint32_t iSampleInChunk = iSample;
  if (iChunk > 0) { iSampleInChunk -= chunkIndexPtr[iChunk - 1]; }
  iDataset = iSampleInChunk + cluster_offsets[label];
}

template <typename scoreDtype>
__global__ void ivfpq_make_outputs(uint32_t numProbes,
                                   uint32_t topk,
                                   uint32_t maxSamples,
                                   uint32_t sizeBatch,
                                   const uint32_t* cluster_offsets,  // [n_clusters + 1]
                                   const uint32_t* data_indices,     // [index_size]
                                   const uint32_t* cluster_labels,   // [sizeBatch, numProbes]
                                   const uint32_t* chunkIndexPtr,    // [sizeBatch, numProbes]
                                   const scoreDtype* scores,         // [sizeBatch, maxSamples] or
                                                                     // [sizeBatch, numProbes, topk]
                                   const uint32_t* scoreTopkIndex,   // [sizeBatch, numProbes, topk]
                                   const uint32_t* topkSampleIds,    // [sizeBatch, topk]
                                   uint64_t* topkNeighbors,          // [sizeBatch, topk]
                                   float* topkScores                 // [sizeBatch, topk]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= topk) return;
  uint32_t iBatch = blockIdx.y;
  if (iBatch >= sizeBatch) return;

  uint32_t iSample = topkSampleIds[i + (topk * iBatch)];
  if (scoreTopkIndex == nullptr) {
    // 0 <= iSample < maxSamples
    topkScores[i + (topk * iBatch)] = scores[iSample + (maxSamples * iBatch)];
    uint32_t iChunk;
    uint32_t label;
    uint32_t iDataset;
    ivfpq_get_id_dataset(iSample,
                         numProbes,
                         cluster_offsets,
                         cluster_labels + (numProbes * iBatch),
                         chunkIndexPtr + (numProbes * iBatch),
                         iChunk,
                         label,
                         iDataset);
    topkNeighbors[i + (topk * iBatch)] = data_indices[iDataset];
  } else {
    // 0 <= iSample < (numProbes * topk)
    topkScores[i + (topk * iBatch)]    = scores[iSample + ((numProbes * topk) * iBatch)];
    uint32_t iDataset                  = scoreTopkIndex[iSample + ((numProbes * topk) * iBatch)];
    topkNeighbors[i + (topk * iBatch)] = data_indices[iDataset];
  }
}

template <int pq_bits, int vecLen, typename T, typename smemLutDtype = float>
__device__ float ivfpq_compute_score(uint32_t pq_dim,
                                     uint32_t iDataset,
                                     const uint8_t* pqDataset,  // [n_rows, pq_dim * pq_bits / 8]
                                     const smemLutDtype* preCompScores  // [pq_dim, pq_width]
)
{
  float score             = 0.0;
  constexpr uint32_t bitT = sizeof(T) * 8;
  const T* headPqDataset  = (T*)(pqDataset + (uint64_t)iDataset * (pq_dim * pq_bits / 8));
  for (int j = 0; j < pq_dim / vecLen; j += 1) {
    T pqCode = headPqDataset[0];
    headPqDataset += 1;
    uint32_t bitLeft = bitT;
#pragma unroll vecLen
    for (int k = 0; k < vecLen; k += 1) {
      uint8_t code = pqCode;
      if (bitLeft > pq_bits) {
        // This condition is always true here (to make the compiler happy)
        if constexpr (bitT > pq_bits) { pqCode >>= pq_bits; }
        bitLeft -= pq_bits;
      } else {
        if (k < vecLen - 1) {
          pqCode = headPqDataset[0];
          headPqDataset += 1;
        }
        code |= (pqCode << bitLeft);
        pqCode >>= (pq_bits - bitLeft);
        bitLeft += (bitT - pq_bits);
      }
      code &= (1 << pq_bits) - 1;
      score += (float)preCompScores[code];
      preCompScores += (1 << pq_bits);
    }
  }
  return score;
}

extern __shared__ float smemArray[];

//
// (*) Restrict the peak GPU occupancy up-to 50% by "__launch_bounds__(1024, 1)",
// as there were cases where performance dropped by a factor of two or more on V100
// when the peak GPU occupancy was set to more than 50%.
//
template <int pq_bits,
          int vecLen,
          typename T,
          int depth,
          bool preCompBaseDiff,
          typename outDtype,
          typename smemLutDtype>
__launch_bounds__(1024, 1) __global__ void ivfpq_compute_similarity(
  uint32_t n_rows,
  uint32_t data_dim,
  uint32_t numProbes,
  uint32_t pq_dim,
  uint32_t sizeBatch,
  uint32_t maxSamples,
  distance::DistanceType metric,
  codebook_gen codebook_kind,
  uint32_t topk,
  const float* cluster_centers,     // [n_clusters, data_dim,]
  const float* pqCenters,           // [pq_dim, pq_width, pq_len,], or
                                    // [numClusetrs, pq_width, pq_len,]
  const uint8_t* pqDataset,         // [n_rows, pq_dim * pq_bits / 8]
  const uint32_t* cluster_offsets,  // [n_clusters + 1,]
  const uint32_t* _clusterLabels,   // [sizeBatch, numProbes,]
  const uint32_t* _chunkIndexPtr,   // [sizeBatch, numProbes,]
  const float* _query,              // [sizeBatch, data_dim,]
  const uint32_t* indexList,        // [sizeBatch * numProbes]
  float* _preCompScores,            // [...]
  outDtype* _output,                // [sizeBatch, maxSamples,] or [sizeBatch, numProbes, topk]
  uint32_t* _topkIndex              // [sizeBatch, numProbes, topk]
)
{
  const uint32_t pq_len = data_dim / pq_dim;

  smemLutDtype* preCompScores = (smemLutDtype*)smemArray;
  float* baseDiff             = nullptr;
  if (preCompBaseDiff) { baseDiff = (float*)(preCompScores + (pq_dim << pq_bits)); }
  bool manageLocalTopk = false;
  if (_topkIndex != nullptr) { manageLocalTopk = true; }

  uint32_t iBatch;
  uint32_t iProbe;
  if (indexList == nullptr) {
    // iBatch = blockIdx.x / numProbes;
    // iProbe = blockIdx.x % numProbes;
    iBatch = blockIdx.x % sizeBatch;
    iProbe = blockIdx.x / sizeBatch;
  } else {
    iBatch = indexList[blockIdx.x] / numProbes;
    iProbe = indexList[blockIdx.x] % numProbes;
  }
  if (iBatch >= sizeBatch || iProbe >= numProbes) return;

  const uint32_t* cluster_labels = _clusterLabels + (numProbes * iBatch);
  const uint32_t* chunkIndexPtr  = _chunkIndexPtr + (numProbes * iBatch);
  const float* query             = _query + (data_dim * iBatch);
  outDtype* output;
  uint32_t* topkIndex = nullptr;
  if (manageLocalTopk) {
    // Store topk calculated distances to output (and its indices to topkIndex)
    output    = _output + (topk * (iProbe + (numProbes * iBatch)));
    topkIndex = _topkIndex + (topk * (iProbe + (numProbes * iBatch)));
  } else {
    // Store all calculated distances to output
    output = _output + (maxSamples * iBatch);
  }
  uint32_t label               = cluster_labels[iProbe];
  const float* myClusterCenter = cluster_centers + (data_dim * label);
  const float* myPqCenters;
  if (codebook_kind == codebook_gen::PER_SUBSPACE) {
    myPqCenters = pqCenters;
  } else {
    myPqCenters = pqCenters + (pq_len << pq_bits) * label;
  }

  if (preCompBaseDiff) {
    // Reduce computational complexity by pre-computing the difference
    // between the cluster centroid and the query.
    for (uint32_t i = threadIdx.x; i < data_dim; i += blockDim.x) {
      baseDiff[i] = query[i] - myClusterCenter[i];
    }
    __syncthreads();
  }

  // Create a lookup table
  for (uint32_t i = threadIdx.x; i < (pq_dim << pq_bits); i += blockDim.x) {
    uint32_t iPq   = i >> pq_bits;
    uint32_t iCode = i & ((1 << pq_bits) - 1);
    float score    = 0.0;
    for (uint32_t j = 0; j < pq_len; j++) {
      uint32_t k = j + (pq_len * iPq);
      float diff;
      if (preCompBaseDiff) {
        diff = baseDiff[k];
      } else {
        diff = query[k] - myClusterCenter[k];
      }
      if (codebook_kind == codebook_gen::PER_SUBSPACE) {
        diff -= myPqCenters[j + (pq_len * i)];
      } else {
        diff -= myPqCenters[j + (pq_len * iCode)];
      }
      score += diff * diff;
    }
    preCompScores[i] = (smemLutDtype)score;
  }

  uint32_t iSampleBase = 0;
  if (iProbe > 0) { iSampleBase = chunkIndexPtr[iProbe - 1]; }
  uint32_t nSamples                = chunkIndexPtr[iProbe] - iSampleBase;
  uint32_t nSamples32              = Pow2<32>::roundUp(nSamples);
  uint32_t selected_cluster_offset = cluster_offsets[label];

  using block_sort_t =
    topk::block_sort<topk::warp_sort_immediate, depth * WarpSize, true, outDtype, uint32_t>;
  block_sort_t block_topk(topk, reinterpret_cast<uint8_t*>(smemArray));
  const outDtype limit = block_sort_t::queue_t::kDummy;

  // Compute a distance for each sample
  for (uint32_t i = threadIdx.x; i < nSamples32; i += blockDim.x) {
    float score = limit;
    if (i < nSamples) {
      score = ivfpq_compute_score<pq_bits, vecLen, T, smemLutDtype>(
        pq_dim, selected_cluster_offset + i, pqDataset, preCompScores);
    }
    if (!manageLocalTopk) {
      if (i < nSamples) { output[i + iSampleBase] = score; }
    } else {
      block_topk.add(score, selected_cluster_offset + i);
    }
  }
  if (!manageLocalTopk) { return; }
  // sync threads before the topk merging operation, because we reuse the shared memory
  __syncthreads();
  block_topk.done();
  block_topk.store(output, topkIndex);
}

//
template <int pq_bits, int vecLen, typename T, int depth, bool preCompBaseDiff, typename outDtype>
__launch_bounds__(1024, 1) __global__ void ivfpq_compute_similarity_no_smem_lut(
  uint32_t n_rows,
  uint32_t data_dim,
  uint32_t numProbes,
  uint32_t pq_dim,
  uint32_t sizeBatch,
  uint32_t maxSamples,
  distance::DistanceType metric,
  codebook_gen codebook_kind,
  uint32_t topk,
  const float* cluster_centers,     // [n_clusters, data_dim,]
  const float* pqCenters,           // [pq_dim, pq_width, pq_len,], or
                                    // [numClusetrs, pq_width, pq_len,]
  const uint8_t* pqDataset,         // [n_rows, pq_dim * pq_bits / 8]
  const uint32_t* cluster_offsets,  // [n_clusters + 1,]
  const uint32_t* _clusterLabels,   // [sizeBatch, numProbes,]
  const uint32_t* _chunkIndexPtr,   // [sizeBatch, numProbes,]
  const float* _query,              // [sizeBatch, data_dim,]
  const uint32_t* indexList,        // [sizeBatch * numProbes]
  float* _preCompScores,            // [..., pq_dim << pq_bits,]
  outDtype* _output,                // [sizeBatch, maxSamples,] or [sizeBatch, numProbes, topk]
  uint32_t* _topkIndex              // [sizeBatch, numProbes, topk]
)
{
  const uint32_t pq_len = data_dim / pq_dim;

  float* preCompScores = _preCompScores + ((pq_dim << pq_bits) * blockIdx.x);
  float* baseDiff      = nullptr;
  if (preCompBaseDiff) { baseDiff = (float*)smemArray; }
  bool manageLocalTopk = false;
  if (_topkIndex != nullptr) { manageLocalTopk = true; }

  for (int ib = blockIdx.x; ib < sizeBatch * numProbes; ib += gridDim.x) {
    uint32_t iBatch;
    uint32_t iProbe;
    if (indexList == nullptr) {
      // iBatch = ib / numProbes;
      // iProbe = ib % numProbes;
      iBatch = ib % sizeBatch;
      iProbe = ib / sizeBatch;
    } else {
      iBatch = indexList[ib] / numProbes;
      iProbe = indexList[ib] % numProbes;
    }

    const uint32_t* cluster_labels = _clusterLabels + (numProbes * iBatch);
    const uint32_t* chunkIndexPtr  = _chunkIndexPtr + (numProbes * iBatch);
    const float* query             = _query + (data_dim * iBatch);
    outDtype* output;
    uint32_t* topkIndex = nullptr;
    if (manageLocalTopk) {
      // Store topk calculated distances to output (and its indices to topkIndex)
      output    = _output + (topk * (iProbe + (numProbes * iBatch)));
      topkIndex = _topkIndex + (topk * (iProbe + (numProbes * iBatch)));
    } else {
      // Store all calculated distances to output
      output = _output + (maxSamples * iBatch);
    }
    uint32_t label               = cluster_labels[iProbe];
    const float* myClusterCenter = cluster_centers + (data_dim * label);
    const float* myPqCenters;
    if (codebook_kind == codebook_gen::PER_SUBSPACE) {
      myPqCenters = pqCenters;
    } else {
      myPqCenters = pqCenters + (pq_len << pq_bits) * label;
    }

    if (preCompBaseDiff) {
      // Reduce computational complexity by pre-computing the difference
      // between the cluster centroid and the query.
      for (uint32_t i = threadIdx.x; i < data_dim; i += blockDim.x) {
        baseDiff[i] = query[i] - myClusterCenter[i];
      }
      __syncthreads();
    }

    // Create a lookup table
    for (uint32_t i = threadIdx.x; i < (pq_dim << pq_bits); i += blockDim.x) {
      uint32_t iPq   = i >> pq_bits;
      uint32_t iCode = i & ((1 << pq_bits) - 1);
      float score    = 0.0;
      for (uint32_t j = 0; j < pq_len; j++) {
        uint32_t k = j + (pq_len * iPq);
        float diff;
        if (preCompBaseDiff) {
          diff = baseDiff[k];
        } else {
          diff = query[k] - myClusterCenter[k];
        }
        if (codebook_kind == codebook_gen::PER_SUBSPACE) {
          diff -= myPqCenters[j + (pq_len * i)];
        } else {
          diff -= myPqCenters[j + (pq_len * iCode)];
        }
        score += diff * diff;
      }
      preCompScores[i] = score;
    }

    uint32_t iSampleBase = 0;
    if (iProbe > 0) { iSampleBase = chunkIndexPtr[iProbe - 1]; }
    uint32_t nSamples                = chunkIndexPtr[iProbe] - iSampleBase;
    uint32_t nSamples32              = Pow2<32>::roundUp(nSamples);
    uint32_t selected_cluster_offset = cluster_offsets[label];

    using block_sort_t =
      topk::block_sort<topk::warp_sort_immediate, depth * WarpSize, true, outDtype, uint32_t>;
    block_sort_t block_topk(topk, reinterpret_cast<uint8_t*>(smemArray));
    const outDtype limit = block_sort_t::queue_t::kDummy;

    // Compute a distance for each sample
    for (uint32_t i = threadIdx.x; i < nSamples32; i += blockDim.x) {
      float score = limit;
      if (i < nSamples) {
        score = ivfpq_compute_score<pq_bits, vecLen, T>(
          pq_dim, selected_cluster_offset + i, pqDataset, preCompScores);
      }
      if (!manageLocalTopk) {
        if (i < nSamples) { output[i + iSampleBase] = score; }
      } else {
        block_topk.add(score, selected_cluster_offset + i);
      }
    }
    __syncthreads();
    if (!manageLocalTopk) {
      continue;  // for (int ib ...)
    }
    block_topk.done();
    block_topk.store(output, topkIndex);
    __syncthreads();
  }
}

// search
template <typename scoreDtype, typename smemLutDtype, typename IdxT>
void ivfpq_search(const handle_t& handle,
                  const index<IdxT>& index,
                  uint32_t n_probes,
                  uint32_t max_batch_size,
                  uint32_t topK,
                  uint32_t preferred_thread_block_size,
                  uint32_t n_queries,
                  const float* cluster_centers,          // [index_size, rot_dim]
                  const float* pqCenters,                // [pq_dim, pq_width, pq_len]
                  const uint8_t* pqDataset,              // [index_size, pq_dim * pq_bits / 8]
                  const uint32_t* data_indices,          // [index_size]
                  const uint32_t* cluster_offsets,       // [n_clusters + 1]
                  const uint32_t* clusterLabelsToProbe,  // [n_queries, numProbes]
                  const float* query,                    // [n_queries, rot_dim]
                  uint64_t* topkNeighbors,               // [n_queries, topK]
                  float* topkDistances,                  // [n_queries, topK]
                  rmm::mr::device_memory_resource* mr)
{
  RAFT_EXPECTS(n_queries <= max_batch_size,
               "number of queries (%u) must be smaller the max batch size (%u)",
               n_queries,
               max_batch_size);
  auto stream = handle.get_stream();

  auto max_samples = Pow2<128>::roundUp(index.inclusiveSumSortedClusterSize()(n_probes - 1));

  bool manage_local_topk =
    raft::ceildiv<int>(topK, 32) <= 4    // depth is not too large
    && n_probes >= 16                    // not too few clusters looked up
    && max_batch_size * n_probes >= 256  // overall amount of work is not too small
    ;

  rmm::device_uvector<uint32_t> cluster_labels_out(max_batch_size * n_probes, stream, mr);
  rmm::device_uvector<uint32_t> index_list_sorted_buf(0, stream, mr);
  uint32_t* index_list_sorted = nullptr;
  rmm::device_uvector<uint32_t> num_samples(max_batch_size, stream, mr);
  rmm::device_uvector<uint32_t> chunk_index(max_batch_size * n_probes, stream, mr);
  rmm::device_uvector<uint32_t> topk_sids(max_batch_size * topK, stream, mr);
  // [maxBatchSize, maxSamples] or  [maxBatchSize, numProbes, topk]
  rmm::device_uvector<scoreDtype> scores_buf(0, stream, mr);
  rmm::device_uvector<uint32_t> topk_index_buf(0, stream, mr);
  uint32_t* topk_index = nullptr;
  if (manage_local_topk) {
    scores_buf.resize(max_batch_size * n_probes * topK, stream);
    topk_index_buf.resize(max_batch_size * n_probes * topK, stream);
    topk_index = topk_index_buf.data();
  } else {
    scores_buf.resize(max_batch_size * max_samples, stream);
  }

  dim3 mc_threads(1024, 1, 1);  // DO NOT CHANGE
  dim3 mc_blocks(n_queries, 1, 1);
  ivfpq_make_chunk_index_ptr<<<mc_blocks, mc_threads, 0, stream>>>(n_probes,
                                                                   n_queries,
                                                                   cluster_offsets,
                                                                   clusterLabelsToProbe,
                                                                   chunk_index.data(),
                                                                   num_samples.data());

  if (n_queries * n_probes > 256) {
    // Sorting index by cluster number (label).
    // The goal is to incrase the L2 cache hit rate to read the vectors
    // of a cluster by processing the cluster at the same time as much as
    // possible.
    index_list_sorted_buf.resize(max_batch_size * n_probes, stream);
    rmm::device_uvector<uint32_t> index_list_buf(max_batch_size * n_probes, stream, mr);
    auto index_list   = index_list_buf.data();
    index_list_sorted = index_list_sorted_buf.data();
    thrust::sequence(handle.get_thrust_policy(),
                     thrust::device_pointer_cast(index_list),
                     thrust::device_pointer_cast(index_list + n_queries * n_probes));

    int begin_bit             = 0;
    int end_bit               = sizeof(uint32_t) * 8;
    size_t cub_workspace_size = 0;
    cub::DeviceRadixSort::SortPairs(nullptr,
                                    cub_workspace_size,
                                    clusterLabelsToProbe,
                                    cluster_labels_out.data(),
                                    index_list,
                                    index_list_sorted,
                                    n_queries * n_probes,
                                    begin_bit,
                                    end_bit,
                                    stream);
    rmm::device_buffer cub_workspace(cub_workspace_size, stream, mr);
    cub::DeviceRadixSort::SortPairs(cub_workspace.data(),
                                    cub_workspace_size,
                                    clusterLabelsToProbe,
                                    cluster_labels_out.data(),
                                    index_list,
                                    index_list_sorted,
                                    n_queries * n_probes,
                                    begin_bit,
                                    end_bit,
                                    stream);
  }

  // Select a GPU kernel for distance calculation
#define SET_KERNEL1(B, V, T, D)                                                                 \
  do {                                                                                          \
    static_assert((B * V) % (sizeof(T) * 8) == 0);                                              \
    kernel_no_basediff = ivfpq_compute_similarity<B, V, T, D, false, scoreDtype, smemLutDtype>; \
    kernel_fast        = ivfpq_compute_similarity<B, V, T, D, true, scoreDtype, smemLutDtype>;  \
    kernel_no_smem_lut = ivfpq_compute_similarity_no_smem_lut<B, V, T, D, true, scoreDtype>;    \
  } while (0)

#define SET_KERNEL2(B, M, D)                                                     \
  do {                                                                           \
    RAFT_EXPECTS(index.pq_dim() % M == 0, "pq_dim must be a multiple of %u", M); \
    if (index.pq_dim() % (M * 8) == 0) {                                         \
      SET_KERNEL1(B, (M * 8), uint64_t, D);                                      \
    } else if (index.pq_dim() % (M * 4) == 0) {                                  \
      SET_KERNEL1(B, (M * 4), uint32_t, D);                                      \
    } else if (index.pq_dim() % (M * 2) == 0) {                                  \
      SET_KERNEL1(B, (M * 2), uint16_t, D);                                      \
    } else if (index.pq_dim() % (M * 1) == 0) {                                  \
      SET_KERNEL1(B, (M * 1), uint8_t, D);                                       \
    }                                                                            \
  } while (0)

#define SET_KERNEL3(D)                     \
  do {                                     \
    switch (index.pq_bits()) {             \
      case 4: SET_KERNEL2(4, 2, D); break; \
      case 5: SET_KERNEL2(5, 8, D); break; \
      case 6: SET_KERNEL2(6, 4, D); break; \
      case 7: SET_KERNEL2(7, 8, D); break; \
      case 8: SET_KERNEL2(8, 1, D); break; \
    }                                      \
  } while (0)

  typedef void (*kernel_t)(uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           uint32_t,
                           distance::DistanceType,
                           codebook_gen,
                           uint32_t,
                           const float*,
                           const float*,
                           const uint8_t*,
                           const uint32_t*,
                           const uint32_t*,
                           const uint32_t*,
                           const float*,
                           const uint32_t*,
                           float*,
                           scoreDtype*,
                           uint32_t*);
  kernel_t kernel_no_basediff;
  kernel_t kernel_fast;
  kernel_t kernel_no_smem_lut;
  uint32_t depth = 1;
  if (manage_local_topk) {
    while (depth * WarpSize < topK) {
      depth *= 2;
    }
  }
  switch (depth) {
    case 1: SET_KERNEL3(1); break;
    case 2: SET_KERNEL3(2); break;
    case 4: SET_KERNEL3(4); break;
    default: RAFT_FAIL("ivf_pq::search(k = %u): depth value is too big (%d)", topK, depth);
  }
  RAFT_LOG_DEBUG("ivf_pq::search(k = %u, depth = %d, dim = %u/%u/%u)",
                 topK,
                 depth,
                 index.dim(),
                 index.rot_dim(),
                 index.pq_dim());
  constexpr size_t thresholdSmem = 48 * 1024;
  size_t sizeSmem                = sizeof(smemLutDtype) * index.pq_dim() * index.pq_width();
  size_t sizeSmemBaseDiff        = sizeof(float) * index.rot_dim();

  uint32_t numCTAs = n_queries * n_probes;
  int numThreads   = 1024;
  // preferred_thread_block_size == 0 means using auto thread block size calculation
  // mode
  if (preferred_thread_block_size == 0) {
    constexpr int minThreads = 256;
    while (numThreads > minThreads) {
      if (numCTAs < uint32_t(getMultiProcessorCount() * (1024 / (numThreads / 2)))) { break; }
      if (handle.get_device_properties().sharedMemPerMultiprocessor * 2 / 3 <
          sizeSmem * (1024 / (numThreads / 2))) {
        break;
      }
      numThreads /= 2;
    }
  } else {
    numThreads = preferred_thread_block_size;
  }
  size_t sizeSmemForLocalTopk =
    topk::template calc_smem_size_for_block_wide<float, uint32_t>(numThreads / WarpSize, topK);
  sizeSmem = max(sizeSmem, sizeSmemForLocalTopk);

  kernel_t kernel = kernel_no_basediff;

  bool kernel_no_basediff_available = true;
  if (sizeSmem > thresholdSmem) {
    cudaError_t cudaError = cudaFuncSetAttribute(
      kernel_no_basediff, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeSmem);
    if (cudaError != cudaSuccess) {
      RAFT_EXPECTS(
        cudaError == cudaGetLastError(),
        "Tried to reset the expected cuda error code, but it didn't match the expectation");
      kernel_no_basediff_available = false;

      // Use "kernel_no_smem_lut" which just uses small amount of shared memory.
      kernel     = kernel_no_smem_lut;
      numThreads = 1024;
      size_t sizeSmemForLocalTopk =
        topk::calc_smem_size_for_block_wide<float, uint32_t>(numThreads / WarpSize, topK);
      sizeSmem = max(sizeSmemBaseDiff, sizeSmemForLocalTopk);
      numCTAs  = getMultiProcessorCount();
    }
  }
  if (kernel_no_basediff_available) {
    bool kernel_fast_available = true;
    if (sizeSmem + sizeSmemBaseDiff > thresholdSmem) {
      cudaError_t cudaError = cudaFuncSetAttribute(
        kernel_fast, cudaFuncAttributeMaxDynamicSharedMemorySize, sizeSmem + sizeSmemBaseDiff);
      if (cudaError != cudaSuccess) {
        RAFT_EXPECTS(
          cudaError == cudaGetLastError(),
          "Tried to reset the expected cuda error code, but it didn't match the expectation");
        kernel_fast_available = false;
      }
    }
    if (kernel_fast_available) {
      int numBlocks_kernel_no_basediff = 0;
      RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks_kernel_no_basediff, kernel_no_basediff, numThreads, sizeSmem));

      int numBlocks_kernel_fast = 0;
      RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &numBlocks_kernel_fast, kernel_fast, numThreads, sizeSmem + sizeSmemBaseDiff));

      // Use "kernel_fast" only if GPU occupancy does not drop
      if (numBlocks_kernel_no_basediff == numBlocks_kernel_fast) {
        kernel = kernel_fast;
        sizeSmem += sizeSmemBaseDiff;
      }
    }
  }

  rmm::device_uvector<float> precomp_scores(
    numCTAs * index.pq_dim() * index.pq_width(), stream, mr);
  dim3 cta_threads(numThreads, 1, 1);
  dim3 cta_blocks(numCTAs, 1, 1);
  kernel<<<cta_blocks, cta_threads, sizeSmem, stream>>>(index.size(),
                                                        index.rot_dim(),
                                                        n_probes,
                                                        index.pq_dim(),
                                                        n_queries,
                                                        max_samples,
                                                        index.metric(),
                                                        index.codebook_kind(),
                                                        topK,
                                                        cluster_centers,
                                                        pqCenters,
                                                        pqDataset,
                                                        cluster_offsets,
                                                        clusterLabelsToProbe,
                                                        chunk_index.data(),
                                                        query,
                                                        index_list_sorted,
                                                        precomp_scores.data(),
                                                        scores_buf.data(),
                                                        topk_index);

  // Select topk vectors for each query
  if (topk_index == nullptr) {
    _cuann_find_topk(handle,
                     topK,
                     n_queries,
                     max_samples,
                     num_samples.data(),
                     scores_buf.data(),
                     topk_sids.data(),
                     mr);
  } else {
    _cuann_find_topk(
      handle, topK, n_queries, (n_probes * topK), nullptr, scores_buf.data(), topk_sids.data(), mr);
  }

  dim3 mo_threads(128, 1, 1);
  dim3 mo_blocks(raft::ceildiv<uint32_t>(topK, mo_threads.x), n_queries, 1);
  ivfpq_make_outputs<scoreDtype><<<mo_blocks, mo_threads, 0, stream>>>(n_probes,
                                                                       topK,
                                                                       max_samples,
                                                                       n_queries,
                                                                       cluster_offsets,
                                                                       data_indices,
                                                                       clusterLabelsToProbe,
                                                                       chunk_index.data(),
                                                                       scores_buf.data(),
                                                                       topk_index,
                                                                       topk_sids.data(),
                                                                       topkNeighbors,
                                                                       topkDistances);
}

/** See raft::spatial::knn::ivf_pq::search docs */
template <typename T, typename IdxT>
inline void search(const handle_t& handle,
                   const search_params& params,
                   const index<IdxT>& index,
                   const T* queries,
                   uint32_t n_queries,
                   uint32_t k,
                   IdxT* neighbors,
                   float* distances,
                   rmm::mr::device_memory_resource* mr = nullptr)
{
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "Unsupported element type.");
  static_assert(std::is_same_v<IdxT, uint64_t>,
                "Only uint64_t index output is supported at this time.");
  common::nvtx::range<common::nvtx::domain::raft> fun_scope(
    "ivf_pq::search(k = %u, n_queries = %u, dim = %zu)", k, n_queries, index.dim());

  RAFT_EXPECTS(
    params.internal_distance_dtype == CUDA_R_16F || params.internal_distance_dtype == CUDA_R_32F,
    "internal_distance_dtype must be either CUDA_R_16F or CUDA_R_32F");
  RAFT_EXPECTS(params.smem_lut_dtype == CUDA_R_16F || params.smem_lut_dtype == CUDA_R_32F ||
                 params.smem_lut_dtype == CUDA_R_8U,
               "smem_lut_dtype must be CUDA_R_16F, CUDA_R_32F or CUDA_R_8U");
  RAFT_EXPECTS(
    params.preferred_thread_block_size == 256 || params.preferred_thread_block_size == 512 ||
      params.preferred_thread_block_size == 1024 || params.preferred_thread_block_size == 0,
    "preferred_thread_block_size must be 0, 256, 512 or 1024, but %u is given.",
    params.preferred_thread_block_size);
  RAFT_EXPECTS(k > 0, "parameter `k` in top-k must be positive.");
  RAFT_EXPECTS(
    k <= index.size(),
    "parameter `k` (%u) in top-k must not be larger that the total size of the index (%u)",
    k,
    index.size());
  RAFT_EXPECTS(params.n_probes > 0,
               "n_probes (number of clusters to probe in the search) must be positive.");
  auto n_probes = std::min<uint32_t>(params.n_probes, index.n_lists());
  {
    uint32_t n_samples_worst_case = index.size();
    if (n_probes < index.n_lists()) {
      n_samples_worst_case = index.size() - index.inclusiveSumSortedClusterSize()(
                                              std::max<uint32_t>(index.numClustersSize0(),
                                                                 index.n_lists() - 1 - n_probes) -
                                              index.numClustersSize0());
    }
    if (k > n_samples_worst_case) {
      RAFT_LOG_WARN(
        "n_probes is too small to get top-k results reliably (n_probes: %u, k: %u, "
        "n_samples_worst_case: %u).",
        n_probes,
        k,
        n_samples_worst_case);
    }
  }

  auto pool_guard = raft::get_pool_memory_resource(mr, n_queries * n_probes * k * 16);
  if (pool_guard) {
    RAFT_LOG_DEBUG("ivf_pq::search: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  // Maximum number of query vectors to search at the same time.
  uint32_t batch_size = std::min<uint32_t>(n_queries, 32768);
  auto max_queries    = min(max(batch_size, 1), 4096);
  auto max_batch_size = max_queries;
  {
    // TODO: copied from {legacy}; figure this out.
    // Adjust max_batch_size to improve GPU occupancy of topk kernel.
    uint32_t numCta_total    = getMultiProcessorCount() * 2;
    uint32_t numCta_perBatch = numCta_total / max_batch_size;
    float utilization        = (float)numCta_perBatch * max_batch_size / numCta_total;
    if (numCta_perBatch > 1 || (numCta_perBatch == 1 && utilization < 0.6)) {
      uint32_t numCta_perBatch_1 = numCta_perBatch + 1;
      uint32_t maxBatchSize_1    = numCta_total / numCta_perBatch_1;
      float utilization_1        = (float)numCta_perBatch_1 * maxBatchSize_1 / numCta_total;
      if (utilization < utilization_1) { max_batch_size = maxBatchSize_1; }
    }
  }

  auto stream = handle.get_stream();

  auto cluster_centers   = index.centers().data_handle();
  auto pqCenters         = index.pq_centers().data_handle();
  auto pqDataset         = index.pq_dataset().data_handle();
  auto data_indices      = index.indices().data_handle();
  auto cluster_offsets   = index.list_offsets().data_handle();
  auto rotationMatrix    = index.rotation_matrix().data_handle();
  auto clusterRotCenters = index.centers_rot().data_handle();

  //
  rmm::device_uvector<T> dev_queries(max_queries * index.dim_ext(), stream, mr);
  rmm::device_uvector<float> cur_queries(max_queries * index.dim_ext(), stream, mr);
  rmm::device_uvector<float> rot_queries(max_queries * index.rot_dim(), stream, mr);
  rmm::device_uvector<uint32_t> clusters_to_probe(max_queries * params.n_probes, stream, mr);
  rmm::device_uvector<float> qc_distances(max_queries * index.n_lists(), stream, mr);

  void (*_ivfpq_search)(const handle_t&,
                        const ivf_pq::index<IdxT>&,
                        uint32_t,
                        uint32_t,
                        uint32_t,
                        uint32_t,
                        uint32_t,
                        const float*,
                        const float*,
                        const uint8_t*,
                        const uint32_t*,
                        const uint32_t*,
                        const uint32_t*,
                        const float*,
                        uint64_t*,
                        float*,
                        rmm::mr::device_memory_resource*);
  if (params.internal_distance_dtype == CUDA_R_16F) {
    if (params.smem_lut_dtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<half, half>;
    } else if (params.smem_lut_dtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<half, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<half, float>;
    }
  } else {
    if (params.smem_lut_dtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<float, half>;
    } else if (params.smem_lut_dtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<float, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<float, float>;
    }
  }

  switch (utils::check_pointer_residency(queries, neighbors, distances)) {
    case utils::pointer_residency::device_only:
    case utils::pointer_residency::host_and_device: break;
    default: RAFT_FAIL("all pointers must be accessible from the device.");
  }

  for (uint32_t i = 0; i < n_queries; i += max_queries) {
    uint32_t nQueries = min(max_queries, n_queries - i);

    float fillValue = 0.0;
    if (index.metric() != raft::distance::DistanceType::InnerProduct) { fillValue = 1.0 / -2.0; }
    utils::copy_fill(nQueries,
                     index.dim(),
                     queries + static_cast<size_t>(index.dim()) * i,
                     index.dim(),
                     cur_queries.data(),
                     index.dim_ext(),
                     fillValue,
                     stream);

    float alpha;
    float beta;
    uint32_t gemmK = index.dim();
    if (index.metric() == distance::DistanceType::InnerProduct) {
      alpha = -1.0;
      beta  = 0.0;
    } else {
      alpha = -2.0;
      beta  = 0.0;
      gemmK = index.dim() + 1;
      RAFT_EXPECTS(gemmK <= index.dim_ext(), "unexpected gemmK or dim_ext");
    }
    linalg::gemm(handle,
                 true,
                 false,
                 index.n_lists(),
                 nQueries,
                 gemmK,
                 &alpha,
                 cluster_centers,
                 index.dim_ext(),
                 cur_queries.data(),
                 index.dim_ext(),
                 &beta,
                 qc_distances.data(),
                 index.n_lists(),
                 stream);

    // Rotate queries
    alpha = 1.0;
    beta  = 0.0;
    linalg::gemm(handle,
                 true,
                 false,
                 index.rot_dim(),
                 nQueries,
                 index.dim(),
                 &alpha,
                 rotationMatrix,
                 index.dim(),
                 cur_queries.data(),
                 index.dim_ext(),
                 &beta,
                 rot_queries.data(),
                 index.rot_dim(),
                 stream);

    // Select neighbor clusters for each query.
    _cuann_find_topk(handle,
                     params.n_probes,
                     nQueries,
                     index.n_lists(),
                     nullptr,
                     qc_distances.data(),
                     clusters_to_probe.data(),
                     mr,
                     false);

    for (uint32_t j = 0; j < nQueries; j += max_batch_size) {
      uint32_t batchSize = min(max_batch_size, nQueries - j);
      _ivfpq_search(handle,
                    index,
                    params.n_probes,
                    max_batch_size,
                    k,
                    params.preferred_thread_block_size,
                    batchSize,
                    clusterRotCenters,
                    pqCenters,
                    pqDataset,
                    data_indices,
                    cluster_offsets,
                    clusters_to_probe.data() + ((uint64_t)(params.n_probes) * j),
                    rot_queries.data() + ((uint64_t)(index.rot_dim()) * j),
                    neighbors + ((uint64_t)(k) * (i + j)),
                    distances + ((uint64_t)(k) * (i + j)),
                    mr);
    }
  }
}

}  // namespace raft::spatial::knn::ivf_pq::detail
