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
#include "ann_kmeans_balanced.cuh"
#include "ann_utils.cuh"
#include "topk/warpsort_topk.cuh"

#include <raft/core/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/device_atomics.cuh>
#include <raft/distance/distance_type.hpp>
#include <raft/linalg/gemm.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/pow2_utils.cuh>
#include <raft/stats/histogram.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/managed_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <thrust/fill.h>

///////////////////
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_fp16.h>
#include <omp.h>

//////////////////

namespace raft::spatial::knn::ivf_pq::detail {

using namespace raft::spatial::knn::detail;  // NOLINT

/**
 *
 *
 *
 *
 *
 * fp_8bit
 */

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

/**
 *
 * end of fp8bit
 *
 */

using namespace cub;

//
extern __shared__ float smemArray[];

#define FP16_MAX 65504.0

// header of index
struct cuannIvfPqIndexHeader {
  // (*) DO NOT CHANGE ORDER
  size_t indexSize;
  uint32_t version;
  uint32_t numClusters;
  uint32_t numDataset;
  uint32_t data_dim;
  uint32_t pq_dim;
  uint32_t metric;
  uint32_t maxClusterSize;
  uint32_t rot_dim;
  uint32_t bitPq;
  uint32_t typePqCenter;
  uint32_t dtypeDataset;
  uint32_t dimDatasetExt;
  uint32_t numDatasetAdded;
  uint32_t _dummy[256 - 15];
};

//
inline char* _cuann_get_dtype_string(cudaDataType_t dtype, char* string)
{
  if (dtype == CUDA_R_32F)
    sprintf(string, "float (CUDA_R_32F)");
  else if (dtype == CUDA_R_16F)
    sprintf(string, "half (CUDA_R_16F)");
  else if (dtype == CUDA_R_8U)
    sprintf(string, "uint8 (CUDA_R_8U)");
  else if (dtype == CUDA_R_8I)
    sprintf(string, "int8 (CUDA_R_8I)");
  else
    sprintf(string, "unknown");
  return string;
}

// copy_fill
template <typename S, typename D>
__global__ void kern_copy_fill(uint32_t nRows,
                               uint32_t nCols,
                               const S* src,  // [nRows, ldSrc]
                               uint32_t ldSrc,
                               D* dst,  // [nRows, ldDst]
                               uint32_t ldDst,
                               D fillValue,
                               D divisor)
{
  uint32_t gid  = threadIdx.x + (blockDim.x * blockIdx.x);
  uint32_t iCol = gid % ldDst;
  uint32_t iRow = gid / ldDst;
  if (iRow >= nRows) return;
  if (iCol < nCols) {
    dst[iCol + (ldDst * iRow)] = src[iCol + (ldSrc * iRow)] / divisor;
  } else {
    dst[iCol + (ldDst * iRow)] = fillValue;
  }
}

// copy_fill
template <typename S, typename D>
inline void _cuann_copy_fill(uint32_t nRows,
                             uint32_t nCols,
                             const S* src,  // [nRows, ldSrc]
                             uint32_t ldSrc,
                             D* dst,  // [nRows, ldDst]
                             uint32_t ldDst,
                             D fillValue,
                             D divisor,
                             cudaStream_t stream)
{
  RAFT_EXPECTS(ldSrc >= nCols, "src leading dimension must be larger than nCols");
  RAFT_EXPECTS(ldDst >= nCols, "dist leading dimension must be larger than nCols");
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((nRows * ldDst) + nThreads - 1) / nThreads;
  kern_copy_fill<S, D>
    <<<nBlocks, nThreads, 0, stream>>>(nRows, nCols, src, ldSrc, dst, ldDst, fillValue, divisor);
}

//
template <typename D, typename S>
__global__ void kern_transpose_copy_3d(uint32_t num0,
                                       uint32_t num1,
                                       uint32_t num2,
                                       D* dst,  // [num2, ld1, ld0]
                                       uint32_t ld0,
                                       uint32_t ld1,
                                       const S* src,  // [...]
                                       uint32_t stride0,
                                       uint32_t stride1,
                                       uint32_t stride2)
{
  uint32_t tid = threadIdx.x + (blockDim.x * blockIdx.x);
  if (tid >= num0 * num1 * num2) return;
  uint32_t i0 = tid % num0;
  uint32_t i1 = (tid / num0) % num1;
  uint32_t i2 = (tid / num0) / num1;

  dst[i0 + (ld0 * i1) + (ld0 * ld1 * i2)] = src[(stride0 * i0) + (stride1 * i1) + (stride2 * i2)];
}

// transpose_copy_3d
template <typename D, typename S>
inline void _cuann_transpose_copy_3d(uint32_t num0,
                                     uint32_t num1,
                                     uint32_t num2,
                                     D* dst,  // [num2, ld1, ld0]
                                     uint32_t ld0,
                                     uint32_t ld1,
                                     const S* src,  // [...]
                                     uint32_t stride0,
                                     uint32_t stride1,
                                     uint32_t stride2,
                                     rmm::cuda_stream_view stream)
{
  uint32_t nThreads = 128;
  uint32_t nBlocks  = ((num0 * num1 * num2) + nThreads - 1) / nThreads;
  kern_transpose_copy_3d<D, S><<<nBlocks, nThreads, 0, stream>>>(
    num0, num1, num2, dst, ld0, ld1, src, stride0, stride1, stride2);
}

//
#define NUM_THREADS      1024  // DO NOT CHANGE
#define STATE_BIT_LENGTH 8     // 0: state not used,  8: state used
#define MAX_VEC_LENGTH   8     // 1, 2, 4 or 8

//
__device__ inline uint32_t convert(uint32_t x)
{
  if (x & 0x80000000) {
    return x ^ 0xffffffff;
  } else {
    return x ^ 0x80000000;
  }
}

//
struct u32_vector {
  uint1 x1;
  uint2 x2;
  uint4 x4;
  ulonglong4 x8;
};

//
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

//
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

//
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
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
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
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
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

//
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
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
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
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
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

//
__device__ inline uint16_t convert(uint16_t x)
{
  if (x & 0x8000) {
    return x ^ 0xffff;
  } else {
    return x ^ 0x8000;
  }
}

//
struct u16_vector {
  ushort1 x1;
  ushort2 x2;
  ushort4 x4;
  uint4 x8;
};

//
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

//
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

//
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
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
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
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
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
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
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
    uint32_t numSample_perThread = (max_len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
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

//
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

//
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
    uint32_t numSample_perThread = (maxSamples + numThreads_perBatch - 1) / numThreads_perBatch;
    uint32_t numState_perThread  = (numSample_perThread + stateBitLen - 1) / stateBitLen;
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

//
int _get_vecLen(uint32_t maxSamples, int maxVecLen = MAX_VEC_LENGTH)
{
  int vecLen = min(maxVecLen, MAX_VEC_LENGTH);
  while ((maxSamples % vecLen) != 0) {
    vecLen /= 2;
  }
  return vecLen;
}

//
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
  int numBlocks_perBatch = (maxSamples + (numThreads * vecLen) - 1) / (numThreads * vecLen);
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

  dim3 stpThreads(128, 1, 1);
  dim3 stpBlocks((max(sizeBatch + 1, sizeBatch * topK) + stpThreads.x - 1) / stpThreads.x, 1, 1);
  _sort_topk_prep<<<stpBlocks, stpThreads, 0, handle.get_stream()>>>(
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

//
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
  int numBlocks_perBatch = (maxSamples + (numThreads * vecLen) - 1) / (numThreads * vecLen);
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

// search
template <typename scoreDtype, typename smemLutDtype>
inline void ivfpq_search(const handle_t& handle,
                         cuannIvfPqDescriptor_t& desc,
                         uint32_t numQueries,
                         const float* cluster_centers,          // [numDataset, data_dim]
                         const float* pqCenters,                // [pq_dim, 256, lenPq]
                         const uint8_t* pqDataset,              // [numDataset, pq_dim]
                         const uint32_t* data_indices,          // [numDataset]
                         const uint32_t* cluster_offsets,       // [numClusters + 1]
                         const uint32_t* clusterLabelsToProbe,  // [numQueries, numProbes]
                         const float* query,                    // [data_dim]
                         uint64_t* topKNeighbors,               // [topK]
                         float* topKDistances,                  // [topK]
                         rmm::mr::device_memory_resource* mr);

//
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

//
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

//
__global__ void ivfpq_make_chunk_index_ptr(
  uint32_t numProbes,
  uint32_t sizeBatch,
  const uint32_t* cluster_offsets,        // [numClusters + 1,]
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

//
__device__ inline void ivfpq_get_id_dataset(uint32_t iSample,
                                            uint32_t numProbes,
                                            const uint32_t* clusterIndexPtr,  // [numClusters + 1,]
                                            const uint32_t* clusterLabels,    // [numProbes,]
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

  label                   = clusterLabels[iChunk];
  uint32_t iSampleInChunk = iSample;
  if (iChunk > 0) { iSampleInChunk -= chunkIndexPtr[iChunk - 1]; }
  iDataset = iSampleInChunk + clusterIndexPtr[label];
}

//
template <typename scoreDtype>
__global__ void ivfpq_make_outputs(uint32_t numProbes,
                                   uint32_t topk,
                                   uint32_t maxSamples,
                                   uint32_t sizeBatch,
                                   const uint32_t* clusterIndexPtr,  // [numClusters + 1]
                                   const uint32_t* data_indices,     // [numDataset]
                                   const uint32_t* clusterLabels,    // [sizeBatch, numProbes]
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
                         clusterIndexPtr,
                         clusterLabels + (numProbes * iBatch),
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

//
inline bool manage_local_topk(cuannIvfPqDescriptor_t& desc)
{
  int depth = raft::ceildiv<int>(desc->topK, 32);
  if (depth > 4) { return false; }
  if (desc->numProbes < 16) { return false; }
  if (desc->maxBatchSize * desc->numProbes < 256) { return false; }
  return true;
}

// return workspace size
inline size_t ivfpq_search_bufferSize(const handle_t& handle, cuannIvfPqDescriptor_t& desc)
{
  size_t size = 0;
  // clusterLabelsOut  [maxBatchSize, numProbes]
  size += Pow2<128>::roundUp(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // indexList  [maxBatchSize * numProbes]
  size += Pow2<128>::roundUp(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // indexListSorted  [maxBatchSize * numProbes]
  size += Pow2<128>::roundUp(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // numSamples  [maxBatchSize,]
  size += Pow2<128>::roundUp(sizeof(uint32_t) * desc->maxBatchSize);
  // cubWorkspace
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  uint32_t* d_keys_in       = nullptr;
  uint32_t* d_keys_out      = nullptr;
  uint32_t* d_values_in     = nullptr;
  uint32_t* d_values_out    = nullptr;
  cub::DeviceRadixSort::SortPairs(d_temp_storage,
                                  temp_storage_bytes,
                                  d_keys_in,
                                  d_keys_out,
                                  d_values_in,
                                  d_values_out,
                                  desc->maxBatchSize * desc->numProbes);
  desc->sizeCubWorkspace = Pow2<128>::roundUp(temp_storage_bytes);
  size += desc->sizeCubWorkspace;
  // chunkIndexPtr  [maxBatchSize, numProbes]
  size += Pow2<128>::roundUp(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes);
  // topkSids  [maxBatchSize, topk]
  size += Pow2<128>::roundUp(sizeof(uint32_t) * desc->maxBatchSize * desc->topK);
  // similarity
  size_t unit_size = sizeof(float);
  if (desc->internalDistanceDtype == CUDA_R_16F) { unit_size = sizeof(half); }
  if (manage_local_topk(desc)) {
    // [matBatchSize, numProbes, topK]
    size += Pow2<128>::roundUp(unit_size * desc->maxBatchSize * desc->numProbes * desc->topK);
  } else {
    // [matBatchSize, maxSamples]
    size += Pow2<128>::roundUp(unit_size * desc->maxBatchSize * desc->maxSamples);
  }
  // simTopkIndex
  if (manage_local_topk(desc)) {
    // [matBatchSize, numProbes, topk]
    size +=
      Pow2<128>::roundUp(sizeof(uint32_t) * desc->maxBatchSize * desc->numProbes * desc->topK);
  }
  // preCompScores  [multiProcessorCount, pq_dim, 1 << bitPq,]
  size += Pow2<128>::roundUp(sizeof(float) * getMultiProcessorCount() * desc->pq_dim *
                             (1 << desc->bitPq));
  // topkWorkspace
  if (manage_local_topk(desc)) {
    size += _cuann_find_topk_bufferSize(handle,
                                        desc->topK,
                                        desc->maxBatchSize,
                                        desc->numProbes * desc->topK,
                                        desc->internalDistanceDtype);
  } else {
    size += _cuann_find_topk_bufferSize(
      handle, desc->topK, desc->maxBatchSize, desc->maxSamples, desc->internalDistanceDtype);
  }
  return size;
}

//
__device__ __host__ inline void ivfpq_encode_core(
  uint32_t ldDataset, uint32_t pq_dim, uint32_t bitPq, const uint32_t* label, uint8_t* output)
{
  for (uint32_t j = 0; j < pq_dim; j++) {
    uint8_t code = label[(ldDataset * j)];
    if (bitPq == 8) {
      uint8_t* ptrOutput = output + j;
      ptrOutput[0]       = code;
    } else if (bitPq == 7) {
      uint8_t* ptrOutput = output + 7 * (j / 8);
      if (j % 8 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 8 == 1) {
        ptrOutput[0] |= code << 7;
        ptrOutput[1] |= code >> 1;
      } else if (j % 8 == 2) {
        ptrOutput[1] |= code << 6;
        ptrOutput[2] |= code >> 2;
      } else if (j % 8 == 3) {
        ptrOutput[2] |= code << 5;
        ptrOutput[3] |= code >> 3;
      } else if (j % 8 == 4) {
        ptrOutput[3] |= code << 4;
        ptrOutput[4] |= code >> 4;
      } else if (j % 8 == 5) {
        ptrOutput[4] |= code << 3;
        ptrOutput[5] |= code >> 5;
      } else if (j % 8 == 6) {
        ptrOutput[5] |= code << 2;
        ptrOutput[6] |= code >> 6;
      } else if (j % 8 == 7) {
        ptrOutput[6] |= code << 1;
      }
    } else if (bitPq == 6) {
      uint8_t* ptrOutput = output + 3 * (j / 4);
      if (j % 4 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 4 == 1) {
        ptrOutput[0] |= code << 6;
        ptrOutput[1] |= code >> 2;
      } else if (j % 4 == 2) {
        ptrOutput[1] |= code << 4;
        ptrOutput[2] |= code >> 4;
      } else if (j % 4 == 3) {
        ptrOutput[2] |= code << 2;
      }
    } else if (bitPq == 5) {
      uint8_t* ptrOutput = output + 5 * (j / 8);
      if (j % 8 == 0) {
        ptrOutput[0] |= code;
      } else if (j % 8 == 1) {
        ptrOutput[0] |= code << 5;
        ptrOutput[1] |= code >> 3;
      } else if (j % 8 == 2) {
        ptrOutput[1] |= code << 2;
      } else if (j % 8 == 3) {
        ptrOutput[1] |= code << 7;
        ptrOutput[2] |= code >> 1;
      } else if (j % 8 == 4) {
        ptrOutput[2] |= code << 4;
        ptrOutput[3] |= code >> 4;
      } else if (j % 8 == 5) {
        ptrOutput[3] |= code << 1;
      } else if (j % 8 == 6) {
        ptrOutput[3] |= code << 6;
        ptrOutput[4] |= code >> 2;
      } else if (j % 8 == 7) {
        ptrOutput[4] |= code << 3;
      }
    } else if (bitPq == 4) {
      uint8_t* ptrOutput = output + (j / 2);
      if (j % 2 == 0) {
        ptrOutput[0] |= code;
      } else {
        ptrOutput[0] |= code << 4;
      }
    }
  }
}

//
__global__ void ivfpq_encode_kernel(uint32_t numDataset,
                                    uint32_t ldDataset,  // (*) ldDataset >= numDataset
                                    uint32_t pq_dim,
                                    uint32_t bitPq,         // 4 <= bitPq <= 8
                                    const uint32_t* label,  // [pq_dim, ldDataset]
                                    uint8_t* output         // [numDataset, pq_dim]
)
{
  uint32_t i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i >= numDataset) return;
  ivfpq_encode_core(ldDataset, pq_dim, bitPq, label + i, output + (pq_dim * bitPq / 8) * i);
}

//
inline void ivfpq_encode(uint32_t numDataset,
                         uint32_t ldDataset,  // (*) ldDataset >= numDataset
                         uint32_t pq_dim,
                         uint32_t bitPq,         // 4 <= bitPq <= 8
                         const uint32_t* label,  // [pq_dim, ldDataset]
                         uint8_t* output         // [numDataset, pq_dim]
)
{
#if 1
  // GPU
  dim3 iekThreads(128, 1, 1);
  dim3 iekBlocks((numDataset + iekThreads.x - 1) / iekThreads.x, 1, 1);
  ivfpq_encode_kernel<<<iekBlocks, iekThreads>>>(
    numDataset, ldDataset, pq_dim, bitPq, label, output);
#else
  // CPU
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  for (uint32_t i = 0; i < numDataset; i++) {
    ivfpq_encode_core(ldDataset, pq_dim, bitPq, label + i, output + (pq_dim * bitPq / 8) * i);
  }
#endif
}

inline void cuannIvfPqGetIndexSize(cuannIvfPqDescriptor_t& desc,
                                   size_t* size /* bytes of data_vectors index */);

inline size_t _cuann_getIndexSize_cluster_centers(cuannIvfPqDescriptor_t& desc)
{
  // [numClusters, dimDatasetExt]
  return Pow2<128>::roundUp(sizeof(float) * desc->numClusters * desc->dimDatasetExt);
}

inline size_t _cuann_getIndexSize_pqCenters(cuannIvfPqDescriptor_t& desc)
{
  size_t size_base = sizeof(float) * (1 << desc->bitPq) * desc->lenPq;
  if (desc->typePqCenter == codebook_gen::PER_SUBSPACE) {
    // [pq_dim, 1 << bitPq, lenPq]
    return Pow2<128>::roundUp(desc->pq_dim * size_base);
  } else {
    // [numClusters, 1 << bitPq, lenPq]
    return Pow2<128>::roundUp(desc->numClusters * size_base);
  }
}

inline size_t _cuann_getIndexSize_pqDataset(cuannIvfPqDescriptor_t& desc)
{
  // [numDataset, pq_dim * bitPq / 8]
  return Pow2<128>::roundUp(sizeof(uint8_t) * desc->numDataset * desc->pq_dim * desc->bitPq / 8);
}

inline size_t _cuann_getIndexSize_originalNumbers(cuannIvfPqDescriptor_t& desc)
{
  // [numDataset,]
  return Pow2<128>::roundUp(sizeof(uint32_t) * desc->numDataset);
}

inline size_t _cuann_getIndexSize_indexPtr(cuannIvfPqDescriptor_t& desc)
{
  // [numClusters + 1,]
  return Pow2<128>::roundUp(sizeof(uint32_t) * (desc->numClusters + 1));
}

inline size_t _cuann_getIndexSize_rotationMatrix(cuannIvfPqDescriptor_t& desc)
{
  // [data_dim, rot_dim]
  return Pow2<128>::roundUp(sizeof(float) * desc->data_dim * desc->rot_dim);
}

inline size_t _cuann_getIndexSize_clusterRotCenters(cuannIvfPqDescriptor_t& desc)
{
  // [numClusters, rot_dim]
  return Pow2<128>::roundUp(sizeof(float) * desc->numClusters * desc->rot_dim);
}

inline void _cuann_get_index_pointers(cuannIvfPqDescriptor_t& desc,
                                      struct cuannIvfPqIndexHeader** header,
                                      float** cluster_centers,  // [numClusters, dimDatasetExt]
                                      float** pqCenters,        // [pq_dim, 1 << bitPq, lenPq], or
                                                                // [numClusters, 1 << bitPq, lenPq]
                                      uint8_t** pqDataset,      // [numDataset, pq_dim * bitPq / 8]
                                      uint32_t** data_indices,  // [numDataset]
                                      uint32_t** cluster_offsets,  // [numClusters + 1]
                                      float** rotationMatrix,      // [data_dim, rot_dim]
                                      float** clusterRotCenters    // [numClusters, rot_dim]
)
{
  *header          = (struct cuannIvfPqIndexHeader*)(desc->index_ptr);
  *cluster_centers = (float*)((uint8_t*)(*header) + sizeof(struct cuannIvfPqIndexHeader));
  *pqCenters = (float*)((uint8_t*)(*cluster_centers) + _cuann_getIndexSize_cluster_centers(desc));
  *pqDataset = (uint8_t*)((uint8_t*)(*pqCenters) + _cuann_getIndexSize_pqCenters(desc));
  *data_indices = (uint32_t*)((uint8_t*)(*pqDataset) + _cuann_getIndexSize_pqDataset(desc));
  *cluster_offsets =
    (uint32_t*)((uint8_t*)(*data_indices) + _cuann_getIndexSize_originalNumbers(desc));
  *rotationMatrix = (float*)((uint8_t*)(*cluster_offsets) + _cuann_getIndexSize_indexPtr(desc));
  *clusterRotCenters =
    (float*)((uint8_t*)(*rotationMatrix) + _cuann_getIndexSize_rotationMatrix(desc));
}

template <typename T>
int descending(const void* a, const void* b)
{
  T valA = ((T*)a)[0];
  T valB = ((T*)b)[0];
  if (valA > valB) return -1;
  if (valA < valB) return 1;
  return 0;
}

// (*) This is temporal. Need to be removed in future.
inline void _cuann_get_random_norm_vector(int len, float* vector)
{
  float sqsum = 0.0;
  for (int i = 0; i < len; i++) {
    vector[i] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    sqsum += vector[i] * vector[i];
  }
  float norm = sqrt(sqsum);
  for (int i = 0; i < len; i++) {
    vector[i] /= norm;
  }
}

inline void _cuann_get_inclusiveSumSortedClusterSize(
  cuannIvfPqDescriptor_t& desc,
  const uint32_t* cluster_offsets,  // [numClusters + 1]
  float* cluster_centers,           // [numClusters, dimDatasetExt]
  uint32_t** output                 // [numClusters]
)
{
  // [CPU]
  if (*output != nullptr) { free(*output); }
  *output                 = (uint32_t*)malloc(sizeof(uint32_t) * desc->numClusters);
  desc->_numClustersSize0 = 0;
  for (uint32_t i = 0; i < desc->numClusters; i++) {
    (*output)[i] = cluster_offsets[i + 1] - cluster_offsets[i];
    if ((*output)[i] > 0) continue;

    desc->_numClustersSize0 += 1;
    // Work-around for clusters of size 0
    _cuann_get_random_norm_vector(desc->dimDatasetExt, cluster_centers + (desc->dimDatasetExt * i));
  }
  RAFT_LOG_DEBUG("Number of clusters of size zero: %d", desc->_numClustersSize0);
  // sort
  qsort(*output, desc->numClusters, sizeof(uint32_t), descending<uint32_t>);
  // scan
  for (uint32_t i = 1; i < desc->numClusters; i++) {
    (*output)[i] += (*output)[i - 1];
  }
  RAFT_EXPECTS((*output)[desc->numClusters - 1] == desc->numDataset, "cluster sizes do not add up");
}

inline void _cuann_get_sqsumClusters(cuannIvfPqDescriptor_t& desc,
                                     const float* cluster_centers,  // [numClusters, data_dim,]
                                     float** output                 // [numClusters,]
)
{
  if (*output != nullptr) { RAFT_CUDA_TRY(cudaFree(*output)); }
  RAFT_CUDA_TRY(cudaMallocManaged(output, sizeof(float) * desc->numClusters));
  switch (utils::check_pointer_residency(cluster_centers, *output)) {
    case utils::pointer_residency::device_only:
    case utils::pointer_residency::host_and_device: break;
    default: RAFT_FAIL("_cuann_get_sqsumClusters: not all pointers are available on the device.");
  }
  rmm::cuda_stream_default.synchronize();
  utils::dots_along_rows(
    desc->numClusters, desc->data_dim, cluster_centers, *output, rmm::cuda_stream_default);
  rmm::cuda_stream_default.synchronize();
}

//
template <typename T>
T _cuann_dot(int n, const T* x, int incX, const T* y, int incY)
{
  T val = 0;
  for (int i = 0; i < n; i++) {
    val += x[incX * i] * y[incY * i];
  }
  return val;
}

//
template <typename T, typename X, typename Y>
T _cuann_dot(int n, const X* x, int incX, const Y* y, int incY, T divisor = 1)
{
  T val = 0;
  for (int i = 0; i < n; i++) {
    val += (T)(x[incX * i]) * (T)(y[incY * i]) / divisor;
  }
  return val;
}

//
template <typename T>
T _cuann_rand()
{
  return (T)rand() / RAND_MAX;
}

// make rotation matrix
inline void _cuann_make_rotation_matrix(uint32_t nRows,
                                        uint32_t nCols,
                                        uint32_t lenPq,
                                        bool randomRotation,
                                        float* rotationMatrix  // [nRows, nCols]
)
{
  RAFT_EXPECTS(
    nRows >= nCols, "number of rows (%u) must be larger than number or cols (%u)", nRows, nCols);
  RAFT_EXPECTS(
    nRows % lenPq == 0, "number of rows (%u) must be a multiple of lenPq (%u)", nRows, lenPq);

  if (randomRotation) {
    RAFT_LOG_DEBUG("Creating a random rotation matrix.");
    double dot, norm;
    std::vector<double> matrix(nRows * nCols, 0.0);
    for (uint32_t i = 0; i < nRows * nCols; i++) {
      matrix[i] = _cuann_rand<double>() - 0.5;
    }
    for (uint32_t j = 0; j < nCols; j++) {
      // normalize the j-th col vector
      norm = sqrt(_cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[j], nCols));
      for (uint32_t i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
      // orthogonalize the j-th col vector with the previous col vectors
      for (uint32_t k = 0; k < j; k++) {
        dot = _cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[k], nCols);
        for (uint32_t i = 0; i < nRows; i++) {
          matrix[j + (nCols * i)] -= dot * matrix[k + (nCols * i)];
        }
      }
      // normalize the j-th col vector again
      norm = sqrt(_cuann_dot<double>(nRows, &matrix[j], nCols, &matrix[j], nCols));
      for (uint32_t i = 0; i < nRows; i++) {
        matrix[j + (nCols * i)] /= norm;
      }
    }
    for (uint32_t i = 0; i < nRows * nCols; i++) {
      rotationMatrix[i] = (float)matrix[i];
    }
  } else {
    if (nRows == nCols) {
      memset(rotationMatrix, 0, sizeof(float) * nRows * nCols);
      for (uint32_t i = 0; i < nCols; i++) {
        rotationMatrix[i + (nCols * i)] = 1.0;
      }
    } else {
      memset(rotationMatrix, 0, sizeof(float) * nRows * nCols);
      uint32_t i = 0;
      for (uint32_t j = 0; j < nCols; j++) {
        rotationMatrix[j + (nCols * i)] = 1.0;
        i += lenPq;
        if (i >= nRows) { i = (i % nRows) + 1; }
      }
    }
  }
}

// show centers (for debuging)
inline void _cuann_kmeans_show_centers(const float* centers,  // [numCenters, dimCenters]
                                       uint32_t numCenters,
                                       uint32_t dimCenters,
                                       const uint32_t* centerSize,
                                       const uint32_t numShow = 5)
{
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  for (uint64_t k = 0; k < numCenters; k++) {
    if ((numShow <= k) && (k < numCenters - numShow)) {
      if (k == numShow) fprintf(stderr, "...\n");
      continue;
    }
    fprintf(stderr, "# centers[%lu]:", k);
    for (uint64_t j = 0; j < dimCenters; j++) {
      if ((numShow <= j) && (j < dimCenters - numShow)) {
        if (j == numShow) fprintf(stderr, " ... ");
        continue;
      }
      fprintf(stderr, " %f,", centers[j + (dimCenters * k)]);
    }
    fprintf(stderr, " %d\n", centerSize[k]);
  }
#endif
}

// show data_vectors (for debugging)
inline void _cuann_show_dataset(const float* data_vectors,  // [numDataset, data_dim]
                                uint32_t numDataset,
                                uint32_t data_dim,
                                const uint32_t numShow = 5)
{
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  for (uint64_t i = 0; i < numDataset; i++) {
    if ((numShow <= i) && (i < numDataset - numShow)) {
      if (i == numShow) fprintf(stderr, "...\n");
      continue;
    }
    fprintf(stderr, "# data_vectors[%lu]:", i);
    for (uint64_t j = 0; j < data_dim; j++) {
      if ((numShow <= j) && (j < data_dim - numShow)) {
        if (j == numShow) fprintf(stderr, " ... ");
        continue;
      }
      fprintf(stderr, " %.3f,", data_vectors[j + (data_dim * i)]);
    }
    fprintf(stderr, "\n");
  }
#endif
}

// show pq code (for debuging)
inline void _cuann_show_pq_code(const uint8_t* pqDataset,  // [numDataset, pq_dim]
                                uint32_t numDataset,
                                uint32_t pq_dim,
                                const uint32_t numShow = 5)
{
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  for (uint64_t i = 0; i < numDataset; i++) {
    if ((numShow <= i) && (i < numDataset - numShow)) {
      if (i == numShow) fprintf(stderr, "...\n");
      continue;
    }
    fprintf(stderr, "# data_vectors[%lu]:", i);
    for (uint64_t j = 0; j < pq_dim; j++) {
      if ((numShow <= j) && (j < pq_dim - numShow)) {
        if (j == numShow) fprintf(stderr, " ... ");
        continue;
      }
      fprintf(stderr, " %u,", pqDataset[j + (pq_dim * i)]);
    }
    fprintf(stderr, "\n");
  }
#endif
}

//
uint32_t _get_num_trainset(uint32_t cluster_size, uint32_t pq_dim, uint32_t bitPq)
{
  return min(cluster_size * pq_dim, 256 * max(1 << bitPq, pq_dim));
}

//
template <typename T>
void _cuann_compute_PQ_code(const handle_t& handle,
                            uint32_t numDataset,
                            uint32_t data_dim,
                            uint32_t rot_dim,
                            uint32_t pq_dim,
                            uint32_t lenPq,
                            uint32_t bitPq,
                            uint32_t numClusters,
                            codebook_gen typePqCenter,
                            uint32_t maxClusterSize,
                            float* cluster_centers,           // [numClusters, data_dim]
                            const float* rotationMatrix,      // [rot_dim, data_dim]
                            const T* data_vectors,            // [numDataset]
                            const uint32_t* data_indices,     // [numDataset]
                            const uint32_t* cluster_sizes,    // [numClusters]
                            const uint32_t* cluster_offsets,  // [numClusters + 1]
                            float* pqCenters,                 // [...]
                            uint32_t numIterations,
                            uint8_t* pqDataset,  // [numDataset, pq_dim * bitPq / 8]
                            rmm::mr::device_memory_resource* managed_memory,
                            rmm::mr::device_memory_resource* device_memory)
{
  auto stream = handle.get_stream();

  //
  // Compute PQ code
  //
  utils::memzero(pqDataset, numDataset * pq_dim * bitPq / 8, stream);

  rmm::device_uvector<float> res_vectors(maxClusterSize * data_dim, stream, managed_memory);
  rmm::device_uvector<float> rot_vectors(maxClusterSize * rot_dim, stream, managed_memory);
  rmm::device_uvector<float> sub_vectors(maxClusterSize * pq_dim * lenPq, stream, managed_memory);
  rmm::device_uvector<uint32_t> sub_vector_labels(maxClusterSize * pq_dim, stream, managed_memory);
  rmm::device_uvector<uint8_t> my_pq_dataset(
    maxClusterSize * pq_dim * bitPq / 8 /* NB: pq_dim * bitPQ % 8 == 0 */, stream, managed_memory);
  rmm::device_uvector<uint32_t> rot_vector_labels(0, stream, managed_memory);
  rmm::device_uvector<uint32_t> pq_cluster_size(0, stream, managed_memory);
  rmm::device_uvector<float> my_pq_centers(0, stream, managed_memory);

  if ((numIterations > 0) && (typePqCenter == codebook_gen::PER_CLUSTER)) {
    utils::memzero(pqCenters, numClusters * (1 << bitPq) * lenPq, stream);
    rot_vector_labels.resize(maxClusterSize * pq_dim, stream);
    pq_cluster_size.resize((1 << bitPq), stream);
    my_pq_centers.resize((1 << bitPq) * lenPq, stream);
  }

  for (uint32_t l = 0; l < numClusters; l++) {
    if (cluster_sizes[l] == 0) continue;

    //
    // Compute the residual vector of the new vector with its cluster
    // centroids.
    //   resVectors[..] = newVectors[..] - cluster_centers[..]
    //
    utils::copy_selected<float, T>(cluster_sizes[l],
                                   data_dim,
                                   data_vectors,
                                   data_indices + cluster_offsets[l],
                                   data_dim,
                                   res_vectors.data(),
                                   data_dim,
                                   stream);

    // substract centers from the vectors in the cluster.
    raft::matrix::linewiseOp(
      res_vectors.data(),
      res_vectors.data(),
      data_dim,
      cluster_sizes[l],
      true,
      [] __device__(float a, float b) { return a - b; },
      stream,
      cluster_centers + (uint64_t)l * data_dim);

    //
    // Rotate the residual vectors using a rotation matrix
    //
    float alpha = 1.0;
    float beta  = 0.0;
    linalg::gemm(handle,
                 true,
                 false,
                 rot_dim,
                 cluster_sizes[l],
                 data_dim,
                 &alpha,
                 rotationMatrix,
                 data_dim,
                 res_vectors.data(),
                 data_dim,
                 &beta,
                 rot_vectors.data(),
                 rot_dim,
                 stream);

    //
    // Training PQ codebook if codebook_gen::PER_CLUSTER
    // (*) PQ codebooks are trained for each cluster.
    //
    if ((numIterations > 0) && (typePqCenter == codebook_gen::PER_CLUSTER)) {
      uint32_t n_rows_train = _get_num_trainset(cluster_sizes[l], pq_dim, bitPq);
      kmeans::build_clusters(handle,
                             numIterations,
                             lenPq,
                             rot_vectors.data(),
                             n_rows_train,
                             (1 << bitPq),
                             my_pq_centers.data(),
                             rot_vector_labels.data(),
                             pq_cluster_size.data(),
                             raft::distance::DistanceType::L2Expanded,
                             device_memory,
                             stream);
      raft::copy(
        pqCenters + ((1 << bitPq) * lenPq) * l, my_pq_centers.data(), (1 << bitPq) * lenPq, stream);
    }

    //
    // Change the order of the vector data to facilitate processing in
    // each vector subspace.
    //   input:  rot_vectors[cluster_sizes, rot_dim]
    //   output: sub_vectors[pq_dim, cluster_sizes, lenPq]
    //
    _cuann_transpose_copy_3d<float, float>(lenPq,
                                           cluster_sizes[l],
                                           pq_dim,
                                           sub_vectors.data(),
                                           lenPq,
                                           cluster_sizes[l],
                                           rot_vectors.data(),
                                           1,
                                           rot_dim,
                                           lenPq,
                                           stream);

    //
    // Find a label (cluster ID) for each vector subspace.
    //
    for (uint32_t j = 0; j < pq_dim; j++) {
      float* curPqCenters = nullptr;
      if (typePqCenter == codebook_gen::PER_SUBSPACE) {
        curPqCenters = pqCenters + ((1 << bitPq) * lenPq) * j;
      } else if (typePqCenter == codebook_gen::PER_CLUSTER) {
        curPqCenters = pqCenters + ((1 << bitPq) * lenPq) * l;
        if (numIterations > 0) { curPqCenters = my_pq_centers.data(); }
      }
      kmeans::predict(handle,
                      curPqCenters,
                      (1 << bitPq),
                      lenPq,
                      sub_vectors.data() + j * (cluster_sizes[l] * lenPq),
                      cluster_sizes[l],
                      sub_vector_labels.data() + j * cluster_sizes[l],
                      raft::distance::DistanceType::L2Expanded,
                      stream,
                      device_memory);
    }
    handle.sync_stream();

    //
    // PQ encoding
    //
    ivfpq_encode(cluster_sizes[l],
                 cluster_sizes[l],
                 pq_dim,
                 bitPq,
                 sub_vector_labels.data(),
                 my_pq_dataset.data());
    RAFT_CUDA_TRY(cudaMemcpy(pqDataset + ((uint64_t)cluster_offsets[l] * pq_dim * bitPq / 8),
                             my_pq_dataset.data(),
                             sizeof(uint8_t) * cluster_sizes[l] * pq_dim * bitPq / 8,
                             cudaMemcpyDeviceToHost));
  }
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
}

inline void cuannIvfPqSetIndexParameters(cuannIvfPqDescriptor_t& desc,
                                         const uint32_t numClusters,
                                         const uint32_t numDataset,
                                         const uint32_t data_dim,
                                         const uint32_t pq_dim,
                                         const uint32_t bitPq,
                                         const distance::DistanceType metric,
                                         const codebook_gen typePqCenter)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");
  RAFT_EXPECTS(numClusters > 0, "(%s) numClusters must be larger than zero.", __func__);
  RAFT_EXPECTS(numDataset > 0, "(%s) numDataset must be larger than zero.", __func__);
  RAFT_EXPECTS(data_dim > 0, "(%s) data_dim must be larger than zero.", __func__);
  RAFT_EXPECTS(pq_dim > 0, "(%s) pq_dim must be larger than zero.", __func__);
  RAFT_EXPECTS(numClusters <= numDataset,
               "(%s) numClusters must be smaller than numDataset (numClusters:%u, numDataset:%u).",
               __func__,
               numClusters,
               numDataset);
  RAFT_EXPECTS(bitPq >= 4 && bitPq <= 8,
               "(%s) bitPq must be within closed range [4,8], but got %u.",
               __func__,
               bitPq);
  RAFT_EXPECTS((bitPq * pq_dim) % 8 == 0,
               "(%s) `bitPq * pq_dim` must be a multiple of 8, but got %u * %u = %u.",
               __func__,
               bitPq,
               pq_dim,
               bitPq * pq_dim);
  desc->numClusters   = numClusters;
  desc->numDataset    = numDataset;
  desc->data_dim      = data_dim;
  desc->dimDatasetExt = data_dim + 1;
  if (desc->dimDatasetExt % 8) { desc->dimDatasetExt += 8 - (desc->dimDatasetExt % 8); }
  RAFT_EXPECTS(desc->dimDatasetExt >= data_dim + 1, "unexpected dimDatasetExt");
  RAFT_EXPECTS(desc->dimDatasetExt % 8 == 0, "unexpected dimDatasetExt");
  desc->pq_dim       = pq_dim;
  desc->bitPq        = bitPq;
  desc->metric       = metric;
  desc->typePqCenter = typePqCenter;

  desc->rot_dim = data_dim;
  if (data_dim % pq_dim) { desc->rot_dim = ((data_dim / pq_dim) + 1) * pq_dim; }
  desc->lenPq = desc->rot_dim / pq_dim;
}

// cuannIvfPqGetIndexSize
inline void cuannIvfPqGetIndexSize(cuannIvfPqDescriptor_t& desc, size_t* size)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");

  *size = sizeof(struct cuannIvfPqIndexHeader);
  RAFT_EXPECTS(*size == 1024, "Critical error: unexpected header size.");
  *size += _cuann_getIndexSize_cluster_centers(desc);
  *size += _cuann_getIndexSize_pqCenters(desc);
  *size += _cuann_getIndexSize_pqDataset(desc);
  *size += _cuann_getIndexSize_originalNumbers(desc);
  *size += _cuann_getIndexSize_indexPtr(desc);
  *size += _cuann_getIndexSize_rotationMatrix(desc);
  *size += _cuann_getIndexSize_clusterRotCenters(desc);
}

template <typename T>
void cuannIvfPqBuildIndex(
  const handle_t& handle,
  cuannIvfPqDescriptor_t& desc,
  const T* data_vectors, /* [numDataset, data_dim] */
  double trainset_fraction,
  uint32_t numIterations, /* Number of iterations to train kmeans */
  bool randomRotation /* If true, rotate vectors with randamly created rotation matrix */)
{
  auto stream = handle.get_stream();
  cudaDataType_t dtype;
  if constexpr (std::is_same_v<T, float>) {
    dtype = CUDA_R_32F;
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    dtype = CUDA_R_8U;
  } else if constexpr (std::is_same_v<T, int8_t>) {
    dtype = CUDA_R_8I;
  } else {
    static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
      "unsupported type");
  }
  if (desc->metric == distance::DistanceType::InnerProduct) {
    RAFT_EXPECTS(dtype == CUDA_R_32F,
                 "Unsupported dtype (inner-product metric support float only)");
  }

  auto trainset_ratio = std::max<size_t>(
    1,
    desc->numDataset / std::max<size_t>(trainset_fraction * desc->numDataset, desc->numClusters));
  auto n_rows_train = desc->numDataset / trainset_ratio;

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG("cuannIvfPqBuildIndex: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  // TODO: move to device_memory, blocked by _cuann_dot
  rmm::device_uvector<T> trainset(n_rows_train * desc->data_dim, stream, &managed_memory);
  // TODO: a proper sampling
  RAFT_CUDA_TRY(cudaMemcpy2DAsync(trainset.data(),
                                  sizeof(T) * desc->data_dim,
                                  data_vectors,
                                  sizeof(T) * desc->data_dim * trainset_ratio,
                                  sizeof(T) * desc->data_dim,
                                  n_rows_train,
                                  cudaMemcpyDefault,
                                  stream));

  desc->dtypeDataset = dtype;
  char dtypeString[64];
  _cuann_get_dtype_string(desc->dtypeDataset, dtypeString);
  RAFT_LOG_DEBUG("Dataset dtype = %s", dtypeString);

  if (desc->index_ptr != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaFree(desc->index_ptr)); }
  size_t index_size;
  cuannIvfPqGetIndexSize(desc, &index_size);
  RAFT_CUDA_TRY(cudaMallocManaged(&(desc->index_ptr), index_size));

  struct cuannIvfPqIndexHeader* header;
  float* cluster_centers;     // [numClusters, data_dim]
  float* pqCenters;           // [pq_dim, 1 << bitPq, lenPq], or
                              // [numClusters, 1 << bitPq, lenPq]
  uint8_t* pqDataset;         // [numDataset, pq_dim * bitPq / 8]
  uint32_t* data_indices;     // [numDataset]
  uint32_t* cluster_offsets;  // [numClusters + 1]
  float* rotationMatrix;      // [data_dim, rot_dim]
  float* clusterRotCenters;   // [numClusters, rot_dim]
  _cuann_get_index_pointers(desc,
                            &header,
                            &cluster_centers,
                            &pqCenters,
                            &pqDataset,
                            &data_indices,
                            &cluster_offsets,
                            &rotationMatrix,
                            &clusterRotCenters);

  rmm::device_uvector<uint32_t> trainset_labels(n_rows_train, stream, &managed_memory);
  rmm::device_uvector<uint32_t> cluster_sizes(desc->numClusters, stream, &managed_memory);

  uint32_t numMesoClusters =
    std::min<uint32_t>(desc->numClusters, std::sqrt(desc->numClusters) + 0.5);
  RAFT_LOG_DEBUG("numMesoClusters: %u", numMesoClusters);

  rmm::device_uvector<float> mesocluster_centers(
    numMesoClusters * desc->data_dim, stream, &managed_memory);

  rmm::device_uvector<uint32_t> mesocluster_labels(n_rows_train, stream, &managed_memory);
  rmm::device_uvector<uint32_t> mesocluster_sizes(numMesoClusters, stream, &managed_memory);

  //
  // Training kmeans for meso-clusters
  //
  kmeans::build_clusters(handle,
                         numIterations,
                         desc->data_dim,
                         trainset.data(),
                         n_rows_train,
                         numMesoClusters,
                         mesocluster_centers.data(),
                         mesocluster_labels.data(),
                         mesocluster_sizes.data(),
                         desc->metric,
                         device_memory,
                         stream);
  handle.sync_stream();

  // Number of centers in each meso cluster
  // [numMesoClusters,]
  std::vector<uint32_t> fine_clusters_nums(numMesoClusters);
  std::vector<uint32_t> fine_clusters_csum(numMesoClusters + 1);
  fine_clusters_csum[0] = 0;

  uint32_t numClustersRemain  = desc->numClusters;
  uint32_t n_rows_trainRemain = n_rows_train;
  uint32_t mesoClusterSizeSum = 0;  // check
  uint32_t mesoClusterSizeMax = 0;
  uint32_t numFineClustersMax = 0;
  for (uint32_t i = 0; i < numMesoClusters; i++) {
    if (i < numMesoClusters - 1) {
      fine_clusters_nums[i] =
        (double)numClustersRemain * mesocluster_sizes.data()[i] / n_rows_trainRemain + .5;
    } else {
      fine_clusters_nums[i] = numClustersRemain;
    }
    fine_clusters_csum[i + 1] = fine_clusters_csum[i] + fine_clusters_nums[i];

    numClustersRemain -= fine_clusters_nums[i];
    n_rows_trainRemain -= mesocluster_sizes.data()[i];
    mesoClusterSizeSum += mesocluster_sizes.data()[i];
    mesoClusterSizeMax = max(mesoClusterSizeMax, mesocluster_sizes.data()[i]);
    numFineClustersMax = max(numFineClustersMax, fine_clusters_nums[i]);
  }
  RAFT_EXPECTS(mesoClusterSizeSum == n_rows_train, "mesocluster sizes do not add up");
  RAFT_EXPECTS(fine_clusters_csum[numMesoClusters] == desc->numClusters,
               "fine cluster sizes do not add up");

  rmm::device_uvector<uint32_t> ids_trainset(mesoClusterSizeMax, stream, &managed_memory);
  rmm::device_uvector<float> sub_trainset(
    mesoClusterSizeMax * desc->data_dim, stream, &managed_memory);
  rmm::device_uvector<uint32_t> labels_mp(mesoClusterSizeMax, stream, &managed_memory);
  rmm::device_uvector<float> cluster_centers_each(
    numFineClustersMax * desc->data_dim, stream, &managed_memory);
  rmm::device_uvector<uint32_t> cluser_size_mp(numFineClustersMax, stream, &managed_memory);

  //
  // Training kmeans for clusters in each meso-clusters
  //
  for (uint32_t i = 0; i < numMesoClusters; i++) {
    uint32_t k = 0;
    for (uint32_t j = 0; j < n_rows_train; j++) {
      if (mesocluster_labels.data()[j] != i) continue;
      ids_trainset.data()[k++] = j;
    }
    RAFT_EXPECTS(k == mesocluster_sizes.data()[i], "unexpected cluster size for cluster %u", i);

    utils::copy_selected<float, T>(mesocluster_sizes.data()[i],
                                   desc->data_dim,
                                   trainset.data(),
                                   ids_trainset.data(),
                                   desc->data_dim,
                                   sub_trainset.data(),
                                   desc->data_dim,
                                   stream);

    kmeans::build_clusters(handle,
                           numIterations,
                           desc->data_dim,
                           sub_trainset.data(),
                           mesocluster_sizes.data()[i],
                           fine_clusters_nums[i],
                           cluster_centers_each.data(),
                           labels_mp.data(),
                           cluser_size_mp.data(),
                           desc->metric,
                           device_memory,
                           stream);
    raft::copy(cluster_centers + (desc->data_dim * fine_clusters_csum[i]),
               cluster_centers_each.data(),
               fine_clusters_nums[i] * desc->data_dim,
               stream);
    handle.sync_stream();
  }

  //
  // Fine-tuning kmeans for whole clusters
  //
  // (*) Since the likely cluster centroids have been calculated
  // hierarchically already, the number of iteration for fine-tuning
  // kmeans for whole clusters should be reduced. However, there
  // is a possibility that the clusters could be unbalanced here,
  // in which case the actual number of iterations would be increased.
  //
  const int X         = 5;
  int numIterations_X = max(numIterations / 10, 2) * X;
  for (int iter = 0; iter < numIterations_X; iter += X) {
    kmeans::predict(handle,
                    cluster_centers,
                    desc->numClusters,
                    desc->data_dim,
                    trainset.data(),
                    n_rows_train,
                    trainset_labels.data(),
                    desc->metric,
                    stream,
                    device_memory);
    kmeans::calc_centers_and_sizes(cluster_centers,
                                   cluster_sizes.data(),
                                   desc->numClusters,
                                   desc->data_dim,
                                   trainset.data(),
                                   n_rows_train,
                                   trainset_labels.data(),
                                   true,
                                   stream);
    switch (desc->metric) {
      // For some metrics, cluster calculation and adjustment tends to favor zero center vectors.
      // To avoid converging to zero, we normalize the center vectors on every iteration.
      case raft::distance::DistanceType::InnerProduct:
      case raft::distance::DistanceType::CosineExpanded:
      case raft::distance::DistanceType::CorrelationExpanded:
        utils::normalize_rows(desc->numClusters, desc->data_dim, cluster_centers, stream);
      default: break;
    }
    handle.sync_stream();

    if ((iter + 1 < numIterations_X) && kmeans::adjust_centers(cluster_centers,
                                                               desc->numClusters,
                                                               desc->data_dim,
                                                               trainset.data(),
                                                               n_rows_train,
                                                               trainset_labels.data(),
                                                               cluster_sizes.data(),
                                                               (float)1.0 / 5,
                                                               device_memory,
                                                               stream)) {
      iter -= (X - 1);
      if (desc->metric == distance::DistanceType::InnerProduct) {
        utils::normalize_rows(desc->numClusters, desc->data_dim, cluster_centers, stream);
      }
    }
  }

  rmm::device_uvector<uint32_t> dataset_labels(desc->numDataset, stream, &managed_memory);

  //
  // Predict labels of whole data_vectors
  //
  kmeans::predict(handle,
                  cluster_centers,
                  desc->numClusters,
                  desc->data_dim,
                  data_vectors,
                  desc->numDataset,
                  dataset_labels.data(),
                  desc->metric,
                  stream,
                  device_memory);
  kmeans::calc_centers_and_sizes(cluster_centers,
                                 cluster_sizes.data(),
                                 desc->numClusters,
                                 desc->data_dim,
                                 data_vectors,
                                 desc->numDataset,
                                 dataset_labels.data(),
                                 true,
                                 stream);
  switch (desc->metric) {
    // For some metrics, cluster calculation and adjustment tends to favor zero center vectors.
    // To avoid converging to zero, we normalize the center vectors on every iteration.
    case raft::distance::DistanceType::InnerProduct:
    case raft::distance::DistanceType::CosineExpanded:
    case raft::distance::DistanceType::CorrelationExpanded:
      utils::normalize_rows(desc->numClusters, desc->data_dim, cluster_centers, stream);
    default: break;
  }
  handle.sync_stream();

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
  _cuann_kmeans_show_centers(
    cluster_centers, desc->numClusters, desc->data_dim, cluster_sizes.data());
#endif

  // Make rotation matrix
  RAFT_LOG_DEBUG("# data_dim: %u\n", desc->data_dim);
  RAFT_LOG_DEBUG("# rot_dim: %u\n", desc->rot_dim);
  RAFT_LOG_DEBUG("# randomRotation: %s\n", randomRotation ? "enabled" : "disabled");
  _cuann_make_rotation_matrix(
    desc->rot_dim, desc->data_dim, desc->lenPq, randomRotation, rotationMatrix);

  // Rotate cluster_centers
  float alpha = 1.0;
  float beta  = 0.0;
  linalg::gemm(handle,
               true,
               false,
               desc->rot_dim,
               desc->numClusters,
               desc->data_dim,
               &alpha,
               rotationMatrix,
               desc->data_dim,
               cluster_centers,
               desc->data_dim,
               &beta,
               clusterRotCenters,
               desc->rot_dim,
               stream);

  //
  // Make cluster_offsets, data_indices and pqDataset
  //
  uint32_t maxClusterSize = 0;
  // cluster_offsets
  cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < desc->numClusters; l++) {
    cluster_offsets[l + 1] = cluster_offsets[l] + cluster_sizes.data()[l];
    if (maxClusterSize < cluster_sizes.data()[l]) { maxClusterSize = cluster_sizes.data()[l]; }
  }
  RAFT_EXPECTS(cluster_offsets[desc->numClusters] == desc->numDataset,
               "Cluster sizes do not add up");
  desc->maxClusterSize = maxClusterSize;

  // data_indices
  for (uint32_t i = 0; i < desc->numDataset; i++) {
    uint32_t l                       = dataset_labels.data()[i];
    data_indices[cluster_offsets[l]] = i;
    cluster_offsets[l] += 1;
  }

  // Recover cluster_offsets
  for (uint32_t l = 0; l < desc->numClusters; l++) {
    cluster_offsets[l] -= cluster_sizes.data()[l];
  }

  rmm::device_uvector<uint32_t> pq_cluster_sizes((1 << desc->bitPq), stream, &managed_memory);

  if (desc->typePqCenter == codebook_gen::PER_SUBSPACE) {
    //
    // Training PQ codebook (codebook_gen::PER_SUBSPACE)
    // (*) PQ codebooks are trained for each subspace.
    //

    // Predict label of trainset again
    kmeans::predict(handle,
                    cluster_centers,
                    desc->numClusters,
                    desc->data_dim,
                    trainset.data(),
                    n_rows_train,
                    trainset_labels.data(),
                    desc->metric,
                    stream,
                    device_memory);
    handle.sync_stream();

    // [pq_dim, n_rows_train, lenPq]
    std::vector<float> mod_trainset(desc->pq_dim * n_rows_train * desc->lenPq, 0.0f);

    // mod_trainset[] = transpose( rotate(trainset[]) - clusterRotCenters[] )
#pragma omp parallel for
    for (uint32_t i = 0; i < n_rows_train; i++) {
      uint32_t l = trainset_labels.data()[i];
      for (uint32_t j = 0; j < desc->rot_dim; j++) {
        float val = FLT_MAX;
        if (dtype == CUDA_R_32F) {
          val = _cuann_dot<float, float, float>(
            desc->data_dim,
            (float*)trainset.data() + ((uint64_t)(desc->data_dim) * i),
            1,
            rotationMatrix + ((uint64_t)(desc->data_dim) * j),
            1);
        } else if (dtype == CUDA_R_8U) {
          float divisor = 256.0;
          val           = _cuann_dot<float, uint8_t, float>(
            desc->data_dim,
            (uint8_t*)trainset.data() + ((uint64_t)(desc->data_dim) * i),
            1,
            rotationMatrix + ((uint64_t)(desc->data_dim) * j),
            1,
            divisor);
        } else if (dtype == CUDA_R_8I) {
          float divisor = 128.0;
          val           = _cuann_dot<float, int8_t, float>(
            desc->data_dim,
            (int8_t*)trainset.data() + ((uint64_t)(desc->data_dim) * i),
            1,
            rotationMatrix + ((uint64_t)(desc->data_dim) * j),
            1,
            divisor);
        }
        uint32_t j0 = j / (desc->lenPq);  // 0 <= j0 < pq_dim
        uint32_t j1 = j % (desc->lenPq);  // 0 <= j1 < lenPq
        uint64_t idx =
          j1 + ((uint64_t)(desc->lenPq) * i) + ((uint64_t)(desc->lenPq) * n_rows_train * j0);
        mod_trainset[idx] = val - clusterRotCenters[j + (desc->rot_dim * l)];
      }
    }

    rmm::device_uvector<float> sub_trainset(n_rows_train * desc->lenPq, stream, &managed_memory);
    rmm::device_uvector<uint32_t> sub_trainset_labels(n_rows_train, stream, &managed_memory);
    rmm::device_uvector<float> pq_centers(
      (1 << desc->bitPq) * desc->lenPq, stream, &managed_memory);

    for (uint32_t j = 0; j < desc->pq_dim; j++) {
      float* curPqCenters = pqCenters + ((1 << desc->bitPq) * desc->lenPq) * j;
      RAFT_CUDA_TRY(cudaMemcpy(sub_trainset.data(),
                               mod_trainset.data() + ((uint64_t)n_rows_train * desc->lenPq * j),
                               sizeof(float) * n_rows_train * desc->lenPq,
                               cudaMemcpyHostToDevice));
      // Train kmeans for each PQ
      kmeans::build_clusters(handle,
                             numIterations,
                             desc->lenPq,
                             sub_trainset.data(),
                             n_rows_train,
                             (1 << desc->bitPq),
                             pq_centers.data(),
                             sub_trainset_labels.data(),
                             pq_cluster_sizes.data(),
                             raft::distance::DistanceType::L2Expanded,
                             device_memory,
                             stream);
      raft::copy(curPqCenters, pq_centers.data(), (1 << desc->bitPq) * desc->lenPq, stream);
      handle.sync_stream();
#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
      if (j == 0) {
        RAFT_CUDA_TRY(cudaDeviceSynchronize());
        _cuann_kmeans_show_centers(
          curPqCenters, (1 << desc->bitPq), desc->lenPq, pq_cluster_size.data());
      }
#endif
    }
  }

  //
  // Compute PQ code for whole data_vectors
  //
  _cuann_compute_PQ_code<T>(handle,
                            desc->numDataset,
                            desc->data_dim,
                            desc->rot_dim,
                            desc->pq_dim,
                            desc->lenPq,
                            desc->bitPq,
                            desc->numClusters,
                            desc->typePqCenter,
                            maxClusterSize,
                            cluster_centers,
                            rotationMatrix,
                            data_vectors,
                            data_indices,
                            cluster_sizes.data(),
                            cluster_offsets,
                            pqCenters,
                            numIterations,
                            pqDataset,
                            &managed_memory,
                            device_memory);

  //
  _cuann_get_inclusiveSumSortedClusterSize(
    desc, cluster_offsets, cluster_centers, &(desc->inclusiveSumSortedClusterSize));
  _cuann_get_sqsumClusters(desc, cluster_centers, &(desc->sqsumClusters));

  {
    // combine cluster_centers and sqsumClusters
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    rmm::device_uvector<float> cluster_centers_tmp(
      desc->numClusters * desc->data_dim, stream, &managed_memory);
    for (uint32_t i = 0; i < desc->numClusters * desc->data_dim; i++) {
      cluster_centers_tmp.data()[i] = cluster_centers[i];
    }
    for (uint32_t i = 0; i < desc->numClusters; i++) {
      for (uint32_t j = 0; j < desc->data_dim; j++) {
        cluster_centers[j + (desc->dimDatasetExt * i)] =
          cluster_centers_tmp.data()[j + (desc->data_dim * i)];
      }
      cluster_centers[desc->data_dim + (desc->dimDatasetExt * i)] = desc->sqsumClusters[i];
    }
  }

  //
  cuannIvfPqGetIndexSize(desc, &(header->indexSize));
  header->metric          = desc->metric;
  header->numClusters     = desc->numClusters;
  header->numDataset      = desc->numDataset;
  header->data_dim        = desc->data_dim;
  header->pq_dim          = desc->pq_dim;
  header->maxClusterSize  = maxClusterSize;
  header->rot_dim         = desc->rot_dim;
  header->bitPq           = desc->bitPq;
  header->typePqCenter    = (uint32_t)(desc->typePqCenter);
  header->dtypeDataset    = desc->dtypeDataset;
  header->dimDatasetExt   = desc->dimDatasetExt;
  header->numDatasetAdded = 0;
}

template <typename T>
auto cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex(
  const handle_t& handle,
  cuannIvfPqDescriptor_t& oldDesc,
  const T* newVectors, /* [numNewVectors, data_dim] */
  uint32_t numNewVectors) -> cuannIvfPqDescriptor_t
{
  cudaDataType_t dtype = oldDesc->dtypeDataset;
  if constexpr (std::is_same_v<T, float>) {
    RAFT_EXPECTS(
      dtype == CUDA_R_32F,
      "The old index type (%d) doesn't much CUDA_R_32F required by the template instantiation",
      dtype);
  } else if constexpr (std::is_same_v<T, uint8_t>) {
    RAFT_EXPECTS(
      dtype == CUDA_R_8U,
      "The old index type (%d) doesn't much CUDA_R_8U required by the template instantiation",
      dtype);
  } else if constexpr (std::is_same_v<T, int8_t>) {
    RAFT_EXPECTS(
      dtype == CUDA_R_8I,
      "The old index type (%d) doesn't much CUDA_R_8I required by the template instantiation",
      dtype);
  } else {
    static_assert(
      std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
      "unsupported type");
  }

  rmm::mr::device_memory_resource* device_memory = nullptr;
  auto pool_guard = raft::get_pool_memory_resource(device_memory, 1024 * 1024);
  if (pool_guard) {
    RAFT_LOG_DEBUG(
      "cuannIvfPqCreateNewIndexByAddingVectorsToOldIndex: using pool memory resource with initial "
      "size %zu bytes",
      pool_guard->pool_size());
  }

  rmm::mr::managed_memory_resource managed_memory_upstream;
  rmm::mr::pool_memory_resource<rmm::mr::managed_memory_resource> managed_memory(
    &managed_memory_upstream, 1024 * 1024);

  char dtypeString[64];
  _cuann_get_dtype_string(dtype, dtypeString);
  RAFT_LOG_DEBUG("dtype: %s", dtypeString);
  RAFT_LOG_DEBUG("data_dim: %u", oldDesc->data_dim);
  struct cuannIvfPqIndexHeader* oldHeader;
  float* oldClusterCenters;       // [numClusters, dimDatasetExt]
  float* oldPqCenters;            // [pq_dim, 1 << bitPq, lenPq], or
                                  // [numClusters, 1 << bitPq, lenPq]
  uint8_t* oldPqDataset;          // [numDataset, pq_dim * bitPq / 8]
  uint32_t* oldOriginalNumbers;   // [numDataset]
  uint32_t* old_cluster_offsets;  // [numClusters + 1]
  float* oldRotationMatrix;       // [data_dim, rot_dim]
  float* oldClusterRotCenters;    // [numClusters, rot_dim]
  _cuann_get_index_pointers(oldDesc,
                            &oldHeader,
                            &oldClusterCenters,
                            &oldPqCenters,
                            &oldPqDataset,
                            &oldOriginalNumbers,
                            &old_cluster_offsets,
                            &oldRotationMatrix,
                            &oldClusterRotCenters);

  //
  // The cluster_centers stored in index contain data other than cluster
  // centroids to speed up the search. Here, only the cluster centroids
  // are extracted.
  //

  rmm::device_uvector<float> cluster_centers(
    oldDesc->numClusters * oldDesc->data_dim, handle.get_stream(), &managed_memory);
  for (uint32_t i = 0; i < oldDesc->numClusters; i++) {
    memcpy(cluster_centers.data() + (uint64_t)i * oldDesc->data_dim,
           oldClusterCenters + (uint64_t)i * oldDesc->dimDatasetExt,
           sizeof(float) * oldDesc->data_dim);
  }

  //
  // Use the existing cluster centroids to find the label (cluster ID)
  // of the vector to be added.
  //

  rmm::device_uvector<uint32_t> new_data_labels(
    numNewVectors, handle.get_stream(), &managed_memory);
  utils::memzero(new_data_labels.data(), numNewVectors, handle.get_stream());
  rmm::device_uvector<uint32_t> cluster_sizes(
    oldDesc->numClusters, handle.get_stream(), &managed_memory);
  utils::memzero(cluster_sizes.data(), oldDesc->numClusters, handle.get_stream());

  kmeans::predict(handle,
                  cluster_centers.data(),
                  oldDesc->numClusters,
                  oldDesc->data_dim,
                  newVectors,
                  numNewVectors,
                  new_data_labels.data(),
                  oldDesc->metric,
                  handle.get_stream());
  raft::stats::histogram<uint32_t, size_t>(raft::stats::HistTypeAuto,
                                           reinterpret_cast<int32_t*>(cluster_sizes.data()),
                                           size_t(oldDesc->numClusters),
                                           new_data_labels.data(),
                                           numNewVectors,
                                           1,
                                           handle.get_stream());
  handle.sync_stream();

#if (RAFT_ACTIVE_LEVEL >= RAFT_LEVEL_DEBUG)
  {
    const int _num_show = 10;
    fprintf(stderr, "# numNewVectors: %u\n", numNewVectors);
    fprintf(stderr, "# new_data_labels: ");
    for (uint32_t i = 0; i < numNewVectors; i++) {
      if ((i < _num_show) || (numNewVectors - i <= _num_show)) {
        fprintf(stderr, "%u, ", new_data_labels.data()[i]);
      } else if (i == _num_show) {
        fprintf(stderr, "..., ");
      }
    }
    fprintf(stderr, "\n");
  }
  {
    const int _num_show = 10;
    fprintf(stderr, "# oldDesc->numClusters: %u\n", oldDesc->numClusters);
    fprintf(stderr, "# cluster_sizes: ");
    int _sum = 0;
    for (uint32_t i = 0; i < oldDesc->numClusters; i++) {
      _sum += cluster_sizes.data()[i];
      if ((i < _num_show) || (oldDesc->numClusters - i <= _num_show)) {
        fprintf(stderr, "%u, ", cluster_sizes.data()[i]);
      } else if (i == _num_show) {
        fprintf(stderr, "..., ");
      }
    }
    fprintf(stderr, "\n");
    fprintf(stderr, "# _sum: %d\n", _sum);
  }
#endif

  //
  // Make cluster_offsets, data_indices
  //
  uint32_t maxClusterSize = 0;
  std::vector<uint32_t> cluster_offsets(oldDesc->numClusters + 1, 0);
  std::vector<uint32_t> data_indices(numNewVectors);
  // cluster_offsets
  cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < oldDesc->numClusters; l++) {
    cluster_offsets[l + 1] = cluster_offsets[l] + cluster_sizes.data()[l];
    maxClusterSize         = max(maxClusterSize, cluster_sizes.data()[l]);
  }
  RAFT_EXPECTS(cluster_offsets[oldDesc->numClusters] == numNewVectors,
               "cluster sizes do not add up.");
  // data_indices
  for (uint32_t i = 0; i < numNewVectors; i++) {
    uint32_t l                       = new_data_labels.data()[i];
    data_indices[cluster_offsets[l]] = i;
    cluster_offsets[l] += 1;
  }
  // Recover cluster_offsets
  for (uint32_t l = 0; l < oldDesc->numClusters; l++) {
    cluster_offsets[l] -= cluster_sizes.data()[l];
  }

  //
  // Compute PQ code for new vectors
  //
  rmm::device_uvector<uint8_t> new_pq_codes(
    numNewVectors * oldDesc->pq_dim * oldDesc->bitPq / 8, handle.get_stream(), &managed_memory);
  _cuann_compute_PQ_code<T>(handle,
                            numNewVectors,
                            oldDesc->data_dim,
                            oldDesc->rot_dim,
                            oldDesc->pq_dim,
                            oldDesc->lenPq,
                            oldDesc->bitPq,
                            oldDesc->numClusters,
                            oldDesc->typePqCenter,
                            maxClusterSize,
                            cluster_centers.data(),
                            oldRotationMatrix,
                            newVectors,
                            data_indices.data(),
                            cluster_sizes.data(),
                            cluster_offsets.data(),
                            oldPqCenters,
                            0,
                            new_pq_codes.data(),
                            &managed_memory,
                            device_memory);

  //
  // Create descriptor for new index
  //
  auto newDesc = cuannIvfPqCreateDescriptor();
  memcpy(newDesc.get(), oldDesc.get(), sizeof(struct cuannIvfPqDescriptor));
  newDesc->numDataset += numNewVectors;
  newDesc->inclusiveSumSortedClusterSize = nullptr;
  newDesc->sqsumClusters                 = nullptr;
  newDesc->index_ptr                     = nullptr;
  RAFT_LOG_DEBUG("numDataset: %u -> %u", oldDesc->numDataset, newDesc->numDataset);

  //
  // Allocate memory for new index
  //
  size_t newIndexSize;
  cuannIvfPqGetIndexSize(newDesc, &newIndexSize);
  RAFT_LOG_DEBUG("indexSize: %lu -> %lu", oldHeader->indexSize, newIndexSize);
  RAFT_CUDA_TRY(cudaMallocManaged(&(newDesc->index_ptr), newIndexSize));
  memset(newDesc->index_ptr, 0, newIndexSize);
  struct cuannIvfPqIndexHeader* newHeader;
  float* newClusterCenters;       // [numClusters, dimDatasetExt]
  float* newPqCenters;            // [pq_dim, 1 << bitPq, lenPq], or
                                  // [numClusters, 1 << bitPq, lenPq]
  uint8_t* newPqDataset;          // [numDataset, pq_dim * bitPq / 8]  ***
  uint32_t* newOriginalNumbers;   // [numDataset]  ***
  uint32_t* new_cluster_offsets;  // [numClusters + 1]  ***
  float* newRotationMatrix;       // [data_dim, rot_dim]
  float* newClusterRotCenters;    // [numClusters, rot_dim]
  _cuann_get_index_pointers(newDesc,
                            &newHeader,
                            &newClusterCenters,
                            &newPqCenters,
                            &newPqDataset,
                            &newOriginalNumbers,
                            &new_cluster_offsets,
                            &newRotationMatrix,
                            &newClusterRotCenters);

  //
  // Copy the unchanged parts
  //    header, cluster_centers, pqCenters, rotationMatrix, clusterRotCenters
  //
  memcpy(newHeader, oldHeader, sizeof(struct cuannIvfPqIndexHeader));
  {
    cuannIvfPqGetIndexSize(newDesc, &(newHeader->indexSize));
    newHeader->numDataset = newDesc->numDataset;
    newHeader->numDatasetAdded += numNewVectors;
  }
  memcpy(newClusterCenters, oldClusterCenters, _cuann_getIndexSize_cluster_centers(oldDesc));
  memcpy(newPqCenters, oldPqCenters, _cuann_getIndexSize_pqCenters(oldDesc));
  memcpy(newRotationMatrix, oldRotationMatrix, _cuann_getIndexSize_rotationMatrix(oldDesc));
  memcpy(
    newClusterRotCenters, oldClusterRotCenters, _cuann_getIndexSize_clusterRotCenters(oldDesc));

  //
  // Make new_cluster_offsets
  //
  maxClusterSize         = 0;
  new_cluster_offsets[0] = 0;
  for (uint32_t l = 0; l < newDesc->numClusters; l++) {
    uint32_t oldClusterSize    = old_cluster_offsets[l + 1] - old_cluster_offsets[l];
    new_cluster_offsets[l + 1] = new_cluster_offsets[l];
    new_cluster_offsets[l + 1] += oldClusterSize + cluster_sizes.data()[l];
    maxClusterSize = max(maxClusterSize, oldClusterSize + cluster_sizes.data()[l]);
  }
  {
    newDesc->maxClusterSize   = maxClusterSize;
    newHeader->maxClusterSize = maxClusterSize;
  }
  RAFT_LOG_DEBUG("maxClusterSize: %u -> %u", oldDesc->maxClusterSize, newDesc->maxClusterSize);

  //
  // Make newOriginalNumbers
  //
  for (uint32_t i = 0; i < numNewVectors; i++) {
    data_indices[i] += oldDesc->numDataset;
  }
  for (uint32_t l = 0; l < newDesc->numClusters; l++) {
    uint32_t oldClusterSize = old_cluster_offsets[l + 1] - old_cluster_offsets[l];
    memcpy(newOriginalNumbers + new_cluster_offsets[l],
           oldOriginalNumbers + old_cluster_offsets[l],
           sizeof(uint32_t) * oldClusterSize);
    memcpy(newOriginalNumbers + new_cluster_offsets[l] + oldClusterSize,
           data_indices.data() + cluster_offsets[l],
           sizeof(uint32_t) * cluster_sizes.data()[l]);
  }

  //
  // Make newPqDataset
  //
  size_t unitPqDataset = newDesc->pq_dim * newDesc->bitPq / 8;
  for (uint32_t l = 0; l < newDesc->numClusters; l++) {
    uint32_t oldClusterSize = old_cluster_offsets[l + 1] - old_cluster_offsets[l];
    memcpy(newPqDataset + unitPqDataset * new_cluster_offsets[l],
           oldPqDataset + unitPqDataset * old_cluster_offsets[l],
           sizeof(uint8_t) * unitPqDataset * oldClusterSize);
    memcpy(newPqDataset + unitPqDataset * (new_cluster_offsets[l] + oldClusterSize),
           new_pq_codes.data() + unitPqDataset * cluster_offsets[l],
           sizeof(uint8_t) * unitPqDataset * cluster_sizes.data()[l]);
  }

  _cuann_get_inclusiveSumSortedClusterSize(
    newDesc, new_cluster_offsets, newClusterCenters, &(newDesc->inclusiveSumSortedClusterSize));

  //
  // Done
  //
  if (newHeader->numDatasetAdded * 2 > newHeader->numDataset) {
    RAFT_LOG_INFO(
      "The total number of vectors in the new index"
      " is now more than twice the initial number of vectors."
      " You may want to re-build the index from scratch."
      " (numVectors: %u, numVectorsAdded: %u)",
      newHeader->numDataset,
      newHeader->numDatasetAdded);
  }

  return newDesc;
}

// cuannIvfPqSetSearchParameters
inline void cuannIvfPqSetSearchParameters(cuannIvfPqDescriptor_t& desc,
                                          const uint32_t numProbes,
                                          const uint32_t topK)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");
  RAFT_EXPECTS(numProbes > 0, "numProbes must be larger than zero");
  RAFT_EXPECTS(topK > 0, "topK must be larger than zero");
  RAFT_EXPECTS(numProbes <= desc->numClusters,
               "numProbes (%u) must be not larger than numClusters (%u)",
               numProbes,
               desc->numClusters);
  RAFT_EXPECTS(topK <= desc->numDataset,
               "topK (%u) must be not larger than numDataset (%u)",
               numProbes,
               desc->numDataset);

  uint32_t numSamplesWorstCase = desc->numDataset;
  if (numProbes < desc->numClusters) {
    numSamplesWorstCase =
      desc->numDataset -
      desc->inclusiveSumSortedClusterSize[desc->numClusters - 1 - numProbes -
                                          desc->_numClustersSize0];  // (*) urgent WA, need to be
                                                                     // fixed.
  }
  RAFT_EXPECTS(topK <= numSamplesWorstCase,
               "numProbes is too small to get topK results reliably (numProbes: %u, topK: %u, "
               "numSamplesWorstCase: %u).",
               numProbes,
               topK,
               numSamplesWorstCase);
  desc->numProbes  = numProbes;
  desc->topK       = topK;
  desc->maxSamples = desc->inclusiveSumSortedClusterSize[numProbes - 1];
  if (desc->maxSamples % 128) { desc->maxSamples += 128 - (desc->maxSamples % 128); }
  desc->internalDistanceDtype    = CUDA_R_32F;
  desc->smemLutDtype             = CUDA_R_32F;
  desc->preferredThreadBlockSize = 0;
}

// cuannIvfPqSetSearchParameters
inline void cuannIvfPqSetSearchTuningParameters(cuannIvfPqDescriptor_t& desc,
                                                cudaDataType_t internalDistanceDtype,
                                                cudaDataType_t smemLutDtype,
                                                const uint32_t preferredThreadBlockSize)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");
  RAFT_EXPECTS(internalDistanceDtype == CUDA_R_16F || internalDistanceDtype == CUDA_R_32F,
               "internalDistanceDtype must be either CUDA_R_16F or CUDA_R_32F");
  RAFT_EXPECTS(
    smemLutDtype == CUDA_R_16F || smemLutDtype == CUDA_R_32F || smemLutDtype == CUDA_R_8U,
    "smemLutDtype must be CUDA_R_16F, CUDA_R_32F or CUDA_R_8U");
  RAFT_EXPECTS(preferredThreadBlockSize == 256 || preferredThreadBlockSize == 512 ||
                 preferredThreadBlockSize == 1024 || preferredThreadBlockSize == 0,
               "preferredThreadBlockSize must be 0, 256, 512 or 1024, but %u is given.",
               preferredThreadBlockSize);
  desc->internalDistanceDtype    = internalDistanceDtype;
  desc->smemLutDtype             = smemLutDtype;
  desc->preferredThreadBlockSize = preferredThreadBlockSize;
}

// cuannIvfPqGetSearchParameters
inline void cuannIvfPqGetSearchParameters(cuannIvfPqDescriptor_t& desc,
                                          uint32_t* numProbes,
                                          uint32_t* topK)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");
  *numProbes = desc->numProbes;
  *topK      = desc->topK;
}

// cuannIvfPqGetSearchTuningParameters
inline void cuannIvfPqGetSearchTuningParameters(cuannIvfPqDescriptor_t& desc,
                                                cudaDataType_t* internalDistanceDtype,
                                                cudaDataType_t* smemLutDtype,
                                                uint32_t* preferredThreadBlockSize)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");
  *internalDistanceDtype    = desc->internalDistanceDtype;
  *smemLutDtype             = desc->smemLutDtype;
  *preferredThreadBlockSize = desc->preferredThreadBlockSize;
}

// cuannIvfPqSearch
inline void cuannIvfPqSearch_bufferSize(const handle_t& handle,
                                        cuannIvfPqDescriptor_t& desc,
                                        uint32_t maxQueries,
                                        size_t maxWorkspaceSize,
                                        size_t* workspaceSize)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");

  size_t max_ws = maxWorkspaceSize;
  if (max_ws == 0) {
    max_ws = (size_t)1 * 1024 * 1024 * 1024;  // default, 1GB
  } else {
    max_ws = max(max_ws, (size_t)512 * 1024 * 1024);
  }

  size_t size_0 =
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // devQueries
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // curQueries
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->rot_dim) +        // rotQueries
    Pow2<128>::roundUp(sizeof(uint32_t) * maxQueries * desc->numProbes) +   // clusterLabels..
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->numClusters) +    // QCDistances
    _cuann_find_topk_bufferSize(handle, desc->numProbes, maxQueries, desc->numClusters);
  if (size_0 > max_ws) {
    maxQueries = maxQueries * max_ws / size_0;
    if (maxQueries > 32) { maxQueries -= (maxQueries % 32); }
  }
  // maxQueries = min(max(maxQueries, 1), 1024);
  // maxQueries = min(max(maxQueries, 1), 2048);
  maxQueries       = min(max(maxQueries, 1), 4096);
  desc->maxQueries = maxQueries;

  *workspaceSize =
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // devQueries
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->dimDatasetExt) +  // curQueries
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->rot_dim) +        // rotQueries
    Pow2<128>::roundUp(sizeof(uint32_t) * maxQueries * desc->numProbes);    // clusterLabels..

  max_ws -= *workspaceSize;
  desc->maxBatchSize = 1;
  while (1) {
    uint32_t nextBatchSize = desc->maxBatchSize * max_ws / ivfpq_search_bufferSize(handle, desc);
    if (desc->maxBatchSize >= nextBatchSize) break;
    desc->maxBatchSize = nextBatchSize;
  }
  desc->maxBatchSize = min(max(desc->maxBatchSize, 1), maxQueries);

  if (maxQueries > desc->maxBatchSize) {
    // Adjust maxBatchSize to reduce workspace size.
    uint32_t num = (maxQueries + desc->maxBatchSize - 1) / desc->maxBatchSize;
    if (1 < num && num < 5) { desc->maxBatchSize = (maxQueries + num - 1) / num; }
  }

  if (1) {
    // Adjust maxBatchSize to improve GPU occupancy of topk kernel.
    uint32_t numCta_total    = getMultiProcessorCount() * 2;
    uint32_t numCta_perBatch = numCta_total / desc->maxBatchSize;
    float utilization        = (float)numCta_perBatch * desc->maxBatchSize / numCta_total;
    if (numCta_perBatch > 1 || (numCta_perBatch == 1 && utilization < 0.6)) {
      uint32_t numCta_perBatch_1 = numCta_perBatch + 1;
      uint32_t maxBatchSize_1    = numCta_total / numCta_perBatch_1;
      float utilization_1        = (float)numCta_perBatch_1 * maxBatchSize_1 / numCta_total;
      if (utilization < utilization_1) { desc->maxBatchSize = maxBatchSize_1; }
    }
  }

  size_t size_1 =
    Pow2<128>::roundUp(sizeof(float) * maxQueries * desc->numClusters) +  // QCDistance
    _cuann_find_topk_bufferSize(handle, desc->numProbes, maxQueries, desc->numClusters);
  size_t size_2 = ivfpq_search_bufferSize(handle, desc);
  *workspaceSize += max(size_1, size_2);

  RAFT_LOG_TRACE("maxQueries: %u", maxQueries);
  RAFT_LOG_TRACE("maxBatchSize: %u", desc->maxBatchSize);
  RAFT_LOG_DEBUG(
    "workspaceSize: %lu (%.3f GiB)", *workspaceSize, (float)*workspaceSize / 1024 / 1024 / 1024);
}

template <typename T>
void cuannIvfPqSearch(const handle_t& handle,
                      cuannIvfPqDescriptor_t& desc,
                      const T* queries, /* [numQueries, data_dim], host or device pointer */
                      uint32_t numQueries,
                      uint64_t* neighbors, /* [numQueries, topK], device pointer */
                      float* distances,    /* [numQueries, topK], device pointer */
                      rmm::mr::device_memory_resource* mr)
{
  RAFT_EXPECTS(desc != nullptr, "the descriptor is not initialized.");
  auto stream = handle.get_stream();

  static_assert(std::is_same_v<T, float> || std::is_same_v<T, uint8_t> || std::is_same_v<T, int8_t>,
                "unsupported type");

  struct cuannIvfPqIndexHeader* header;
  float* cluster_centers;     // [numClusters, dimDatasetExt]
  float* pqCenters;           // [pq_dim, 1 << bitPq, lenPq], or
                              // [numClusters, 1 << bitPq, lenPq]
  uint8_t* pqDataset;         // [numDataset, pq_dim * bitPq / 8]
  uint32_t* data_indices;     // [numDataset]
  uint32_t* cluster_offsets;  // [numClusters + 1]
  float* rotationMatrix;      // [data_dim, rot_dim]
  float* clusterRotCenters;   // [numClusters, rot_dim]
  _cuann_get_index_pointers(desc,
                            &header,
                            &cluster_centers,
                            &pqCenters,
                            &pqDataset,
                            &data_indices,
                            &cluster_offsets,
                            &rotationMatrix,
                            &clusterRotCenters);
  //
  rmm::device_uvector<T> dev_queries(desc->maxQueries * desc->dimDatasetExt, stream, mr);
  rmm::device_uvector<float> cur_queries(desc->maxQueries * desc->dimDatasetExt, stream, mr);
  rmm::device_uvector<float> rot_queries(desc->maxQueries * desc->rot_dim, stream, mr);
  rmm::device_uvector<uint32_t> clusters_to_probe(desc->maxQueries * desc->numProbes, stream, mr);
  rmm::device_uvector<float> qc_distances(desc->maxQueries * desc->numClusters, stream, mr);

  void (*_ivfpq_search)(const handle_t&,
                        cuannIvfPqDescriptor_t&,
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
  if (desc->internalDistanceDtype == CUDA_R_16F) {
    if (desc->smemLutDtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<half, half>;
    } else if (desc->smemLutDtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<half, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<half, float>;
    }
  } else {
    if (desc->smemLutDtype == CUDA_R_16F) {
      _ivfpq_search = ivfpq_search<float, half>;
    } else if (desc->smemLutDtype == CUDA_R_8U) {
      _ivfpq_search = ivfpq_search<float, fp_8bit<5>>;
    } else {
      _ivfpq_search = ivfpq_search<float, float>;
    }
  }

  switch (utils::check_pointer_residency(neighbors, distances)) {
    case utils::pointer_residency::device_only:
    case utils::pointer_residency::host_and_device: break;
    default: RAFT_FAIL("output pointers must be accessible from the device.");
  }

  cudaPointerAttributes attr;
  RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, queries));

  for (uint32_t i = 0; i < numQueries; i += desc->maxQueries) {
    uint32_t nQueries = min(desc->maxQueries, numQueries - i);

    float fillValue = 0.0;
    if (desc->metric != raft::distance::DistanceType::InnerProduct) { fillValue = 1.0 / -2.0; }
    float divisor = 1.0;
    if (desc->dtypeDataset == CUDA_R_8U) {
      divisor = 256.0;
    } else if (desc->dtypeDataset == CUDA_R_8I) {
      divisor = 128.0;
    }
    if constexpr (std::is_same_v<T, float>) {
      float* ptrQueries = (float*)queries + ((uint64_t)(desc->data_dim) * i);
      if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        raft::copy(dev_queries.data(), ptrQueries, nQueries * desc->data_dim, stream);
        ptrQueries = dev_queries.data();
      }
      _cuann_copy_fill<float, float>(nQueries,
                                     desc->data_dim,
                                     ptrQueries,
                                     desc->data_dim,
                                     cur_queries.data(),
                                     desc->dimDatasetExt,
                                     fillValue,
                                     divisor,
                                     stream);
    }
    if constexpr (std::is_same_v<T, uint8_t>) {
      uint8_t* ptrQueries = (uint8_t*)queries + ((uint64_t)(desc->data_dim) * i);
      if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        raft::copy(dev_queries.data(), ptrQueries, nQueries * desc->data_dim, stream);
        ptrQueries = dev_queries.data();
      }
      _cuann_copy_fill<uint8_t, float>(nQueries,
                                       desc->data_dim,
                                       ptrQueries,
                                       desc->data_dim,
                                       cur_queries.data(),
                                       desc->dimDatasetExt,
                                       fillValue,
                                       divisor,
                                       stream);
    }
    if constexpr (std::is_same_v<T, int8_t>) {
      int8_t* ptrQueries = (int8_t*)queries + ((uint64_t)(desc->data_dim) * i);
      if (attr.type != cudaMemoryTypeDevice && attr.type != cudaMemoryTypeManaged) {
        raft::copy(dev_queries.data(), ptrQueries, nQueries * desc->data_dim, stream);
        ptrQueries = dev_queries.data();
      }
      _cuann_copy_fill<int8_t, float>(nQueries,
                                      desc->data_dim,
                                      ptrQueries,
                                      desc->data_dim,
                                      cur_queries.data(),
                                      desc->dimDatasetExt,
                                      fillValue,
                                      divisor,
                                      stream);
    }

    float alpha;
    float beta;
    uint32_t gemmK = desc->data_dim;
    if (desc->metric == distance::DistanceType::InnerProduct) {
      alpha = -1.0;
      beta  = 0.0;
    } else {
      alpha = -2.0;
      beta  = 0.0;
      gemmK = desc->data_dim + 1;
      RAFT_EXPECTS(gemmK <= desc->dimDatasetExt, "unexpected gemmK or dimDatasetExt");
    }
    linalg::gemm(handle,
                 true,
                 false,
                 desc->numClusters,
                 nQueries,
                 gemmK,
                 &alpha,
                 cluster_centers,
                 desc->dimDatasetExt,
                 cur_queries.data(),
                 desc->dimDatasetExt,
                 &beta,
                 qc_distances.data(),
                 desc->numClusters,
                 stream);

    // Rotate queries
    alpha = 1.0;
    beta  = 0.0;
    linalg::gemm(handle,
                 true,
                 false,
                 desc->rot_dim,
                 nQueries,
                 desc->data_dim,
                 &alpha,
                 rotationMatrix,
                 desc->data_dim,
                 cur_queries.data(),
                 desc->dimDatasetExt,
                 &beta,
                 rot_queries.data(),
                 desc->rot_dim,
                 stream);

    // Select neighbor clusters for each query.
    _cuann_find_topk(handle,
                     desc->numProbes,
                     nQueries,
                     desc->numClusters,
                     nullptr,
                     qc_distances.data(),
                     clusters_to_probe.data(),
                     mr,
                     false);

    for (uint32_t j = 0; j < nQueries; j += desc->maxBatchSize) {
      uint32_t batchSize = min(desc->maxBatchSize, nQueries - j);
      _ivfpq_search(handle,
                    desc,
                    batchSize,
                    clusterRotCenters,
                    pqCenters,
                    pqDataset,
                    data_indices,
                    cluster_offsets,
                    clusters_to_probe.data() + ((uint64_t)(desc->numProbes) * j),
                    rot_queries.data() + ((uint64_t)(desc->rot_dim) * j),
                    neighbors + ((uint64_t)(desc->topK) * (i + j)),
                    distances + ((uint64_t)(desc->topK) * (i + j)),
                    mr);
    }
  }
}

//
template <int bitPq, int vecLen, typename T, typename smemLutDtype = float>
__device__ inline float ivfpq_compute_score(
  uint32_t pq_dim,
  uint32_t iDataset,
  const uint8_t* pqDataset,          // [numDataset, pq_dim * bitPq / 8]
  const smemLutDtype* preCompScores  // [pq_dim, 1 << bitPq]
)
{
  float score             = 0.0;
  constexpr uint32_t bitT = sizeof(T) * 8;
  const T* headPqDataset  = (T*)(pqDataset + (uint64_t)iDataset * (pq_dim * bitPq / 8));
  for (int j = 0; j < pq_dim / vecLen; j += 1) {
    T pqCode = headPqDataset[0];
    headPqDataset += 1;
    uint32_t bitLeft = bitT;
#pragma unroll vecLen
    for (int k = 0; k < vecLen; k += 1) {
      uint8_t code = pqCode;
      if (bitLeft > bitPq) {
        // This condition is always true here (to make the compiler happy)
        if constexpr (bitT > bitPq) { pqCode >>= bitPq; }
        bitLeft -= bitPq;
      } else {
        if (k < vecLen - 1) {
          pqCode = headPqDataset[0];
          headPqDataset += 1;
        }
        code |= (pqCode << bitLeft);
        pqCode >>= (bitPq - bitLeft);
        bitLeft += (bitT - bitPq);
      }
      code &= (1 << bitPq) - 1;
      score += (float)preCompScores[code];
      preCompScores += (1 << bitPq);
    }
  }
  return score;
}

//
// (*) Restrict the peak GPU occupancy up-to 50% by "__launch_bounds__(1024, 1)",
// as there were cases where performance dropped by a factor of two or more on V100
// when the peak GPU occupancy was set to more than 50%.
//
template <int bitPq,
          int vecLen,
          typename T,
          int depth,
          bool preCompBaseDiff,
          typename outDtype,
          typename smemLutDtype>
__launch_bounds__(1024, 1) __global__ void ivfpq_compute_similarity(
  uint32_t numDataset,
  uint32_t data_dim,
  uint32_t numProbes,
  uint32_t pq_dim,
  uint32_t sizeBatch,
  uint32_t maxSamples,
  distance::DistanceType metric,
  codebook_gen typePqCenter,
  uint32_t topk,
  const float* cluster_centers,     // [numClusters, data_dim,]
  const float* pqCenters,           // [pq_dim, 1 << bitPq, lenPq,], or
                                    // [numClusetrs, 1 << bitPq, lenPq,]
  const uint8_t* pqDataset,         // [numDataset, pq_dim * bitPq / 8]
  const uint32_t* clusterIndexPtr,  // [numClusters + 1,]
  const uint32_t* _clusterLabels,   // [sizeBatch, numProbes,]
  const uint32_t* _chunkIndexPtr,   // [sizeBatch, numProbes,]
  const float* _query,              // [sizeBatch, data_dim,]
  const uint32_t* indexList,        // [sizeBatch * numProbes]
  float* _preCompScores,            // [...]
  outDtype* _output,                // [sizeBatch, maxSamples,] or [sizeBatch, numProbes, topk]
  uint32_t* _topkIndex              // [sizeBatch, numProbes, topk]
)
{
  const uint32_t lenPq = data_dim / pq_dim;

  smemLutDtype* preCompScores = (smemLutDtype*)smemArray;
  float* baseDiff             = nullptr;
  if (preCompBaseDiff) { baseDiff = (float*)(preCompScores + (pq_dim << bitPq)); }
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

  const uint32_t* clusterLabels = _clusterLabels + (numProbes * iBatch);
  const uint32_t* chunkIndexPtr = _chunkIndexPtr + (numProbes * iBatch);
  const float* query            = _query + (data_dim * iBatch);
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
  uint32_t label               = clusterLabels[iProbe];
  const float* myClusterCenter = cluster_centers + (data_dim * label);
  const float* myPqCenters;
  if (typePqCenter == codebook_gen::PER_SUBSPACE) {
    myPqCenters = pqCenters;
  } else {
    myPqCenters = pqCenters + (lenPq << bitPq) * label;
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
  for (uint32_t i = threadIdx.x; i < (pq_dim << bitPq); i += blockDim.x) {
    uint32_t iPq   = i >> bitPq;
    uint32_t iCode = i & ((1 << bitPq) - 1);
    float score    = 0.0;
    for (uint32_t j = 0; j < lenPq; j++) {
      uint32_t k = j + (lenPq * iPq);
      float diff;
      if (preCompBaseDiff) {
        diff = baseDiff[k];
      } else {
        diff = query[k] - myClusterCenter[k];
      }
      if (typePqCenter == codebook_gen::PER_SUBSPACE) {
        diff -= myPqCenters[j + (lenPq * i)];
      } else {
        diff -= myPqCenters[j + (lenPq * iCode)];
      }
      score += diff * diff;
    }
    preCompScores[i] = (smemLutDtype)score;
  }

  uint32_t iSampleBase = 0;
  if (iProbe > 0) { iSampleBase = chunkIndexPtr[iProbe - 1]; }
  uint32_t nSamples   = chunkIndexPtr[iProbe] - iSampleBase;
  uint32_t nSamples32 = nSamples;
  if (nSamples32 % 32 > 0) { nSamples32 = nSamples32 + (32 - (nSamples % 32)); }
  uint32_t iDatasetBase = clusterIndexPtr[label];

  using block_sort_t =
    topk::block_sort<topk::warp_sort_immediate, depth * WarpSize, true, outDtype, uint32_t>;
  block_sort_t block_topk(topk, reinterpret_cast<uint8_t*>(smemArray));
  const outDtype limit = block_sort_t::queue_t::kDummy;

  // Compute a distance for each sample
  for (uint32_t i = threadIdx.x; i < nSamples32; i += blockDim.x) {
    float score = limit;
    if (i < nSamples) {
      score = ivfpq_compute_score<bitPq, vecLen, T, smemLutDtype>(
        pq_dim, i + iDatasetBase, pqDataset, preCompScores);
    }
    if (!manageLocalTopk) {
      if (i < nSamples) { output[i + iSampleBase] = score; }
    } else {
      block_topk.add(score, iDatasetBase + i);
    }
  }
  if (!manageLocalTopk) { return; }
  // sync threads before the topk merging operation, because we reuse the shared memory
  __syncthreads();
  block_topk.done();
  block_topk.store(output, topkIndex);
}

//
template <int bitPq, int vecLen, typename T, int depth, bool preCompBaseDiff, typename outDtype>
__launch_bounds__(1024, 1) __global__ void ivfpq_compute_similarity_no_smem_lut(
  uint32_t numDataset,
  uint32_t data_dim,
  uint32_t numProbes,
  uint32_t pq_dim,
  uint32_t sizeBatch,
  uint32_t maxSamples,
  distance::DistanceType metric,
  codebook_gen typePqCenter,
  uint32_t topk,
  const float* cluster_centers,     // [numClusters, data_dim,]
  const float* pqCenters,           // [pq_dim, 1 << bitPq, lenPq,], or
                                    // [numClusetrs, 1 << bitPq, lenPq,]
  const uint8_t* pqDataset,         // [numDataset, pq_dim * bitPq / 8]
  const uint32_t* clusterIndexPtr,  // [numClusters + 1,]
  const uint32_t* _clusterLabels,   // [sizeBatch, numProbes,]
  const uint32_t* _chunkIndexPtr,   // [sizeBatch, numProbes,]
  const float* _query,              // [sizeBatch, data_dim,]
  const uint32_t* indexList,        // [sizeBatch * numProbes]
  float* _preCompScores,            // [..., pq_dim << bitPq,]
  outDtype* _output,                // [sizeBatch, maxSamples,] or [sizeBatch, numProbes, topk]
  uint32_t* _topkIndex              // [sizeBatch, numProbes, topk]
)
{
  const uint32_t lenPq = data_dim / pq_dim;

  float* preCompScores = _preCompScores + ((pq_dim << bitPq) * blockIdx.x);
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

    const uint32_t* clusterLabels = _clusterLabels + (numProbes * iBatch);
    const uint32_t* chunkIndexPtr = _chunkIndexPtr + (numProbes * iBatch);
    const float* query            = _query + (data_dim * iBatch);
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
    uint32_t label               = clusterLabels[iProbe];
    const float* myClusterCenter = cluster_centers + (data_dim * label);
    const float* myPqCenters;
    if (typePqCenter == codebook_gen::PER_SUBSPACE) {
      myPqCenters = pqCenters;
    } else {
      myPqCenters = pqCenters + (lenPq << bitPq) * label;
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
    for (uint32_t i = threadIdx.x; i < (pq_dim << bitPq); i += blockDim.x) {
      uint32_t iPq   = i >> bitPq;
      uint32_t iCode = i & ((1 << bitPq) - 1);
      float score    = 0.0;
      for (uint32_t j = 0; j < lenPq; j++) {
        uint32_t k = j + (lenPq * iPq);
        float diff;
        if (preCompBaseDiff) {
          diff = baseDiff[k];
        } else {
          diff = query[k] - myClusterCenter[k];
        }
        if (typePqCenter == codebook_gen::PER_SUBSPACE) {
          diff -= myPqCenters[j + (lenPq * i)];
        } else {
          diff -= myPqCenters[j + (lenPq * iCode)];
        }
        score += diff * diff;
      }
      preCompScores[i] = score;
    }

    uint32_t iSampleBase = 0;
    if (iProbe > 0) { iSampleBase = chunkIndexPtr[iProbe - 1]; }
    uint32_t nSamples   = chunkIndexPtr[iProbe] - iSampleBase;
    uint32_t nSamples32 = nSamples;
    if (nSamples32 % 32 > 0) { nSamples32 = nSamples32 + (32 - (nSamples % 32)); }
    uint32_t iDatasetBase = clusterIndexPtr[label];

    using block_sort_t =
      topk::block_sort<topk::warp_sort_immediate, depth * WarpSize, true, outDtype, uint32_t>;
    block_sort_t block_topk(topk, reinterpret_cast<uint8_t*>(smemArray));
    const outDtype limit = block_sort_t::queue_t::kDummy;

    // Compute a distance for each sample
    for (uint32_t i = threadIdx.x; i < nSamples32; i += blockDim.x) {
      float score = limit;
      if (i < nSamples) {
        score =
          ivfpq_compute_score<bitPq, vecLen, T>(pq_dim, i + iDatasetBase, pqDataset, preCompScores);
      }
      if (!manageLocalTopk) {
        if (i < nSamples) { output[i + iSampleBase] = score; }
      } else {
        block_topk.add(score, iDatasetBase + i);
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
template <typename scoreDtype, typename smemLutDtype>
inline void ivfpq_search(const handle_t& handle,
                         cuannIvfPqDescriptor_t& desc,
                         uint32_t numQueries,
                         const float* cluster_centers,          // [numDataset, rot_dim]
                         const float* pqCenters,                // [pq_dim, 1 << desc->bitPq, lenPq]
                         const uint8_t* pqDataset,              // [numDataset, pq_dim * bitPq / 8]
                         const uint32_t* data_indices,          // [numDataset]
                         const uint32_t* cluster_offsets,       // [numClusters + 1]
                         const uint32_t* clusterLabelsToProbe,  // [numQueries, numProbes]
                         const float* query,                    // [numQueries, rot_dim]
                         uint64_t* topkNeighbors,               // [numQueries, topK]
                         float* topkDistances,                  // [numQueries, topK]
                         rmm::mr::device_memory_resource* mr)
{
  RAFT_EXPECTS(numQueries <= desc->maxBatchSize,
               "number of queries (%u) must be smaller the max batch size (%u)",
               numQueries,
               desc->maxBatchSize);
  auto stream = handle.get_stream();

  rmm::device_uvector<uint32_t> cluster_labels_out(
    desc->maxBatchSize * desc->numProbes, stream, mr);
  rmm::device_uvector<uint32_t> index_list_sorted_buf(0, stream, mr);
  uint32_t* index_list_sorted = nullptr;
  rmm::device_uvector<uint32_t> num_samples(desc->maxBatchSize, stream, mr);
  rmm::device_uvector<uint32_t> chunk_index(desc->maxBatchSize * desc->numProbes, stream, mr);
  rmm::device_uvector<uint32_t> topk_sids(desc->maxBatchSize * desc->topK, stream, mr);
  // [maxBatchSize, maxSamples] or  [maxBatchSize, numProbes, topk]
  rmm::device_uvector<scoreDtype> scores_buf(0, stream, mr);
  rmm::device_uvector<uint32_t> topk_index_buf(0, stream, mr);
  uint32_t* topk_index = nullptr;
  if (manage_local_topk(desc)) {
    scores_buf.resize(desc->maxBatchSize * desc->numProbes * desc->topK, stream);
    topk_index_buf.resize(desc->maxBatchSize * desc->numProbes * desc->topK, stream);
    topk_index = topk_index_buf.data();
  } else {
    scores_buf.resize(desc->maxBatchSize * desc->maxSamples, stream);
  }

  dim3 mcThreads(1024, 1, 1);  // DO NOT CHANGE
  dim3 mcBlocks(numQueries, 1, 1);
  ivfpq_make_chunk_index_ptr<<<mcBlocks, mcThreads, 0, stream>>>(desc->numProbes,
                                                                 numQueries,
                                                                 cluster_offsets,
                                                                 clusterLabelsToProbe,
                                                                 chunk_index.data(),
                                                                 num_samples.data());

  if (numQueries * desc->numProbes > 256) {
    // Sorting index by cluster number (label).
    // The goal is to incrase the L2 cache hit rate to read the vectors
    // of a cluster by processing the cluster at the same time as much as
    // possible.
    index_list_sorted_buf.resize(desc->maxBatchSize * desc->numProbes, stream);
    rmm::device_uvector<uint32_t> index_list_buf(desc->maxBatchSize * desc->numProbes, stream, mr);
    rmm::device_buffer cub_workspace(desc->sizeCubWorkspace, stream, mr);
    auto index_list   = index_list_buf.data();
    index_list_sorted = index_list_sorted_buf.data();
    thrust::sequence(handle.get_thrust_policy(),
                     thrust::device_pointer_cast(index_list),
                     thrust::device_pointer_cast(index_list + numQueries * desc->numProbes));

    int begin_bit = 0;
    int end_bit   = sizeof(uint32_t) * 8;
    cub::DeviceRadixSort::SortPairs(cub_workspace.data(),
                                    desc->sizeCubWorkspace,
                                    clusterLabelsToProbe,
                                    cluster_labels_out.data(),
                                    index_list,
                                    index_list_sorted,
                                    numQueries * desc->numProbes,
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

#define SET_KERNEL2(B, M, D)                                                   \
  do {                                                                         \
    RAFT_EXPECTS(desc->pq_dim % M == 0, "pq_dim must be a multiple of %u", M); \
    if (desc->pq_dim % (M * 8) == 0) {                                         \
      SET_KERNEL1(B, (M * 8), uint64_t, D);                                    \
    } else if (desc->pq_dim % (M * 4) == 0) {                                  \
      SET_KERNEL1(B, (M * 4), uint32_t, D);                                    \
    } else if (desc->pq_dim % (M * 2) == 0) {                                  \
      SET_KERNEL1(B, (M * 2), uint16_t, D);                                    \
    } else if (desc->pq_dim % (M * 1) == 0) {                                  \
      SET_KERNEL1(B, (M * 1), uint8_t, D);                                     \
    }                                                                          \
  } while (0)

#define SET_KERNEL3(D)                     \
  do {                                     \
    switch (desc->bitPq) {                 \
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
  if (manage_local_topk(desc)) {
    while (depth * WarpSize < desc->topK) {
      depth *= 2;
    }
  }
  switch (depth) {
    case 1: SET_KERNEL3(1); break;
    case 2: SET_KERNEL3(2); break;
    case 4: SET_KERNEL3(4); break;
    default: RAFT_FAIL("ivf_pq::search(k = %u): depth value is too big (%d)", desc->topK, depth);
  }
  RAFT_LOG_DEBUG("ivf_pq::search(k = %u, depth = %d, dim = %u/%u/%u)",
                 desc->topK,
                 depth,
                 desc->data_dim,
                 desc->rot_dim,
                 desc->pq_dim);
  constexpr size_t thresholdSmem = 48 * 1024;
  size_t sizeSmem                = sizeof(smemLutDtype) * desc->pq_dim * (1 << desc->bitPq);
  size_t sizeSmemBaseDiff        = sizeof(float) * desc->rot_dim;

  uint32_t numCTAs = numQueries * desc->numProbes;
  int numThreads   = 1024;
  // desc->preferredThreadBlockSize == 0 means using auto thread block size calculation mode
  if (desc->preferredThreadBlockSize == 0) {
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
    numThreads = desc->preferredThreadBlockSize;
  }
  size_t sizeSmemForLocalTopk = topk::template calc_smem_size_for_block_wide<float, uint32_t>(
    numThreads / WarpSize, desc->topK);
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
        topk::calc_smem_size_for_block_wide<float, uint32_t>(numThreads / WarpSize, desc->topK);
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
    numCTAs * desc->pq_dim * (1 << desc->bitPq), stream, mr);
  dim3 ctaThreads(numThreads, 1, 1);
  dim3 ctaBlocks(numCTAs, 1, 1);
  kernel<<<ctaBlocks, ctaThreads, sizeSmem, stream>>>(desc->numDataset,
                                                      desc->rot_dim,
                                                      desc->numProbes,
                                                      desc->pq_dim,
                                                      numQueries,
                                                      desc->maxSamples,
                                                      desc->metric,
                                                      desc->typePqCenter,
                                                      desc->topK,
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
                     desc->topK,
                     numQueries,
                     desc->maxSamples,
                     num_samples.data(),
                     scores_buf.data(),
                     topk_sids.data(),
                     mr);
  } else {
    _cuann_find_topk(handle,
                     desc->topK,
                     numQueries,
                     (desc->numProbes * desc->topK),
                     nullptr,
                     scores_buf.data(),
                     topk_sids.data(),
                     mr);
  }

  dim3 moThreads(128, 1, 1);
  dim3 moBlocks((desc->topK + moThreads.x - 1) / moThreads.x, numQueries, 1);
  ivfpq_make_outputs<scoreDtype><<<moBlocks, moThreads, 0, stream>>>(desc->numProbes,
                                                                     desc->topK,
                                                                     desc->maxSamples,
                                                                     numQueries,
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

}  // namespace raft::spatial::knn::ivf_pq::detail
