/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "topk.h"

#include <cub/cub.cuh>

#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

namespace raft::neighbors::cagra::detail {
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
__device__ inline uint16_t convert(uint16_t x)
{
  if (x & 0x8000) {
    return x ^ 0xffff;
  } else {
    return x ^ 0x8000;
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
struct u16_vector {
  ushort1 x1;
  ushort2 x2;
  ushort4 x4;
  uint4 x8;
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

template <typename T>
__device__ inline void block_scan(const T input, T& output)
{
  switch (blockDim.x) {
    case 32: {
      typedef cub::BlockScan<T, 32> BlockScanT;
      __shared__ typename BlockScanT::TempStorage temp_storage;
      BlockScanT(temp_storage).InclusiveSum(input, output);
    } break;
    case 64: {
      typedef cub::BlockScan<T, 64> BlockScanT;
      __shared__ typename BlockScanT::TempStorage temp_storage;
      BlockScanT(temp_storage).InclusiveSum(input, output);
    } break;
    case 128: {
      typedef cub::BlockScan<T, 128> BlockScanT;
      __shared__ typename BlockScanT::TempStorage temp_storage;
      BlockScanT(temp_storage).InclusiveSum(input, output);
    } break;
    case 256: {
      typedef cub::BlockScan<T, 256> BlockScanT;
      __shared__ typename BlockScanT::TempStorage temp_storage;
      BlockScanT(temp_storage).InclusiveSum(input, output);
    } break;
    case 512: {
      typedef cub::BlockScan<T, 512> BlockScanT;
      __shared__ typename BlockScanT::TempStorage temp_storage;
      BlockScanT(temp_storage).InclusiveSum(input, output);
    } break;
    case 1024: {
      typedef cub::BlockScan<T, 1024> BlockScanT;
      __shared__ typename BlockScanT::TempStorage temp_storage;
      BlockScanT(temp_storage).InclusiveSum(input, output);
    } break;
    default: break;
  }
}

//
template <typename T, int stateBitLen, int vecLen>
__device__ inline void update_histogram(int itr,
                                        uint32_t thread_id,
                                        uint32_t num_threads,
                                        uint32_t hint,
                                        uint32_t threshold,
                                        uint32_t& num_bins,
                                        uint32_t& shift,
                                        const T* x,  // [nx,]
                                        uint32_t nx,
                                        uint32_t* hist,  // [num_bins]
                                        uint8_t* state,
                                        uint32_t* output,  // [topk]
                                        uint32_t* output_count)
{
  if (sizeof(T) == 4) {
    // 32-bit (uint32_t)
    // itr:0, calculate histogram with 11 bits from bit-21 to bit-31
    // itr:1, calculate histogram with 11 bits from bit-10 to bit-20
    // itr:2, calculate histogram with 10 bits from bit-0 to bit-9
    if (itr == 0) {
      shift    = 21;
      num_bins = 2048;
    } else if (itr == 1) {
      shift    = 10;
      num_bins = 2048;
    } else {
      shift    = 0;
      num_bins = 1024;
    }
  } else if (sizeof(T) == 2) {
    // 16-bit (uint16_t)
    // itr:0, calculate histogram with 8 bits from bit-8 to bit-15
    // itr:1, calculate histogram with 8 bits from bit-0 to bit-7
    if (itr == 0) {
      shift    = 8;
      num_bins = 256;
    } else {
      shift    = 0;
      num_bins = 256;
    }
  } else {
    return;
  }
  if (itr > 0) {
    for (int i = threadIdx.x; i < num_bins; i += blockDim.x) {
      hist[i] = 0;
    }
    __syncthreads();
  }

  // (*) Note that 'thread_id' may be different from 'threadIdx.x',
  // and 'num_threads' may be different from 'blockDim.x'
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if ((stateBitLen == 8) && (itr > 0)) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      const int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u32_vector x_u32_vec;
      struct u16_vector x_u16_vec;
      if (sizeof(T) == 4) {
        load_u32_vector<vecLen>(x_u32_vec, (const uint32_t*)x, iv);
      } else {
        load_u16_vector<vecLen>(x_u16_vec, (const uint16_t*)x, iv);
      }
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        const int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;

        uint32_t xi;
        if (sizeof(T) == 4) {
          xi = get_element_from_u32_vector<vecLen>(x_u32_vec, u);
        } else {
          xi = get_element_from_u16_vector<vecLen>(x_u16_vec, u);
        }
        if ((xi > hint) && (itr == 0)) {
          if (stateBitLen == 8) { iState |= mask; }
        } else if (xi < threshold) {
          if (stateBitLen == 8) {
            // If the condition is already met, record the index.
            output[atomicAdd(output_count, 1)] = ivu;
            iState |= mask;
          }
        } else {
          const uint32_t k = (xi - threshold) >> shift;  // 0 <= k
          if (k >= num_bins) {
            if (stateBitLen == 8) { iState |= mask; }
          } else if (k + 1 < num_bins) {
            // Update histogram
            atomicAdd(&(hist[k + 1]), 1);
          }
        }
      }
    }
    if (stateBitLen == 8) { state[thread_id + (num_threads * ii)] = iState; }
  }
  __syncthreads();
}

template <unsigned blockDim_x>
__device__ inline void select_best_index_for_next_threshold_core(uint32_t& my_index,
                                                                 uint32_t& my_csum,
                                                                 const unsigned num_bins,
                                                                 const uint32_t* const hist,
                                                                 const uint32_t nx_below_threshold,
                                                                 const uint32_t max_threshold,
                                                                 const uint32_t threshold,
                                                                 const uint32_t shift,
                                                                 const uint32_t topk)
{
  typedef cub::BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;
  if (num_bins == 2048) {
    constexpr int n_data = 2048 / blockDim_x;
    uint32_t csum[n_data];
    for (int i = 0; i < n_data; i++) {
      csum[i] = hist[i + (n_data * threadIdx.x)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);
    for (int i = n_data - 1; i >= 0; i--) {
      if (nx_below_threshold + csum[i] > topk) continue;
      const uint32_t index = i + (n_data * threadIdx.x);
      if (threshold + (index << shift) > max_threshold) continue;
      my_index = index;
      my_csum  = csum[i];
      break;
    }
  } else if (num_bins == 1024) {
    constexpr int n_data = 1024 / blockDim_x;
    uint32_t csum[n_data];
    for (int i = 0; i < n_data; i++) {
      csum[i] = hist[i + (n_data * threadIdx.x)];
    }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);
    for (int i = n_data - 1; i >= 0; i--) {
      if (nx_below_threshold + csum[i] > topk) continue;
      const uint32_t index = i + (n_data * threadIdx.x);
      if (threshold + (index << shift) > max_threshold) continue;
      my_index = index;
      my_csum  = csum[i];
      break;
    }
  }
}

//
__device__ inline void select_best_index_for_next_threshold(
  const uint32_t topk,
  const uint32_t threshold,
  const uint32_t max_threshold,
  const uint32_t nx_below_threshold,
  const uint32_t num_bins,
  const uint32_t shift,
  const uint32_t* const hist,  // [num_bins]
  uint32_t* const best_index,
  uint32_t* const best_csum)
{
  // Scan the histogram ('hist') and compute csum. Then, find the largest
  // index under the condition that the sum of the number of elements found
  // so far ('nx_below_threshold') and the csum value does not exceed the
  // topk value.
  uint32_t my_index = 0xffffffff;
  uint32_t my_csum  = 0;
  if (num_bins <= blockDim.x) {
    uint32_t csum = 0;
    if (threadIdx.x < num_bins) { csum = hist[threadIdx.x]; }
    detail::block_scan(csum, csum);
    if (threadIdx.x < num_bins) {
      const uint32_t index = threadIdx.x;
      if ((nx_below_threshold + csum <= topk) && (threshold + (index << shift) <= max_threshold)) {
        my_index = index;
        my_csum  = csum;
      }
    }
  } else {
    switch (blockDim.x) {
      case 64:
        select_best_index_for_next_threshold_core<64>(my_index,
                                                      my_csum,
                                                      num_bins,
                                                      hist,
                                                      nx_below_threshold,
                                                      max_threshold,
                                                      threshold,
                                                      shift,
                                                      topk);
        break;
      case 128:
        select_best_index_for_next_threshold_core<128>(my_index,
                                                       my_csum,
                                                       num_bins,
                                                       hist,
                                                       nx_below_threshold,
                                                       max_threshold,
                                                       threshold,
                                                       shift,
                                                       topk);
        break;
      case 256:
        select_best_index_for_next_threshold_core<256>(my_index,
                                                       my_csum,
                                                       num_bins,
                                                       hist,
                                                       nx_below_threshold,
                                                       max_threshold,
                                                       threshold,
                                                       shift,
                                                       topk);
        break;
      case 512:
        select_best_index_for_next_threshold_core<512>(my_index,
                                                       my_csum,
                                                       num_bins,
                                                       hist,
                                                       nx_below_threshold,
                                                       max_threshold,
                                                       threshold,
                                                       shift,
                                                       topk);
        break;
      case 1024:
        select_best_index_for_next_threshold_core<1024>(my_index,
                                                        my_csum,
                                                        num_bins,
                                                        hist,
                                                        nx_below_threshold,
                                                        max_threshold,
                                                        threshold,
                                                        shift,
                                                        topk);
        break;
    }
  }
  if (threadIdx.x < num_bins) {
    const int laneid = 31 - __clz(__ballot_sync(0xffffffff, (my_index != 0xffffffff)));
    if ((threadIdx.x & 0x1f) == laneid) {
      const uint32_t old_index = atomicMax(best_index, my_index);
      if (old_index < my_index) { atomicMax(best_csum, my_csum); }
    }
  }
  __syncthreads();
}

//
template <typename T, int stateBitLen, int vecLen>
__device__ inline void output_index_below_threshold(const uint32_t topk,
                                                    const uint32_t thread_id,
                                                    const uint32_t num_threads,
                                                    const uint32_t threshold,
                                                    const uint32_t nx_below_threshold,
                                                    const T* const x,  // [nx,]
                                                    const uint32_t nx,
                                                    const uint8_t* state,
                                                    uint32_t* const output,  // [topk]
                                                    uint32_t* const output_count,
                                                    uint32_t* const output_count_eq)
{
  int ii = 0;
  for (int i = thread_id * vecLen; i < nx; i += num_threads * max(vecLen, stateBitLen), ii++) {
    uint8_t iState = 0;
    if (stateBitLen == 8) {
      iState = state[thread_id + (num_threads * ii)];
      if (iState == (uint8_t)0xff) continue;
    }
#pragma unroll
    for (int v = 0; v < max(vecLen, stateBitLen); v += vecLen) {
      const int iv = i + (num_threads * v);
      if (iv >= nx) break;

      struct u32_vector u32_vec;
      struct u16_vector u16_vec;
      if (sizeof(T) == 4) {
        load_u32_vector<vecLen>(u32_vec, (const uint32_t*)x, iv);
      } else {
        load_u16_vector<vecLen>(u16_vec, (const uint16_t*)x, iv);
      }
#pragma unroll
      for (int u = 0; u < vecLen; u++) {
        const int ivu = iv + u;
        if (ivu >= nx) break;

        const uint8_t mask = (uint8_t)0x1 << (v + u);
        if ((stateBitLen == 8) && (iState & mask)) continue;

        uint32_t xi;
        if (sizeof(T) == 4) {
          xi = get_element_from_u32_vector<vecLen>(u32_vec, u);
        } else {
          xi = get_element_from_u16_vector<vecLen>(u16_vec, u);
        }
        if (xi < threshold) {
          output[atomicAdd(output_count, 1)] = ivu;
        } else if (xi == threshold) {
          // (*) If the value is equal to the threshold, the index
          // processed first is recorded. Cause of non-determinism.
          if (nx_below_threshold + atomicAdd(output_count_eq, 1) < topk) {
            output[atomicAdd(output_count, 1)] = ivu;
          }
        }
      }
    }
  }
}

//
template <typename T>
__device__ inline void swap(T& val1, T& val2)
{
  const T val0 = val1;
  val1         = val2;
  val2         = val0;
}

//
template <typename K>
__device__ inline bool swap_if_needed(K& key1, K& key2)
{
  if (key1 > key2) {
    swap<K>(key1, key2);
    return true;
  }
  return false;
}

//
template <typename K, typename V>
__device__ inline bool swap_if_needed(K& key1, K& key2, V& val1, V& val2)
{
  if (key1 > key2) {
    swap<K>(key1, key2);
    swap<V>(val1, val2);
    return true;
  }
  return false;
}

//
template <typename K, typename V>
__device__ inline bool swap_if_needed(K& key1, K& key2, V& val1, V& val2, bool ascending)
{
  if (key1 == key2) { return false; }
  if ((key1 > key2) == ascending) {
    swap<K>(key1, key2);
    swap<V>(val1, val2);
    return true;
  }
  return false;
}

//
template <typename T>
__device__ inline T max_value_of();
template <>
__device__ inline float max_value_of<float>()
{
  return FLT_MAX;
}
template <>
__device__ inline uint32_t max_value_of<uint32_t>()
{
  return ~0u;
}

template <int stateBitLen, unsigned BLOCK_SIZE = 0>
__device__ __host__ inline uint32_t get_state_size(uint32_t len_x)
{
#ifdef __CUDA_ARCH__
  const uint32_t num_threads = blockDim.x;
#else
  const uint32_t num_threads = BLOCK_SIZE;
#endif
  if (stateBitLen == 8) {
    uint32_t numElements_perThread = (len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread    = (numElements_perThread + stateBitLen - 1) / stateBitLen;
    return numState_perThread * num_threads;
  }
  return 0;
}

//
template <int stateBitLen, int vecLen, int maxTopk, int numSortThreads, class ValT>
__device__ inline void topk_cta_11_core(uint32_t topk,
                                        uint32_t len_x,
                                        const uint32_t* _x,    // [size_batch, ld_x,]
                                        const ValT* _in_vals,  // [size_batch, ld_iv,]
                                        uint32_t* _y,          // [size_batch, ld_y,]
                                        ValT* _out_vals,       // [size_batch, ld_ov,]
                                        uint8_t* _state,       // [size_batch, ...,]
                                        uint32_t* _hint,
                                        bool sort,
                                        uint32_t* _smem)
{
  uint32_t* const smem_out_vals = _smem;
  uint32_t* const hist          = &(_smem[2 * maxTopk]);
  uint32_t* const best_index    = &(_smem[2 * maxTopk + 2048]);
  uint32_t* const best_csum     = &(_smem[2 * maxTopk + 2048 + 3]);

  const uint32_t num_threads = blockDim.x;
  const uint32_t thread_id   = threadIdx.x;
  uint32_t nx                = len_x;
  const uint32_t* const x    = _x;
  const ValT* in_vals        = NULL;
  if (_in_vals) { in_vals = _in_vals; }
  uint32_t* y = NULL;
  if (_y) { y = _y; }
  ValT* out_vals = NULL;
  if (_out_vals) { out_vals = _out_vals; }
  uint8_t* state      = _state;
  const uint32_t hint = (_hint == NULL ? ~0u : *_hint);

  // Initialize shared memory
  for (int i = 2 * maxTopk + thread_id; i < 2 * maxTopk + 2048 + 8; i += num_threads) {
    _smem[i] = 0;
  }
  uint32_t* const output_count    = &(_smem[2 * maxTopk + 2048 + 6]);
  uint32_t* const output_count_eq = &(_smem[2 * maxTopk + 2048 + 7]);
  uint32_t threshold              = 0;
  uint32_t nx_below_threshold     = 0;
  __syncthreads();

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() <= topk".
  //
#pragma unroll
  for (int j = 0; j < 3; j += 1) {
    uint32_t num_bins;
    uint32_t shift;

    update_histogram<uint32_t, stateBitLen, vecLen>(j,
                                                    thread_id,
                                                    num_threads,
                                                    hint,
                                                    threshold,
                                                    num_bins,
                                                    shift,
                                                    x,
                                                    nx,
                                                    hist,
                                                    state,
                                                    smem_out_vals,
                                                    output_count);
    select_best_index_for_next_threshold(topk,
                                         threshold,
                                         hint,
                                         nx_below_threshold,
                                         num_bins,
                                         shift,
                                         hist,
                                         best_index + j,
                                         best_csum + j);

    threshold += (best_index[j] << shift);
    nx_below_threshold += best_csum[j];
    if (nx_below_threshold == topk) break;
  }

  if ((_hint != NULL) && (thread_id == 0)) { *_hint = min(threshold, hint); }

  //
  // Output index that satisfies "x[i] < threshold".
  //
  output_index_below_threshold<uint32_t, stateBitLen, vecLen>(topk,
                                                              thread_id,
                                                              num_threads,
                                                              threshold,
                                                              nx_below_threshold,
                                                              x,
                                                              nx,
                                                              state,
                                                              smem_out_vals,
                                                              output_count,
                                                              output_count_eq);
  __syncthreads();

#ifdef CUANN_DEBUG
  if (thread_id == 0 && output_count[0] < topk) {
    RAFT_LOG_DEBUG(
      "# i_batch:%d, topk:%d, output_count:%d, nx_below_threshold:%d, threshold:%08x\n",
      i_batch,
      topk,
      output_count[0],
      nx_below_threshold,
      threshold);
  }
#endif

  if (!sort) {
    for (int k = thread_id; k < topk; k += blockDim.x) {
      const uint32_t i = smem_out_vals[k];
      if (y) { y[k] = x[i]; }
      if (out_vals) {
        if (in_vals) {
          out_vals[k] = in_vals[i];
        } else {
          out_vals[k] = i;
        }
      }
    }
    return;
  }

  constexpr int numTopkPerThread = maxTopk / numSortThreads;
  float my_keys[numTopkPerThread];
  ValT my_vals[numTopkPerThread];

  // Read keys and values to registers
  if (thread_id < numSortThreads) {
    for (int i = 0; i < numTopkPerThread; i++) {
      const int k = thread_id + (numSortThreads * i);
      if (k < topk) {
        const int j = smem_out_vals[k];
        my_keys[i]  = ((float*)x)[j];
        if (in_vals) {
          my_vals[i] = in_vals[j];
        } else {
          my_vals[i] = j;
        }
      } else {
        my_keys[i] = FLT_MAX;
        my_vals[i] = ~static_cast<ValT>(0);
      }
    }
  }

  uint32_t mask = 1;

  // Sorting by thread
  if (thread_id < numSortThreads) {
    const bool ascending = ((thread_id & mask) == 0);
    if (numTopkPerThread == 3) {
      swap_if_needed<float, ValT>(my_keys[0], my_keys[1], my_vals[0], my_vals[1], ascending);
      swap_if_needed<float, ValT>(my_keys[0], my_keys[2], my_vals[0], my_vals[2], ascending);
      swap_if_needed<float, ValT>(my_keys[1], my_keys[2], my_vals[1], my_vals[2], ascending);
    } else {
      for (int j = 0; j < numTopkPerThread / 2; j += 1) {
#pragma unroll
        for (int i = 0; i < numTopkPerThread; i += 2) {
          swap_if_needed<float, ValT>(
            my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
        }
#pragma unroll
        for (int i = 1; i < numTopkPerThread - 1; i += 2) {
          swap_if_needed<float, ValT>(
            my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
        }
      }
    }
  }

  // Bitonic Sorting
  while (mask < numSortThreads) {
    uint32_t next_mask = mask << 1;

    for (uint32_t curr_mask = mask; curr_mask > 0; curr_mask >>= 1) {
      const bool ascending = ((thread_id & curr_mask) == 0) == ((thread_id & next_mask) == 0);
      if (curr_mask >= 32) {
        // inter warp
        ValT* const smem_vals = reinterpret_cast<ValT*>(_smem);  // [maxTopk]
        float* const smem_keys =
          reinterpret_cast<float*>(smem_vals + maxTopk);  // [numTopkPerThread, numSortThreads]
        __syncthreads();
        if (thread_id < numSortThreads) {
#pragma unroll
          for (int i = 0; i < numTopkPerThread; i++) {
            smem_keys[thread_id + (numSortThreads * i)] = my_keys[i];
            smem_vals[thread_id + (numSortThreads * i)] = my_vals[i];
          }
        }
        __syncthreads();
        if (thread_id < numSortThreads) {
#pragma unroll
          for (int i = 0; i < numTopkPerThread; i++) {
            float opp_key = smem_keys[(thread_id ^ curr_mask) + (numSortThreads * i)];
            ValT opp_val  = smem_vals[(thread_id ^ curr_mask) + (numSortThreads * i)];
            swap_if_needed<float, ValT>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
          }
        }
      } else {
        // intra warp
        if (thread_id < numSortThreads) {
#pragma unroll
          for (int i = 0; i < numTopkPerThread; i++) {
            float opp_key = __shfl_xor_sync(0xffffffff, my_keys[i], curr_mask);
            ValT opp_val  = __shfl_xor_sync(0xffffffff, my_vals[i], curr_mask);
            swap_if_needed<float, ValT>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
          }
        }
      }
    }

    if (thread_id < numSortThreads) {
      const bool ascending = ((thread_id & next_mask) == 0);
      if (numTopkPerThread == 3) {
        swap_if_needed<float, ValT>(my_keys[0], my_keys[1], my_vals[0], my_vals[1], ascending);
        swap_if_needed<float, ValT>(my_keys[0], my_keys[2], my_vals[0], my_vals[2], ascending);
        swap_if_needed<float, ValT>(my_keys[1], my_keys[2], my_vals[1], my_vals[2], ascending);
      } else {
#pragma unroll
        for (uint32_t curr_mask = numTopkPerThread / 2; curr_mask > 0; curr_mask >>= 1) {
#pragma unroll
          for (int i = 0; i < numTopkPerThread; i++) {
            const int j = i ^ curr_mask;
            if (i > j) continue;
            swap_if_needed<float, ValT>(my_keys[i], my_keys[j], my_vals[i], my_vals[j], ascending);
          }
        }
      }
    }
    mask = next_mask;
  }

  // Write sorted keys and values
  if (thread_id < numSortThreads) {
    for (int i = 0; i < numTopkPerThread; i++) {
      const int k = i + (numTopkPerThread * thread_id);
      if (k < topk) {
        if (y) { y[k] = reinterpret_cast<uint32_t*>(my_keys)[i]; }
        if (out_vals) { out_vals[k] = my_vals[i]; }
      }
    }
  }
}

namespace {

//
constexpr std::uint32_t NUM_THREADS      = 1024;  // DO NOT CHANGE
constexpr std::uint32_t STATE_BIT_LENGTH = 8;     // 0: state not used,  8: state used
constexpr std::uint32_t MAX_VEC_LENGTH   = 4;     // 1, 2, 4 or 8

//
//
int _get_vecLen(uint32_t maxSamples, int maxVecLen = MAX_VEC_LENGTH)
{
  int vecLen = min(maxVecLen, (int)MAX_VEC_LENGTH);
  while ((maxSamples % vecLen) != 0) {
    vecLen /= 2;
  }
  return vecLen;
}
}  // unnamed namespace

template <int stateBitLen, int vecLen, int maxTopk, int numSortThreads, class ValT>
__launch_bounds__(1024, 1) RAFT_KERNEL
  kern_topk_cta_11(uint32_t topk,
                   uint32_t size_batch,
                   uint32_t len_x,
                   const uint32_t* _x,  // [size_batch, ld_x,]
                   uint32_t ld_x,
                   const ValT* _in_vals,  // [size_batch, ld_iv,]
                   uint32_t ld_iv,
                   uint32_t* _y,  // [size_batch, ld_y,]
                   uint32_t ld_y,
                   ValT* _out_vals,  // [size_batch, ld_ov,]
                   uint32_t ld_ov,
                   uint8_t* _state,   // [size_batch, ...,]
                   uint32_t* _hints,  // [size_batch,]
                   bool sort)
{
  const uint32_t i_batch = blockIdx.x;
  if (i_batch >= size_batch) return;

  constexpr uint32_t smem_len = 2 * maxTopk + 2048 + 8;
  static_assert(maxTopk * (1 + utils::size_of<ValT>() / utils::size_of<uint32_t>()) <= smem_len,
                "maxTopk * sizeof(ValT) must be smaller or equal to 8192 byte");
  __shared__ uint32_t _smem[smem_len];

  topk_cta_11_core<stateBitLen, vecLen, maxTopk, numSortThreads, ValT>(
    topk,
    len_x,
    (_x == NULL ? NULL : _x + i_batch * ld_x),
    (_in_vals == NULL ? NULL : _in_vals + i_batch * ld_iv),
    (_y == NULL ? NULL : _y + i_batch * ld_y),
    (_out_vals == NULL ? NULL : _out_vals + i_batch * ld_ov),
    (_state == NULL ? NULL : _state + i_batch * get_state_size<stateBitLen>(len_x)),
    (_hints == NULL ? NULL : _hints + i_batch),
    sort,
    _smem);
}

//
size_t inline _cuann_find_topk_bufferSize(uint32_t topK,
                                          uint32_t sizeBatch,
                                          uint32_t numElements,
                                          cudaDataType_t sampleDtype)
{
  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);

  size_t workspaceSize = 1;
  // state
  if (stateBitLen == 8) {
    workspaceSize = _cuann_aligned(
      sizeof(uint8_t) * get_state_size<stateBitLen, numThreads>(numElements) * sizeBatch);
  }

  return workspaceSize;
}

template <class ValT>
inline void _cuann_find_topk(uint32_t topK,
                             uint32_t sizeBatch,
                             uint32_t numElements,
                             const float* inputKeys,  // [sizeBatch, ldIK,]
                             uint32_t ldIK,           // (*) ldIK >= numElements
                             const ValT* inputVals,   // [sizeBatch, ldIV,]
                             uint32_t ldIV,           // (*) ldIV >= numElements
                             float* outputKeys,       // [sizeBatch, ldOK,]
                             uint32_t ldOK,           // (*) ldOK >= topK
                             ValT* outputVals,        // [sizeBatch, ldOV,]
                             uint32_t ldOV,           // (*) ldOV >= topK
                             void* workspace,
                             bool sort,
                             uint32_t* hints,
                             cudaStream_t stream)
{
  assert(ldIK >= numElements);
  assert(ldIV >= numElements);
  assert(ldOK >= topK);
  assert(ldOV >= topK);

  constexpr int numThreads  = NUM_THREADS;
  constexpr int stateBitLen = STATE_BIT_LENGTH;
  assert(stateBitLen == 0 || stateBitLen == 8);

  uint8_t* state = NULL;
  if (stateBitLen == 8) { state = (uint8_t*)workspace; }

  dim3 threads(numThreads, 1, 1);
  dim3 blocks(sizeBatch, 1, 1);

  void (*cta_kernel)(uint32_t,
                     uint32_t,
                     uint32_t,
                     const uint32_t*,
                     uint32_t,
                     const ValT*,
                     uint32_t,
                     uint32_t*,
                     uint32_t,
                     ValT*,
                     uint32_t,
                     uint8_t*,
                     uint32_t*,
                     bool) = nullptr;

  // V:vecLen, K:maxTopk, T:numSortThreads
#define SET_KERNEL_VKT(V, K, T, ValT)                          \
  do {                                                         \
    assert(numThreads >= T);                                   \
    assert((K % T) == 0);                                      \
    assert((K / T) <= 4);                                      \
    cta_kernel = kern_topk_cta_11<stateBitLen, V, K, T, ValT>; \
  } while (0)

  // V: vecLen
#define SET_KERNEL_V(V, ValT)                                \
  do {                                                       \
    if (topK <= 32) {                                        \
      SET_KERNEL_VKT(V, 32, 32, ValT);                       \
    } else if (topK <= 64) {                                 \
      SET_KERNEL_VKT(V, 64, 32, ValT);                       \
    } else if (topK <= 96) {                                 \
      SET_KERNEL_VKT(V, 96, 32, ValT);                       \
    } else if (topK <= 128) {                                \
      SET_KERNEL_VKT(V, 128, 32, ValT);                      \
    } else if (topK <= 192) {                                \
      SET_KERNEL_VKT(V, 192, 64, ValT);                      \
    } else if (topK <= 256) {                                \
      SET_KERNEL_VKT(V, 256, 64, ValT);                      \
    } else if (topK <= 384) {                                \
      SET_KERNEL_VKT(V, 384, 128, ValT);                     \
    } else if (topK <= 512) {                                \
      SET_KERNEL_VKT(V, 512, 128, ValT);                     \
    } else if (topK <= 768) {                                \
      SET_KERNEL_VKT(V, 768, 256, ValT);                     \
    } else if (topK <= 1024) {                               \
      SET_KERNEL_VKT(V, 1024, 256, ValT);                    \
    } \
        /* else if (topK <= 1536) { SET_KERNEL_VKT(V, 1536, 512); } */ \
        /* else if (topK <= 2048) { SET_KERNEL_VKT(V, 2048, 512); } */ \
        /* else if (topK <= 3072) { SET_KERNEL_VKT(V, 3072, 1024); } */ \
        /* else if (topK <= 4096) { SET_KERNEL_VKT(V, 4096, 1024); } */ \
        else {                                                      \
      RAFT_FAIL("topk must be lower than or equal to 1024"); \
    }                                                        \
  } while (0)

  int _vecLen = _get_vecLen(ldIK, 2);
  if (_vecLen == 2) {
    SET_KERNEL_V(2, ValT);
  } else if (_vecLen == 1) {
    SET_KERNEL_V(1, ValT);
  }

  cta_kernel<<<blocks, threads, 0, stream>>>(topK,
                                             sizeBatch,
                                             numElements,
                                             (const uint32_t*)inputKeys,
                                             ldIK,
                                             inputVals,
                                             ldIV,
                                             (uint32_t*)outputKeys,
                                             ldOK,
                                             outputVals,
                                             ldOV,
                                             state,
                                             hints,
                                             sort);

  return;
}
}  // namespace raft::neighbors::cagra::detail
