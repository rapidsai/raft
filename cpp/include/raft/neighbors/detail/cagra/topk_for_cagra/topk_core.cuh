/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include <assert.h>
#include <cub/cub.cuh>
#include <float.h>
#include <stdint.h>
#include <stdio.h>

namespace raft::neighbors::experimental::cagra::detail {
using namespace cub;

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

//
template <typename T, int blockDim_x, int stateBitLen, int vecLen>
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
    for (int i = threadIdx.x; i < num_bins; i += blockDim_x) {
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
      int iv = i + (num_threads * v);
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
        int ivu = iv + u;
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
          uint32_t k = (xi - threshold) >> shift;  // 0 <= k
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

//
template <int blockDim_x>
__device__ inline void select_best_index_for_next_threshold(uint32_t topk,
                                                            uint32_t threshold,
                                                            uint32_t max_threshold,
                                                            uint32_t nx_below_threshold,
                                                            uint32_t num_bins,
                                                            uint32_t shift,
                                                            const uint32_t* hist,  // [num_bins]
                                                            uint32_t* best_index,
                                                            uint32_t* best_csum)
{
  // Scan the histogram ('hist') and compute csum. Then, find the largest
  // index under the condition that the sum of the number of elements found
  // so far ('nx_below_threshold') and the csum value does not exceed the
  // topk value.
  typedef BlockScan<uint32_t, blockDim_x> BlockScanT;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  uint32_t my_index = 0xffffffff;
  uint32_t my_csum  = 0;
  if (num_bins <= blockDim_x) {
    uint32_t csum = 0;
    if (threadIdx.x < num_bins) { csum = hist[threadIdx.x]; }
    BlockScanT(temp_storage).InclusiveSum(csum, csum);
    if (threadIdx.x < num_bins) {
      uint32_t index = threadIdx.x;
      if ((nx_below_threshold + csum <= topk) && (threshold + (index << shift) <= max_threshold)) {
        my_index = index;
        my_csum  = csum;
      }
    }
  } else {
    if (num_bins == 2048) {
      constexpr int n_data = 2048 / blockDim_x;
      uint32_t csum[n_data];
      for (int i = 0; i < n_data; i++) {
        csum[i] = hist[i + (n_data * threadIdx.x)];
      }
      BlockScanT(temp_storage).InclusiveSum(csum, csum);
      for (int i = n_data - 1; i >= 0; i--) {
        if (nx_below_threshold + csum[i] > topk) continue;
        uint32_t index = i + (n_data * threadIdx.x);
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
        uint32_t index = i + (n_data * threadIdx.x);
        if (threshold + (index << shift) > max_threshold) continue;
        my_index = index;
        my_csum  = csum[i];
        break;
      }
    }
  }
  if (threadIdx.x < num_bins) {
    int laneid = 31 - __clz(__ballot_sync(0xffffffff, (my_index != 0xffffffff)));
    if ((threadIdx.x & 0x1f) == laneid) {
      uint32_t old_index = atomicMax(best_index, my_index);
      if (old_index < my_index) { atomicMax(best_csum, my_csum); }
    }
  }
  __syncthreads();
}

//
template <typename T, int stateBitLen, int vecLen>
__device__ inline void output_index_below_threshold(uint32_t topk,
                                                    uint32_t thread_id,
                                                    uint32_t num_threads,
                                                    uint32_t threshold,
                                                    uint32_t nx_below_threshold,
                                                    const T* x,  // [nx,]
                                                    uint32_t nx,
                                                    const uint8_t* state,
                                                    uint32_t* output,  // [topk]
                                                    uint32_t* output_count,
                                                    uint32_t* output_count_eq)
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
      int iv = i + (num_threads * v);
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
        int ivu = iv + u;
        if (ivu >= nx) break;

        uint8_t mask = (uint8_t)0x1 << (v + u);
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
  T val0 = val1;
  val1   = val2;
  val2   = val0;
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

template <int blockDim_x, int stateBitLen>
__device__ __host__ inline uint32_t get_state_size(uint32_t len_x)
{
  const uint32_t num_threads = blockDim_x;
  if (stateBitLen == 8) {
    uint32_t numElements_perThread = (len_x + num_threads - 1) / num_threads;
    uint32_t numState_perThread    = (numElements_perThread + stateBitLen - 1) / stateBitLen;
    return numState_perThread * num_threads;
  }
  return 0;
}

//
template <int blockDim_x, int stateBitLen, int vecLen, int maxTopk, int numSortThreads>
__device__ inline void topk_cta_11_core(uint32_t topk,
                                        uint32_t len_x,
                                        const uint32_t* _x,        // [size_batch, ld_x,]
                                        const uint32_t* _in_vals,  // [size_batch, ld_iv,]
                                        uint32_t* _y,              // [size_batch, ld_y,]
                                        uint32_t* _out_vals,       // [size_batch, ld_ov,]
                                        uint8_t* _state,           // [size_batch, ...,]
                                        uint32_t* _hint,
                                        bool sort,
                                        uint32_t* _smem)
{
  uint32_t* smem_out_vals = _smem;
  uint32_t* hist          = &(_smem[2 * maxTopk]);
  uint32_t* best_index    = &(_smem[2 * maxTopk + 2048]);
  uint32_t* best_csum     = &(_smem[2 * maxTopk + 2048 + 3]);

  const uint32_t num_threads = blockDim_x;
  const uint32_t thread_id   = threadIdx.x;
  uint32_t nx                = len_x;
  const uint32_t* x          = _x;
  const uint32_t* in_vals    = NULL;
  if (_in_vals) { in_vals = _in_vals; }
  uint32_t* y = NULL;
  if (_y) { y = _y; }
  uint32_t* out_vals = NULL;
  if (_out_vals) { out_vals = _out_vals; }
  uint8_t* state = _state;
  uint32_t hint  = (_hint == NULL ? ~0u : *_hint);

  // Initialize shared memory
  for (int i = 2 * maxTopk + thread_id; i < 2 * maxTopk + 2048 + 8; i += num_threads) {
    _smem[i] = 0;
  }
  uint32_t* output_count      = &(_smem[2 * maxTopk + 2048 + 6]);
  uint32_t* output_count_eq   = &(_smem[2 * maxTopk + 2048 + 7]);
  uint32_t threshold          = 0;
  uint32_t nx_below_threshold = 0;
  __syncthreads();

  //
  // Search for the maximum threshold that satisfies "(x < threshold).sum() <= topk".
  //
#pragma unroll
  for (int j = 0; j < 3; j += 1) {
    uint32_t num_bins;
    uint32_t shift;
    update_histogram<uint32_t, blockDim_x, stateBitLen, vecLen>(j,
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

    select_best_index_for_next_threshold<blockDim_x>(topk,
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
    for (int k = thread_id; k < topk; k += blockDim_x) {
      uint32_t i = smem_out_vals[k];
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
  uint32_t my_vals[numTopkPerThread];

  // Read keys and values to registers
  if (thread_id < numSortThreads) {
    for (int i = 0; i < numTopkPerThread; i++) {
      int k = thread_id + (numSortThreads * i);
      if (k < topk) {
        int j      = smem_out_vals[k];
        my_keys[i] = ((float*)x)[j];
        if (in_vals) {
          my_vals[i] = in_vals[j];
        } else {
          my_vals[i] = j;
        }
      } else {
        my_keys[i] = FLT_MAX;
        my_vals[i] = 0xffffffffU;
      }
    }
  }

  uint32_t mask = 1;

  // Sorting by thread
  if (thread_id < numSortThreads) {
    bool ascending = ((thread_id & mask) == 0);
    if (numTopkPerThread == 3) {
      swap_if_needed<float, uint32_t>(my_keys[0], my_keys[1], my_vals[0], my_vals[1], ascending);
      swap_if_needed<float, uint32_t>(my_keys[0], my_keys[2], my_vals[0], my_vals[2], ascending);
      swap_if_needed<float, uint32_t>(my_keys[1], my_keys[2], my_vals[1], my_vals[2], ascending);
    } else {
      for (int j = 0; j < numTopkPerThread / 2; j += 1) {
#pragma unroll
        for (int i = 0; i < numTopkPerThread; i += 2) {
          swap_if_needed<float, uint32_t>(
            my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
        }
#pragma unroll
        for (int i = 1; i < numTopkPerThread - 1; i += 2) {
          swap_if_needed<float, uint32_t>(
            my_keys[i], my_keys[i + 1], my_vals[i], my_vals[i + 1], ascending);
        }
      }
    }
  }

  // Bitonic Sorting
  while (mask < numSortThreads) {
    uint32_t next_mask = mask << 1;

    for (uint32_t curr_mask = mask; curr_mask > 0; curr_mask >>= 1) {
      bool ascending = ((thread_id & curr_mask) == 0) == ((thread_id & next_mask) == 0);
      if (curr_mask >= 32) {
        // inter warp
        uint32_t* smem_vals = _smem;  // [numTopkPerThread, numSortThreads]
        float* smem_keys    = (float*)(_smem + numTopkPerThread * numSortThreads);
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
            float opp_key    = smem_keys[(thread_id ^ curr_mask) + (numSortThreads * i)];
            uint32_t opp_val = smem_vals[(thread_id ^ curr_mask) + (numSortThreads * i)];
            swap_if_needed<float, uint32_t>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
          }
        }
      } else {
        // intra warp
        if (thread_id < numSortThreads) {
#pragma unroll
          for (int i = 0; i < numTopkPerThread; i++) {
            float opp_key    = __shfl_xor_sync(0xffffffff, my_keys[i], curr_mask);
            uint32_t opp_val = __shfl_xor_sync(0xffffffff, my_vals[i], curr_mask);
            swap_if_needed<float, uint32_t>(my_keys[i], opp_key, my_vals[i], opp_val, ascending);
          }
        }
      }
    }

    if (thread_id < numSortThreads) {
      bool ascending = ((thread_id & next_mask) == 0);
      if (numTopkPerThread == 3) {
        swap_if_needed<float, uint32_t>(my_keys[0], my_keys[1], my_vals[0], my_vals[1], ascending);
        swap_if_needed<float, uint32_t>(my_keys[0], my_keys[2], my_vals[0], my_vals[2], ascending);
        swap_if_needed<float, uint32_t>(my_keys[1], my_keys[2], my_vals[1], my_vals[2], ascending);
      } else {
#pragma unroll
        for (uint32_t curr_mask = numTopkPerThread / 2; curr_mask > 0; curr_mask >>= 1) {
#pragma unroll
          for (int i = 0; i < numTopkPerThread; i++) {
            int j = i ^ curr_mask;
            if (i > j) continue;
            swap_if_needed<float, uint32_t>(
              my_keys[i], my_keys[j], my_vals[i], my_vals[j], ascending);
          }
        }
      }
    }
    mask = next_mask;
  }

  // Write sorted keys and values
  if (thread_id < numSortThreads) {
    for (int i = 0; i < numTopkPerThread; i++) {
      int k = i + (numTopkPerThread * thread_id);
      if (k < topk) {
        if (y) { y[k] = ((uint32_t*)my_keys)[i]; }
        if (out_vals) { out_vals[k] = my_vals[i]; }
      }
    }
  }
}
}  // namespace raft::neighbors::experimental::cagra::detail