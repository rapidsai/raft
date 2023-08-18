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
#include <type_traits>
namespace raft::neighbors::cagra::detail {
// Ref:
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
// (39.2.2 A Work-Efficient Parallel Scan)
template <class T>
__device__ T exclusive_warp_scan_step1(const T input, const unsigned lane_id)
{
  constexpr unsigned warp_size = 32;
  auto sum                     = input;
  for (auto mask = 1; mask < warp_size; mask <<= 1) {
    const auto v        = __shfl_xor_sync(0xffffffff, sum, mask);
    const auto sum_mask = (mask << 1) - 1;
    if ((sum_mask & lane_id) == sum_mask) { sum += v; }
  }
  return sum;
}

template <class T>
__device__ T exclusive_warp_scan_step2(T sum, const unsigned lane_id)
{
  constexpr unsigned warp_size = 32;

  for (auto mask = warp_size >> 1; mask >= 1; mask >>= 1) {
    const auto v        = __shfl_xor_sync(~0u, sum, mask);
    const auto sum_mask = mask - 1;
    if ((sum_mask & lane_id) == sum_mask) {
      if ((lane_id ^ mask) < lane_id) {
        sum += v;
      } else {
        sum = v;
      }
    }
  }
  return sum;
}

struct inclusive;
struct exclusive;

template <class T, class Mode>
struct warp_scan;

template <class T>
struct warp_scan<T, inclusive> {
  __device__ inline void operator()(const T input, T& output)
  {
    // 1. perform exclusive scan
    const auto lane_id = threadIdx.x & 0x1f;
    auto sum           = exclusive_warp_scan_step1(input, lane_id);

    T tmp31;
    if (lane_id == 31) {
      tmp31 = sum;
      sum   = 0;
    }
    sum = exclusive_warp_scan_step2(sum, lane_id);

    // 2. Shift array to be an inclusive scan result
    sum = __shfl_down_sync(~0u, sum, 1);
    if (lane_id == 31) { sum = tmp31; }
    output = sum;
  }
};

template <class T>
struct warp_scan<T, exclusive> {
  __device__ inline void operator()(const T input, T& output)
  {
    const auto lane_id = threadIdx.x & 0x1f;
    auto sum           = exclusive_warp_scan_step1(input, lane_id);
    if (lane_id == 31) { sum = 0; }
    sum    = exclusive_warp_scan_step2(sum, lane_id);
    output = sum;
  }
};

template <class T, unsigned max_thead_block_size = 1024>
struct block_scan {
  static constexpr unsigned warp_size = 32;
  using TempStorage                   = T[max_thead_block_size / warp_size];
  TempStorage& temp;

  __device__ block_scan(TempStorage& temp) : temp(temp) {}

  template <class Mode>
  __device__ inline void scan(const T input, T& output)
  {
    T local_scan;
    // Inclusive scan within a warp
    warp_scan<T, Mode>{}(input, local_scan);

    // Merge the warp-local scan results.
    // 1. Store the largest value in the temp storage on shared memory
    const auto lane_id = threadIdx.x & 0x1f;
    const auto warp_id = threadIdx.x >> 5;
    if (lane_id == 31) {
      if constexpr (std::is_same<Mode, exclusive>::value) {
        temp[warp_id] = local_scan + input;
      } else {
        temp[warp_id] = local_scan;
      }
    }
    __syncthreads();
    // 2. Calculate the offset (Sum(temp[i])) for each warp by scan
    //                            [scanned 2] ...
    //                [scanned 1] [ temp[1] ] ...
    // +) [scanned 0] [ temp[0] ] [ temp[0] ] ...
    // -------------------------------------- ...
    //    [ warp 0  ] [ warp 1  ] [ warp 2  ] ...
    if (warp_id == 0) {
      T offset = 0;
      if (lane_id < blockDim.x / warp_size) { offset = temp[lane_id]; }

      warp_scan<T, inclusive>{}(offset, offset);

      if (lane_id < blockDim.x / warp_size) { temp[lane_id] = offset; }
    }
    __syncthreads();

    // 3. Add the offset
    if (warp_id > 0) { local_scan += temp[warp_id - 1]; }

    output = local_scan;
  }

  template <class Mode, unsigned N>
  __device__ inline void scan(const T (&input)[N], T (&output)[N])
  {
    T offset;
    if constexpr (std::is_same<Mode, inclusive>::value) {
      T sum = 0;
      for (unsigned i = 0; i < N; i++) {
        sum += input[i];
        output[i] = sum;
      }
      offset = sum;
    } else {
      T sum = 0;
      for (unsigned i = 0; i < N; i++) {
        // In case output == input
        const auto tmp = input[i];

        output[i] = sum;
        sum += tmp;
      }
      offset = sum;
    }

    scan<exclusive>(offset, offset);

    for (unsigned i = 0; i < N; i++) {
      output[i] += offset;
    }
  }
};
}  // namespace raft::neighbors::cagra::detail
