/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "../haversine_distance.cuh"
#include "registers_types.cuh"

#include <thrust/functional.h>
#include <thrust/tuple.h>

#include <cstdint>

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

struct NNComp {
  template <typename one, typename two>
  __host__ __device__ bool operator()(const one& t1, const two& t2)
  {
    // sort first by each sample's reference landmark,
    if (thrust::get<0>(t1) < thrust::get<0>(t2)) return true;
    if (thrust::get<0>(t1) > thrust::get<0>(t2)) return false;

    // then by closest neighbor,
    return thrust::get<1>(t1) < thrust::get<1>(t2);
  }
};

/**
 * Zeros the bit at location h in a one-hot encoded 32-bit int array
 */
__device__ inline void _zero_bit(std::uint32_t* arr, std::uint32_t h)
{
  int bit = h % 32;
  int idx = h / 32;

  std::uint32_t assumed;
  std::uint32_t old = arr[idx];
  do {
    assumed = old;
    old     = atomicCAS(arr + idx, assumed, assumed & ~(1 << bit));
  } while (assumed != old);
}

/**
 * Returns whether or not bit at location h is nonzero in a one-hot
 * encoded 32-bit in array.
 */
__device__ inline bool _get_val(std::uint32_t* arr, std::uint32_t h)
{
  int bit = h % 32;
  int idx = h / 32;
  return (arr[idx] & (1 << bit)) > 0;
}

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
