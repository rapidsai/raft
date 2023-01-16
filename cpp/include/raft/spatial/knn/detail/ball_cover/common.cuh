/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
#include <cstdint>
#include <thrust/functional.h>
#include <thrust/tuple.h>

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

template <typename value_t, typename value_int = std::uint32_t>
struct DistFunc {
  virtual __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                                 const value_t* b,
                                                                 const value_int n_dims)
  {
    return -1;
  };
};

template <typename value_t, typename value_int = std::uint32_t>
struct HaversineFunc : public DistFunc<value_t, value_int> {
  __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                         const value_t* b,
                                                         const value_int n_dims) override
  {
    return raft::spatial::knn::detail::compute_haversine(a[0], b[0], a[1], b[1]);
  }
};

template <typename value_t, typename value_int = std::uint32_t>
struct EuclideanFunc : public DistFunc<value_t, value_int> {
  __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                         const value_t* b,
                                                         const value_int n_dims) override
  {
    value_t sum_sq = 0;
    for (value_int i = 0; i < n_dims; ++i) {
      value_t diff = a[i] - b[i];
      sum_sq += diff * diff;
    }

    return raft::sqrt(sum_sq);
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