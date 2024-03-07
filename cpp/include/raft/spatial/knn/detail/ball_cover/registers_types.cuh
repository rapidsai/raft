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

#include "../haversine_distance.cuh"  // compute_haversine

#include <cstdint>  // uint32_t

namespace raft {
namespace spatial {
namespace knn {
namespace detail {

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

template <typename value_t, typename value_int = std::uint32_t>
struct EuclideanSqFunc : public DistFunc<value_t, value_int> {
  __device__ __host__ __forceinline__ value_t operator()(const value_t* a,
                                                         const value_t* b,
                                                         const value_int n_dims) override
  {
    value_t sum_sq = 0;
    for (value_int i = 0; i < n_dims; ++i) {
      value_t diff = a[i] - b[i];
      sum_sq += diff * diff;
    }
    return sum_sq;
  }
};

};  // namespace detail
};  // namespace knn
};  // namespace spatial
};  // namespace raft
