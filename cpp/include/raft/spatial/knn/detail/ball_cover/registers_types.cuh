/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
