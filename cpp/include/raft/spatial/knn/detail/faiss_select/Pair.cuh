/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <raft/spatial/knn/detail/faiss_select/MathOperators.cuh>
#include <raft/spatial/knn/detail/faiss_select/WarpShuffles.cuh>

namespace raft::spatial::knn::detail::faiss_select {

/// A simple pair type for CUDA device usage
template <typename K, typename V>
struct Pair {
  constexpr __device__ inline Pair() {}

  constexpr __device__ inline Pair(K key, V value) : k(key), v(value) {}

  __device__ inline bool operator==(const Pair<K, V>& rhs) const
  {
    return Math<K>::eq(k, rhs.k) && Math<V>::eq(v, rhs.v);
  }

  __device__ inline bool operator!=(const Pair<K, V>& rhs) const { return !operator==(rhs); }

  __device__ inline bool operator<(const Pair<K, V>& rhs) const
  {
    return Math<K>::lt(k, rhs.k) || (Math<K>::eq(k, rhs.k) && Math<V>::lt(v, rhs.v));
  }

  __device__ inline bool operator>(const Pair<K, V>& rhs) const
  {
    return Math<K>::gt(k, rhs.k) || (Math<K>::eq(k, rhs.k) && Math<V>::gt(v, rhs.v));
  }

  K k;
  V v;
};
}  // namespace raft::spatial::knn::detail::faiss_select
