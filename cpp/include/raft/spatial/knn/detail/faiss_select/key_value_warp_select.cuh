/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <raft/core/kvp.hpp>
#include <raft/spatial/knn/detail/faiss_select/MergeNetworkUtils.cuh>
#include <raft/spatial/knn/detail/faiss_select/StaticUtils.h>

namespace raft::spatial::knn::detail::faiss_select {

//
// This file contains functions to:
//
// -perform bitonic merges on pairs of sorted lists, held in
// registers. Each list contains N * WarpSize (multiple of 32)
// elements for some N.
// The bitonic merge is implemented for arbitrary sizes;
// sorted list A of size N1 * WarpSize registers
// sorted list B of size N2 * WarpSize registers =>
// sorted list C if size (N1 + N2) * WarpSize registers. N1 and N2
// are >= 1 and don't have to be powers of 2.
//
// -perform bitonic sorts on a set of N * WarpSize key/value pairs
// held in registers, by using the above bitonic merge as a
// primitive.
// N can be an arbitrary N >= 1; i.e., the bitonic sort here supports
// odd sizes and doesn't require the input to be a power of 2.
//
// The sort or merge network is completely statically instantiated via
// template specialization / expansion and constexpr, and it uses warp
// shuffles to exchange values between warp lanes.
//
// A note about comparisons:
//
// For a sorting network of keys only, we only need one
// comparison (a < b). However, what we really need to know is
// if one lane chooses to exchange a value, then the
// corresponding lane should also do the exchange.
// Thus, if one just uses the negation !(x < y) in the higher
// lane, this will also include the case where (x == y). Thus, one
// lane in fact performs an exchange and the other doesn't, but
// because the only value being exchanged is equivalent, nothing has
// changed.
// So, you can get away with just one comparison and its negation.
//
// If we're sorting keys and values, where equivalent keys can
// exist, then this is a problem, since we want to treat (x, v1)
// as not equivalent to (x, v2).
//
// To remedy this, you can either compare with a lexicographic
// ordering (a.k < b.k || (a.k == b.k && a.v < b.v)), which since
// we're predicating all of the choices results in 3 comparisons
// being executed, or we can invert the selection so that there is no
// middle choice of equality; the other lane will likewise
// check that (b.k > a.k) (the higher lane has the values
// swapped). Then, the first lane swaps if and only if the
// second lane swaps; if both lanes have equivalent keys, no
// swap will be performed. This results in only two comparisons
// being executed.
//
// If you don't consider values as well, then this does not produce a
// consistent ordering among (k, v) pairs with equivalent keys but
// different values; for us, we don't really care about ordering or
// stability here.
//
// I have tried both re-arranging the order in the higher lane to get
// away with one comparison or adding the value to the check; both
// result in greater register consumption or lower speed than just
// performing both < and > comparisons with the variables, so I just
// stick with this.

// This function merges WarpSize / 2L lists in parallel using warp
// shuffles.
// It works on at most size-16 lists, as we need 32 threads for this
// shuffle merge.
//
// If IsBitonic is false, the first stage is reversed, so we don't
// need to sort directionally. It's still technically a bitonic sort.
template <typename K, typename V, int L, bool Dir, typename Comp, bool IsBitonic>
inline __device__ void warpBitonicMergeLE16KVP(K& k, KeyValuePair<K, V>& v)
{
  static_assert(utils::isPowerOf2(L), "L must be a power-of-2");
  static_assert(L <= WarpSize / 2, "merge list size must be <= 16");

  int laneId = raft::laneId();

  if (!IsBitonic) {
    // Reverse the first comparison stage.
    // For example, merging a list of size 8 has the exchanges:
    // 0 <-> 15, 1 <-> 14, ...
    K otherK  = shfl_xor(k, 2 * L - 1);
    K otherVk = shfl_xor(v.key, 2 * L - 1);
    V otherVv = shfl_xor(v.value, 2 * L - 1);

    KeyValuePair<K, V> otherV = KeyValuePair(otherVk, otherVv);

    // Whether we are the lesser thread in the exchange
    bool small = !(laneId & L);

    if (Dir) {
      // See the comment above how performing both of these
      // comparisons in the warp seems to win out over the
      // alternatives in practice
      bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
      assign(s, k, otherK);
      assign(s, v.key, otherV.key);
      assign(s, v.value, otherV.value);

    } else {
      bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
      assign(s, k, otherK);
      assign(s, v.value, otherV.value);
      assign(s, v.key, otherV.key);
    }
  }

#pragma unroll
  for (int stride = IsBitonic ? L : L / 2; stride > 0; stride /= 2) {
    K otherK  = shfl_xor(k, stride);
    K otherVk = shfl_xor(v.key, stride);
    V otherVv = shfl_xor(v.value, stride);

    KeyValuePair<K, V> otherV = KeyValuePair(otherVk, otherVv);

    // Whether we are the lesser thread in the exchange
    bool small = !(laneId & stride);

    if (Dir) {
      bool s = small ? Comp::gt(k, otherK) : Comp::lt(k, otherK);
      assign(s, k, otherK);
      assign(s, v.key, otherV.key);
      assign(s, v.value, otherV.value);

    } else {
      bool s = small ? Comp::lt(k, otherK) : Comp::gt(k, otherK);
      assign(s, k, otherK);
      assign(s, v.key, otherV.key);
      assign(s, v.value, otherV.value);
    }
  }
}

// Template for performing a bitonic merge of an arbitrary set of
// registers
template <typename K, typename V, int N, bool Dir, typename Comp, bool Low, bool Pow2>
struct BitonicMergeStepKVP {
};

//
// Power-of-2 merge specialization
//

// All merges eventually call this
template <typename K, typename V, bool Dir, typename Comp, bool Low>
struct BitonicMergeStepKVP<K, V, 1, Dir, Comp, Low, true> {
  static inline __device__ void merge(K k[1], KeyValuePair<K, V> v[1])
  {
    // Use warp shuffles
    warpBitonicMergeLE16KVP<K, V, 16, Dir, Comp, true>(k[0], v[0]);
  }
};

template <typename K, typename V, int N, bool Dir, typename Comp, bool Low>
struct BitonicMergeStepKVP<K, V, N, Dir, Comp, Low, true> {
  static inline __device__ void merge(K k[N], KeyValuePair<K, V> v[N])
  {
    static_assert(utils::isPowerOf2(N), "must be power of 2");
    static_assert(N > 1, "must be N > 1");

#pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      K& ka                  = k[i];
      KeyValuePair<K, V>& va = v[i];

      K& kb                  = k[i + N / 2];
      KeyValuePair<K, V>& vb = v[i + N / 2];

      bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
      swap(s, ka, kb);
      swap(s, va.key, vb.key);
      swap(s, va.value, vb.value);
    }

    {
      K newK[N / 2];
      KeyValuePair<K, V> newV[N / 2];

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        newK[i]       = k[i];
        newV[i].key   = v[i].key;
        newV[i].value = v[i].value;
      }

      BitonicMergeStepKVP<K, V, N / 2, Dir, Comp, true, true>::merge(newK, newV);

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        k[i]       = newK[i];
        v[i].key   = newV[i].key;
        v[i].value = newV[i].value;
      }
    }

    {
      K newK[N / 2];
      KeyValuePair<K, V> newV[N / 2];

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        newK[i]       = k[i + N / 2];
        newV[i].key   = v[i + N / 2].key;
        newV[i].value = v[i + N / 2].value;
      }

      BitonicMergeStepKVP<K, V, N / 2, Dir, Comp, false, true>::merge(newK, newV);

#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        k[i + N / 2]       = newK[i];
        v[i + N / 2].key   = newV[i].key;
        v[i + N / 2].value = newV[i].value;
      }
    }
  }
};

//
// Non-power-of-2 merge specialization
//

// Low recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicMergeStepKVP<K, V, N, Dir, Comp, true, false> {
  static inline __device__ void merge(K k[N], KeyValuePair<K, V> v[N])
  {
    static_assert(!utils::isPowerOf2(N), "must be non-power-of-2");
    static_assert(N >= 3, "must be N >= 3");

    constexpr int kNextHighestPowerOf2 = utils::nextHighestPowerOf2(N);

#pragma unroll
    for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
      K& ka                  = k[i];
      KeyValuePair<K, V>& va = v[i];

      K& kb                  = k[i + kNextHighestPowerOf2 / 2];
      KeyValuePair<K, V>& vb = v[i + kNextHighestPowerOf2 / 2];

      bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
      swap(s, ka, kb);
      swap(s, va.key, vb.key);
      swap(s, va.value, vb.value);
    }

    constexpr int kLowSize  = N - kNextHighestPowerOf2 / 2;
    constexpr int kHighSize = kNextHighestPowerOf2 / 2;
    {
      K newK[kLowSize];
      KeyValuePair<K, V> newV[kLowSize];

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        newK[i]       = k[i];
        newV[i].key   = v[i].key;
        newV[i].value = v[i].value;
      }

      constexpr bool kLowIsPowerOf2 = utils::isPowerOf2(N - kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kLowIsPowerOf2 = utils::isPowerOf2(kLowSize);
      BitonicMergeStepKVP<K,
                          V,
                          kLowSize,
                          Dir,
                          Comp,
                          true,  // low
                          kLowIsPowerOf2>::merge(newK, newV);

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        k[i]       = newK[i];
        v[i].key   = newV[i].key;
        v[i].value = newV[i].value;
      }
    }

    {
      K newK[kHighSize];
      KeyValuePair<K, V> newV[kHighSize];

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        newK[i]       = k[i + kLowSize];
        newV[i].key   = v[i + kLowSize].key;
        newV[i].value = v[i + kLowSize].value;
      }

      constexpr bool kHighIsPowerOf2 = utils::isPowerOf2(kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kHighIsPowerOf2 = utils::isPowerOf2(kHighSize);
      BitonicMergeStepKVP<K,
                          V,
                          kHighSize,
                          Dir,
                          Comp,
                          false,  // high
                          kHighIsPowerOf2>::merge(newK, newV);

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        k[i + kLowSize]       = newK[i];
        v[i + kLowSize].key   = newV[i].key;
        v[i + kLowSize].value = newV[i].value;
      }
    }
  }
};

// High recursion
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicMergeStepKVP<K, V, N, Dir, Comp, false, false> {
  static inline __device__ void merge(K k[N], KeyValuePair<K, V> v[N])
  {
    static_assert(!utils::isPowerOf2(N), "must be non-power-of-2");
    static_assert(N >= 3, "must be N >= 3");

    constexpr int kNextHighestPowerOf2 = utils::nextHighestPowerOf2(N);

#pragma unroll
    for (int i = 0; i < N - kNextHighestPowerOf2 / 2; ++i) {
      K& ka                  = k[i];
      KeyValuePair<K, V>& va = v[i];

      K& kb                  = k[i + kNextHighestPowerOf2 / 2];
      KeyValuePair<K, V>& vb = v[i + kNextHighestPowerOf2 / 2];

      bool s = Dir ? Comp::gt(ka, kb) : Comp::lt(ka, kb);
      swap(s, ka, kb);
      swap(s, va.key, vb.key);
      swap(s, va.value, vb.value);
    }

    constexpr int kLowSize  = kNextHighestPowerOf2 / 2;
    constexpr int kHighSize = N - kNextHighestPowerOf2 / 2;
    {
      K newK[kLowSize];
      KeyValuePair<K, V> newV[kLowSize];

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        newK[i]       = k[i];
        newV[i].key   = v[i].key;
        newV[i].value = v[i].value;
      }

      constexpr bool kLowIsPowerOf2 = utils::isPowerOf2(kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kLowIsPowerOf2 = utils::isPowerOf2(kLowSize);
      BitonicMergeStepKVP<K,
                          V,
                          kLowSize,
                          Dir,
                          Comp,
                          true,  // low
                          kLowIsPowerOf2>::merge(newK, newV);

#pragma unroll
      for (int i = 0; i < kLowSize; ++i) {
        k[i]       = newK[i];
        v[i].key   = newV[i].key;
        v[i].value = newV[i].value;
      }
    }

    {
      K newK[kHighSize];
      KeyValuePair<K, V> newV[kHighSize];

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        newK[i]       = k[i + kLowSize];
        newV[i].key   = v[i + kLowSize].key;
        newV[i].value = v[i + kLowSize].value;
      }

      constexpr bool kHighIsPowerOf2 = utils::isPowerOf2(N - kNextHighestPowerOf2 / 2);
      // FIXME: compiler doesn't like this expression? compiler bug?
      //      constexpr bool kHighIsPowerOf2 = utils::isPowerOf2(kHighSize);
      BitonicMergeStepKVP<K,
                          V,
                          kHighSize,
                          Dir,
                          Comp,
                          false,  // high
                          kHighIsPowerOf2>::merge(newK, newV);

#pragma unroll
      for (int i = 0; i < kHighSize; ++i) {
        k[i + kLowSize]       = newK[i];
        v[i + kLowSize].key   = newV[i].key;
        v[i + kLowSize].value = newV[i].value;
      }
    }
  }
};

/// Merges two sets of registers across the warp of any size;
/// i.e., merges a sorted k/v list of size WarpSize * N1 with a
/// sorted k/v list of size WarpSize * N2, where N1 and N2 are any
/// value >= 1
template <typename K, typename V, int N1, int N2, bool Dir, typename Comp, bool FullMerge = true>
inline __device__ void warpMergeAnyRegistersKVP(K k1[N1],
                                                KeyValuePair<K, V> v1[N1],
                                                K k2[N2],
                                                KeyValuePair<K, V> v2[N2])
{
  constexpr int kSmallestN = N1 < N2 ? N1 : N2;

#pragma unroll
  for (int i = 0; i < kSmallestN; ++i) {
    K& ka                  = k1[N1 - 1 - i];
    KeyValuePair<K, V>& va = v1[N1 - 1 - i];

    K& kb                  = k2[i];
    KeyValuePair<K, V>& vb = v2[i];

    K otherKa;
    KeyValuePair<K, V> otherVa;

    if (FullMerge) {
      // We need the other values
      otherKa    = shfl_xor(ka, WarpSize - 1);
      K otherVak = shfl_xor(va.key, WarpSize - 1);
      V otherVav = shfl_xor(va.value, WarpSize - 1);
      otherVa    = KeyValuePair(otherVak, otherVav);
    }

    K otherKb  = shfl_xor(kb, WarpSize - 1);
    K otherVbk = shfl_xor(vb.key, WarpSize - 1);
    V otherVbv = shfl_xor(vb.value, WarpSize - 1);

    // ka is always first in the list, so we needn't use our lane
    // in this comparison
    bool swapa = Dir ? Comp::gt(ka, otherKb) : Comp::lt(ka, otherKb);
    assign(swapa, ka, otherKb);
    assign(swapa, va.key, otherVbk);
    assign(swapa, va.value, otherVbv);

    // kb is always second in the list, so we needn't use our lane
    // in this comparison
    if (FullMerge) {
      bool swapb = Dir ? Comp::lt(kb, otherKa) : Comp::gt(kb, otherKa);
      assign(swapb, kb, otherKa);
      assign(swapb, vb.key, otherVa.key);
      assign(swapb, vb.value, otherVa.value);

    } else {
      // We don't care about updating elements in the second list
    }
  }

  BitonicMergeStepKVP<K, V, N1, Dir, Comp, true, utils::isPowerOf2(N1)>::merge(k1, v1);
  if (FullMerge) {
    // Only if we care about N2 do we need to bother merging it fully
    BitonicMergeStepKVP<K, V, N2, Dir, Comp, false, utils::isPowerOf2(N2)>::merge(k2, v2);
  }
}

// Recursive template that uses the above bitonic merge to perform a
// bitonic sort
template <typename K, typename V, int N, bool Dir, typename Comp>
struct BitonicSortStepKVP {
  static inline __device__ void sort(K k[N], KeyValuePair<K, V> v[N])
  {
    static_assert(N > 1, "did not hit specialized case");

    // Sort recursively
    constexpr int kSizeA = N / 2;
    constexpr int kSizeB = N - kSizeA;

    K aK[kSizeA];
    KeyValuePair<K, V> aV[kSizeA];

#pragma unroll
    for (int i = 0; i < kSizeA; ++i) {
      aK[i]       = k[i];
      aV[i].key   = v[i].key;
      aV[i].value = v[i].value;
    }

    BitonicSortStepKVP<K, V, kSizeA, Dir, Comp>::sort(aK, aV);

    K bK[kSizeB];
    KeyValuePair<K, V> bV[kSizeB];

#pragma unroll
    for (int i = 0; i < kSizeB; ++i) {
      bK[i]       = k[i + kSizeA];
      bV[i].key   = v[i + kSizeA].key;
      bV[i].value = v[i + kSizeA].value;
    }

    BitonicSortStepKVP<K, V, kSizeB, Dir, Comp>::sort(bK, bV);

    // Merge halves
    warpMergeAnyRegistersKVP<K, V, kSizeA, kSizeB, Dir, Comp>(aK, aV, bK, bV);

#pragma unroll
    for (int i = 0; i < kSizeA; ++i) {
      k[i]       = aK[i];
      v[i].key   = aV[i].key;
      v[i].value = aV[i].value;
    }

#pragma unroll
    for (int i = 0; i < kSizeB; ++i) {
      k[i + kSizeA]       = bK[i];
      v[i + kSizeA].key   = bV[i].key;
      v[i + kSizeA].value = bV[i].value;
    }
  }
};

// Single warp (N == 1) sorting specialization
template <typename K, typename V, bool Dir, typename Comp>
struct BitonicSortStepKVP<K, V, 1, Dir, Comp> {
  static inline __device__ void sort(K k[1], KeyValuePair<K, V> v[1])
  {
    // Update this code if this changes
    // should go from 1 -> WarpSize in multiples of 2
    static_assert(WarpSize == 32, "unexpected warp size");

    warpBitonicMergeLE16KVP<K, V, 1, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16KVP<K, V, 2, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16KVP<K, V, 4, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16KVP<K, V, 8, Dir, Comp, false>(k[0], v[0]);
    warpBitonicMergeLE16KVP<K, V, 16, Dir, Comp, false>(k[0], v[0]);
  }
};

/// Sort a list of WarpSize * N elements in registers, where N is an
/// arbitrary >= 1
template <typename K, typename V, int N, bool Dir, typename Comp>
inline __device__ void warpSortAnyRegistersKVP(K k[N], KeyValuePair<K, V> v[N])
{
  BitonicSortStepKVP<K, V, N, Dir, Comp>::sort(k, v);
}

}  // namespace raft::spatial::knn::detail::faiss_select
