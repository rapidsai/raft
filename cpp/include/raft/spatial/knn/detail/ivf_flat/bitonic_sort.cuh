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

#include <raft/cuda_utils.cuh>

namespace raft::spatial::knn::detail::ivf_flat {

namespace helpers {

template <typename T>
__device__ __forceinline__ void swap(T& x, T& y)
{
  T t = x;
  x   = y;
  y   = t;
}

template <typename T>
__device__ __forceinline__ void assign(bool cond, T* ptr, T x)
{
  if (cond) { *ptr = x; }
}

template <typename T>
__device__ __forceinline__ void assign(bool cond, T& ptr, T x)
{
  if (cond) { ptr = x; }
}

template <typename T, typename... Ts>
__device__ __forceinline__ auto first(T x, Ts... xs) -> T
{
  return x;
}

}  // namespace helpers

/**
 * Bitonic merge at the warp level.
 *
 * @tparam Size is the number of elements (must be power of two).
 * @tparam Ascending is the resulting order (true: ascending, false: descending).
 * @tparam Cross whether the right half of the input is sorted in the opposite direction.
 */
template <int Size, bool Ascending, bool Cross = true>
struct bitonic_merge {
  static_assert(isPo2(Size));

  /** How many contiguous elements are processed by one thread. */
  static constexpr int kArrLen = Size / WarpSize;
  static constexpr int kStride = kArrLen / 2;

  template <bool Fits, typename Dummy>
  using when_fits_in_warp =
    std::enable_if_t<(Fits == (Size <= WarpSize)) && std::is_same_v<Dummy, Dummy>, void>;

  template <typename KeyT, typename... PayloadTs>
  static __device__ auto run(bool reverse,
                             KeyT* __restrict__ keys,
                             PayloadTs* __restrict__... payloads) -> when_fits_in_warp<false, KeyT>
  {
    static_assert(Cross, "Straight merging is not implemented for Size > WarpSize.");
    for (int i = 0; i < kStride; i++) {
      const int other_i = i + kStride;
      KeyT& key         = keys[i];
      KeyT& other       = keys[other_i];
      bool do_swap      = Ascending != reverse ? key > other : key < other;
      // Normally, we expect `payloads` to be the array of indices from 0 to len;
      // in that case, the construct below makes the sorting stable.
      if constexpr (sizeof...(payloads) > 0) {
        if (key == other) {
          do_swap =
            reverse != (helpers::first(payloads...)[i] > helpers::first(payloads...)[other_i]);
        }
      }
      if (do_swap) {
        helpers::swap(key, other);
        (helpers::swap(payloads[i], payloads[other_i]), ...);
      }
    }

    bitonic_merge<Size / 2, Ascending, true>::run(reverse, keys, payloads...);
    bitonic_merge<Size / 2, Ascending, true>::run(reverse, keys + kStride, (payloads + kStride)...);
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ auto run(bool reverse,
                             KeyT& __restrict__ key,
                             PayloadTs& __restrict__... payload) -> when_fits_in_warp<true, KeyT>
  {
    const int lane = threadIdx.x % Size;
    int stride     = Size / 2;
    if constexpr (!Cross) {
      bool is_second = lane & stride;
      KeyT other     = shfl(key, Size - lane - 1, Size);

      bool asc       = Ascending != reverse;
      bool do_assign = key != other && ((key > other) == (asc != is_second));
      // Normally, we expect `payloads` to be the array of indices from 0 to len;
      // in that case, the construct below makes the sorting stable.
      if constexpr (sizeof...(payload) > 0) {
        auto payload_this = helpers::first(payload...);
        auto payload_that = shfl(payload_this, Size - lane - 1, Size);
        if (key == other) { do_assign = reverse != ((payload_this > payload_that) != is_second); }
      }

      helpers::assign(do_assign, key, other);
      // NB: don't put shfl_xor in a conditional; it must be called by all threads in a warp.
      (helpers::assign(do_assign, payload, shfl(payload, Size - lane - 1, Size)), ...);

      stride /= 2;
    }
    for (; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      KeyT other     = shfl_xor(key, stride, Size);

      bool asc       = Ascending != reverse;
      bool do_assign = key != other && ((key > other) == (asc != is_second));
      // Normally, we expect `payloads` to be the array of indices from 0 to len;
      // in that case, the construct below makes the sorting stable.
      if constexpr (sizeof...(payload) > 0) {
        auto payload_this = helpers::first(payload...);
        auto payload_that = shfl_xor(payload_this, stride, Size);
        if (key == other) { do_assign = reverse != ((payload_this > payload_that) != is_second); }
      }

      helpers::assign(do_assign, key, other);
      // NB: don't put shfl_xor in a conditional; it must be called by all threads in a warp.
      (helpers::assign(do_assign, payload, shfl_xor(payload, stride, Size)), ...);
    }
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ auto run(bool reverse,
                             KeyT* __restrict__ keys,
                             PayloadTs* __restrict__... payloads) -> when_fits_in_warp<true, KeyT>
  {
    return run(reverse, *keys, *payloads...);
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ void run(KeyT* __restrict__ keys, PayloadTs* __restrict__... payloads)
  {
    return run(false, keys, payloads...);
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ auto run(KeyT& __restrict__ key, PayloadTs& __restrict__... payload)
    -> when_fits_in_warp<true, KeyT>
  {
    return run(false, key, payload...);
  }
};

/**
 * Bitonic sort at the warp level.
 *
 * @tparam Size is the number of elements (must be power of two).
 * @tparam Ascending is the resulting order (true: ascending, false: descending).
 */
template <int Size, bool Ascending, int AlreadySortedSize = 1>
struct bitonic_sort {
  static_assert(isPo2(Size));
  static_assert(isPo2(AlreadySortedSize));
  static_assert(isPo2(Size >= AlreadySortedSize));

  template <typename KeyT, typename... PayloadTs>
  static __device__ void run(bool reverse,
                             KeyT* __restrict__ keys,
                             PayloadTs* __restrict__... payloads)
  {
    constexpr int kSize2 = Size / 2;
    if constexpr (kSize2 > AlreadySortedSize) {
      // NB: the `reverse` expression here is always `0` (false) when `Size > WarpSize`
      bitonic_sort<kSize2, Ascending, AlreadySortedSize>::run(laneId() & kSize2, keys, payloads...);
      if constexpr (Size > WarpSize) {
        // NB: this part is executed only if the size of the input arrays is larger than the warp.
        constexpr int kShift = kSize2 / WarpSize;
        bitonic_sort<kSize2, Ascending, AlreadySortedSize>::run(
          true, keys + kShift, (payloads + kShift)...);
      }
    }
    bitonic_merge<Size, Ascending, (kSize2 > AlreadySortedSize)>::run(reverse, keys, payloads...);
  }

  /**
   * Execute the sort.
   *
   * @param keys
   *   is a device pointer to a contiguous array of keys, unique per thread;
   * @param payloads
   *   are zero or more associated arrays of the same size as keys, which are sorted together with
   *   the keys.
   */
  template <typename KeyT, typename... PayloadTs>
  static __device__ void run(KeyT* __restrict__ keys, PayloadTs* __restrict__... payloads)
  {
    return run(false, keys, payloads...);
  }
};

template <bool Ascending, bool Cross>
struct bitonic_merge<1, Ascending, Cross> {
  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(bool reverse,
                                             KeyT* __restrict__ keys,
                                             PayloadTs* __restrict__... payloads)
  {
  }
  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(bool reverse,
                                             KeyT& __restrict__ keys,
                                             PayloadTs& __restrict__... payloads)
  {
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(KeyT* __restrict__ keys,
                                             PayloadTs* __restrict__... payloads)
  {
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(KeyT& __restrict__ keys,
                                             PayloadTs& __restrict__... payloads)
  {
  }
};

template <int Size, bool Ascending>
struct bitonic_sort<Size, Ascending, Size> {
  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(bool reverse,
                                             KeyT* __restrict__ keys,
                                             PayloadTs* __restrict__... payloads)
  {
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(KeyT* __restrict__ keys,
                                             PayloadTs* __restrict__... payloads)
  {
  }
};

}  // namespace raft::spatial::knn::detail::ivf_flat
