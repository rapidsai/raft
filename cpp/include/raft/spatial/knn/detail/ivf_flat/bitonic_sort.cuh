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
 */
template <int Size, bool Ascending>
struct bitonic_merge {
  static_assert(isPo2(Size));

  /** How many contiguous elements are processed by one thread. */
  static constexpr int kArrLen = Size / WarpSize;
  static constexpr int kStride = kArrLen / 2;

  template <bool Fits, typename Dummy>
  using when_fits_in_warp =
    std::enable_if_t<(Fits == (Size <= WarpSize)) && std::is_same_v<Dummy, Dummy>, void>;

  template <typename KeyT, typename... PayloadTs>
  static __device__ auto run(bool reverse, KeyT* keys, PayloadTs*... payloads)
    -> when_fits_in_warp<false, KeyT>
  {
    for (int i = 0; i < kStride; ++i) {
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

    bitonic_merge<Size / 2, Ascending>::run(reverse, keys, payloads...);
    bitonic_merge<Size / 2, Ascending>::run(reverse, keys + kStride, (payloads + kStride)...);
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ auto run(bool reverse, KeyT* keys, PayloadTs*... payloads)
    -> when_fits_in_warp<true, KeyT>
  {
    const int lane = threadIdx.x % Size;
    for (int stride = Size / 2; stride > 0; stride /= 2) {
      bool is_second = lane & stride;
      KeyT& key      = *keys;
      KeyT other     = shfl_xor(key, stride, Size);

      bool asc       = Ascending != reverse;
      bool do_assign = key != other && ((key > other) == (asc != is_second));
      // Normally, we expect `payloads` to be the array of indices from 0 to len;
      // in that case, the construct below makes the sorting stable.
      if constexpr (sizeof...(payloads) > 0) {
        auto payload_this = *helpers::first(payloads...);
        auto payload_that = shfl_xor(payload_this, stride, Size);
        if (key == other) { do_assign = reverse != ((payload_this > payload_that) != is_second); }
      }

      helpers::assign(do_assign, keys, other);
      // NB: don't put shfl_xor in a conditional; it must be called by all threads in a warp.
      (helpers::assign(do_assign, payloads, shfl_xor(*payloads, stride, Size)), ...);
    }
  }

  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(KeyT* keys, PayloadTs*... payloads)
  {
    return run(false, keys, payloads...);
  }
};

/**
 * Bitonic sort at the warp level.
 *
 * @tparam Size is the number of elements (must be power of two).
 * @tparam Ascending is the resulting order (true: ascending, false: descending).
 */
template <int Size, bool Ascending>
struct bitonic_sort {
  static_assert(isPo2(Size));

  static constexpr int kSize2 = Size / 2;

  template <typename KeyT, typename... PayloadTs>
  static __device__ __forceinline__ void run(bool reverse, KeyT* keys, PayloadTs*... payloads)
  {
    if constexpr (Size > 2) {
      // NB: the `reverse` expression here is always `0` (false) when `Size > WarpSize`
      bitonic_sort<kSize2, Ascending>::run(laneId() & kSize2, keys, payloads...);
    }
    if constexpr (Size > WarpSize) {
      // NB: this part is executed only if the size of the input arrays is larger than the warp.
      constexpr int kShift = kSize2 / WarpSize;
      bitonic_sort<kSize2, Ascending>::run(true, keys + kShift, (payloads + kShift)...);
    }
    bitonic_merge<Size, Ascending>::run(reverse, keys, payloads...);
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
  static __device__ __forceinline__ void run(KeyT* keys, PayloadTs*... payloads)
  {
    return run(false, keys, payloads...);
  }
};

}  // namespace raft::spatial::knn::detail::ivf_flat
