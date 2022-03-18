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

#include "bitonic_sort.cuh"

#include <raft/cuda_utils.cuh>
#include <raft/pow2_utils.cuh>

#include <algorithm>
#include <functional>
#include <type_traits>

/*
  Three APIs of different scope are provided:
    1. host function: warp_sort_topk()
    2. block-wide API: class WarpSortBlockWide
    3. warp-wide API: class WarpSelect and class WarpBitonic


  1. warp_sort_topk()
    Like CUB functions, it should be called twice.
    First for getting required buffer size, and a second for the real top-k computation.
    For the first call, buf==nullptr should be passed, and required buffer
    size is returned as parameter buf_size.
    For the second call, pass allocated buffer of required size.

    Example:
      void* buf = nullptr;
      size_t buf_size;
      warp_sort_topk(nullptr, buf_size, ...);  // will set buf_size
      cudaMalloc(&buf, buf_size);
      warp_sort_topk(buf, buf_size, ...);


  2. class WarpSortBlockWide
    It can be regarded as a fixed size priority queue for a thread block,
    although the API is not typical.
    class WarpSelect and WarpBitonic can be used to instantiate WarpSortBlockWide.

    It uses dynamic shared memory as intermediate buffer.
    So the required shared memory size should be calculated using
    calc_smem_size_for_block_wide() and passed as the 3rd kernel launch parameter.

    Two overloaded add() functions can be used to add items to the queue.
    One is add(const T* in, IdxT start, IdxT end) and it adds a range of items,
    namely [start, end) of in. The idx is inferred from start.
    This function should be called only once to add all items, and should not be
    used together with the second form of add().
    The second one is add(T val, IdxT idx), and it adds only one item pair.
    Note that the range [start, end) is for the whole block of threads, that is,
    each thread in the same block should get the same start/end.
    In contrast, the parameters of the second form are for only one thread,
    so each thread must get different val/idx.

    After adding is finished, function done() should be called. And finally,
    dump() is used to get the top-k result.

    Example:
      __global__ void kernel() {
        WarpSortBlockWide<WarpBitonic, ...> queue(...);

        // way 1, [0, len) is same for the whole block
        queue.add(in, 0, len);
        // way 2, each thread gets its own val/idx pair
        for (IdxT i = threadIdx.x; i < len, i += blockDim.x) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        queue.dump(out, out_idx);
     }

     int smem_size = calc_smem_size_for_block_wide<T>(...);
     kernel<<grid_dim, block_dim, smem_size>>>();


  3. class WarpSelect and class WarpBitonic
    These two classes can be regarded as fixed sized priority queue for a warp.
    Usage is similar to class WarpSortBlockWide.
    Two types of add() functions are provided, and also note that [start, end) is
    for a whole warp, while val/idx is for a thread.
    No shared memory is needed.

    The host function uses a heuristic to choose between these two classes for sorting,
    WarpBitonic being chosen when the number of inputs per warp is somewhat small
    (see the usage of LaunchThreshold<WarpBitonic>::len_factor_for_choosing).

    Example:
      __global__ void kernel() {
        WarpBitonic<...> queue(...);
        int warp_id = threadIdx.x / WarpSize;
        int lane_id = threadIdx.x % WarpSize;

        // way 1, [0, len) is same for the whole warp
        queue.add(in, 0, len);
        // way 2, each thread gets its own val/idx pair
        for (IdxT i = lane_id; i < len, i += WarpSize) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        // each warp outputs to a different offset
        queue.dump(out+ warp_id * k * sizeof(T), out_idx+ warp_id * k * sizeof(IdxT));
      }
 */

namespace raft::spatial::knn::detail::ivf_flat {

static constexpr int kMaxCapacity = 256;

namespace {

template <bool greater, typename T>
__device__ inline bool is_greater_than(T val, T baseline)
{
  if constexpr (greater) { return val > baseline; }
  if constexpr (!greater) { return val < baseline; }
}

int calc_capacity(int k)
{
  int capacity = isPo2(k) ? k : (1 << (log2(k) + 1));
  if (capacity < WarpSize) { capacity = WarpSize; }  // TODO: remove this to allow small sizes.
  return capacity;
}

}  // namespace

/**
 * A fixed-size warp-level priority queue.
 * By feeding the data through this queue, you get the `k <= capacity`
 * smallest/greatest values in the data.
 *
 * @tparam capacity
 *   maximum number of elements in the queue.
 * @tparam greater
 *   which comparison to use: `true` means `>`, `false` means `<`.
 * @tparam T
 *   the type of keys (what is being compared)
 * @tparam IdxT
 *   the type of payload (normally, indices of elements), i.e.
 *   the content sorted alongside the keys.
 */
template <int capacity, bool greater, typename T, typename IdxT>
class WarpSort {
  static_assert(isPo2(capacity));

 public:
  __device__ WarpSort(IdxT k, T dummy) : k_(k), dummy_(dummy)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_arr_[i] = dummy_;
    }
  }

  /**
   * Load k values from the pointers at the given position, and merge them in the storage.
   */
  __device__ void load_sorted(const T* in, const IdxT* in_idx, IdxT start)
  {
    IdxT idx = start + kWarpWidth - 1 - Pow2<kWarpWidth>::mod(laneId());
#pragma unroll
    for (int i = kMaxArrLen - 1; i >= 0; --i, idx += kWarpWidth) {
      if (idx < start + k_) {
        T t = in[idx];
        if (is_greater_than<greater>(t, val_arr_[i])) {
          val_arr_[i] = t;
          idx_arr_[i] = in_idx[idx];
        }
      }
    }
    ivf_flat::bitonic_merge<capacity, !greater>::run(val_arr_, idx_arr_);
  }

  __device__ void dump(T* out, IdxT* out_idx) const
  {
    IdxT idx = Pow2<kWarpWidth>::mod(laneId());
#pragma unroll kMaxArrLen
    for (int i = 0; i < kMaxArrLen && idx < k_; i++, idx += kWarpWidth) {
      out[idx]     = val_arr_[i];
      out_idx[idx] = idx_arr_[i];
    }
  }

  // TODO: do all merging in the bitonic_sort.cuh
  // /**
  //  * When capacity < WarpSize, merges sorted queues within the warp.
  //  * As a result, the top k selected values are placed in the first queue in the group
  //  * (i.e. within the first kWarpWidth values, since k <= capacity == kWarpWidth).
  //  *
  //  * It does nothing when capacity >= WarpSize
  //  */
  // __device__ __forceinline__ void merge_within_warp()
  // {
  //   if constexpr (kWarpWidth < WarpSize) {
  //     ivf_flat::bitonic_sort<WarpSize, !greater, kWarpWidth>::run(val_arr_, idx_arr_);
  //   }
  // }

 protected:
  static constexpr int kWarpWidth = std::min<int>(capacity, WarpSize);
  static constexpr int kMaxArrLen = capacity / kWarpWidth;

  const IdxT k_;
  const T dummy_;
  T val_arr_[kMaxArrLen];
  IdxT idx_arr_[kMaxArrLen];
};

template <int capacity, bool greater, typename T, typename IdxT>
class WarpSelect : public WarpSort<capacity, greater, T, IdxT> {
  static_assert(capacity >= WarpSize);

 public:
  __device__ WarpSelect(int k, T dummy)
    : WarpSort<capacity, greater, T, IdxT>(k, dummy), buf_len_(0), k_th_(dummy)
  {
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = dummy_;
    }
  }

  __device__ void add(const T* in, IdxT start, IdxT end)
  {
    const IdxT end_for_fullwarp = Pow2<WarpSize>::roundUp(end - start) + start;
    for (IdxT i = start + laneId(); i < end_for_fullwarp; i += WarpSize) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i);
    }
  }

  __device__ void add(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    const IdxT end_for_fullwarp = Pow2<WarpSize>::roundUp(end - start) + start;
    for (IdxT i = start + laneId(); i < end_for_fullwarp; i += WarpSize) {
      T val    = (i < end) ? in[i] : dummy_;
      IdxT idx = (i < end) ? in_idx[i] : std::numeric_limits<IdxT>::max();
      add(val, idx);
    }
  }

  __device__ void add(T val, IdxT idx)
  {
    // comparing for k_th should reduce the total amount of updates:
    // `false` means the input value is surely not in the top-k values.
    if (is_greater_than<greater>(val, k_th_)) {
      // NB: the loop is used here to ensure the constant indexing,
      //     to not force the buffers spill into the local memory.
#pragma unroll
      for (int i = 0; i < kMaxBufLen; i++) {
        if (i == buf_len_) {
          val_buf_[i] = val;
          idx_buf_[i] = idx;
        }
      }
      ++buf_len_;
    }
    if (any(buf_len_ == kMaxBufLen)) { merge_buf_(); }
  }

  __device__ void done()
  {
    if (any(buf_len_ != 0)) { merge_buf_(); }
  }

 private:
  __device__ void set_k_th_()
  {
    // it's the best we can do, should use "val_arr_[k_th_row_]"
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], Pow2<WarpSize>::mod(k_ - 1));
  }

  __device__ void merge_buf_()
  {
    ivf_flat::bitonic_sort<kMaxBufLen * WarpSize, greater>::run(val_buf_, idx_buf_);
    //         // merge the tails of both value arrays, which means possibly updating
    //     // smallest of the val_arr_ with the largest of the val_buf_, because
    //     // they are sorted in the opposite directions.
    //     // Essentially, this is the first step of bitonic_merge<capacity * 2, !greater>
    // #pragma unroll
    //     for (int i = std::min(kMaxArrLen, kMaxBufLen); i > 0; i--) {
    //       T& val = val_arr_[kMaxArrLen - i];
    //       T buf  = val_buf_[kMaxBufLen - i];
    //       if (is_greater_than<greater>(buf, val)) {
    //         val                      = buf;
    //         idx_arr_[kMaxArrLen - i] = idx_buf_[kMaxBufLen - i];
    //       }
    //     }

    if (kMaxArrLen > kMaxBufLen) {
      for (int i = 0; i < kMaxBufLen; ++i) {
        T& val = val_arr_[kMaxArrLen - kMaxBufLen + i];
        T& buf = val_buf_[i];
        if (is_greater_than<greater>(buf, val)) {
          val                                   = buf;
          idx_arr_[kMaxArrLen - kMaxBufLen + i] = idx_buf_[i];
        }
      }
    } else if (kMaxArrLen < kMaxBufLen) {
      for (int i = 0; i < kMaxArrLen; ++i) {
        T& val = val_arr_[i];
        T& buf = val_buf_[kMaxBufLen - kMaxArrLen + i];
        if (is_greater_than<greater>(buf, val)) {
          val         = buf;
          idx_arr_[i] = idx_buf_[kMaxBufLen - kMaxArrLen + i];
        }
      }
    } else {
      for (int i = 0; i < kMaxArrLen; ++i) {
        if (is_greater_than<greater>(val_buf_[i], val_arr_[i])) {
          val_arr_[i] = val_buf_[i];
          idx_arr_[i] = idx_buf_[i];
        }
      }
    }

    ivf_flat::bitonic_merge<capacity, !greater>::run(val_arr_, idx_arr_);

    buf_len_ = 0;
    set_k_th_();  // contains warp sync
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = dummy_;
    }
  }

  using WarpSort<capacity, greater, T, IdxT>::kMaxArrLen;
  using WarpSort<capacity, greater, T, IdxT>::val_arr_;
  using WarpSort<capacity, greater, T, IdxT>::idx_arr_;
  using WarpSort<capacity, greater, T, IdxT>::k_;
  using WarpSort<capacity, greater, T, IdxT>::dummy_;

  static constexpr int kMaxBufLen = (capacity <= 64) ? 2 : 4;

  T val_buf_[kMaxBufLen];
  IdxT idx_buf_[kMaxBufLen];
  int buf_len_;

  T k_th_;
};

template <int capacity, bool greater, typename T, typename IdxT>
class WarpBitonic : public WarpSort<capacity, greater, T, IdxT> {
  static_assert(capacity >= WarpSize);

 public:
  __device__ WarpBitonic(int k, T dummy)
    : WarpSort<capacity, greater, T, IdxT>(k, dummy), buf_len_(0)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_buf_[i] = dummy_;
    }
  }

  __device__ void add(const T* in, IdxT start, IdxT end)
  {
    add_first_(in, start, end);
    start += capacity;
    while (start < end) {
      add_extra_(in, start, end);
      merge_();
      start += capacity;
    }
  }

  __device__ void add(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    add_first_(in, in_idx, start, end);
    start += capacity;
    while (start < end) {
      add_extra_(in, in_idx, start, end);
      merge_();
      start += capacity;
    }
  }

  __device__ void add(T val, IdxT idx)
  {
    // NB: the loop is used here to ensure the constant indexing,
    //     to not force the buffers spill into the local memory.
#pragma unroll
    for (int i = 0; i < kMaxArrLen; ++i) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }

    ++buf_len_;
    if (buf_len_ == kMaxArrLen) {
      ivf_flat::bitonic_sort<capacity, greater>::run(val_buf_, idx_buf_);
      merge_();
#pragma unroll
      for (int i = 0; i < kMaxArrLen; i++) {
        val_buf_[i] = dummy_;
      }
      buf_len_ = 0;
    }
  }

  __device__ void done()
  {
    if (buf_len_ != 0) {
      ivf_flat::bitonic_sort<capacity, greater>::run(val_buf_, idx_buf_);
      merge_();
    }
  }

 private:
  // Fill in the primary val_arr_/idx_arr_
  __device__ void add_first_(const T* in, IdxT start, IdxT end)
  {
    IdxT idx = start + laneId();
    for (int i = 0; i < kMaxArrLen; ++i, idx += WarpSize) {
      if (idx < end) {
        val_arr_[i] = in[idx];
        idx_arr_[i] = idx;
      }
    }
    ivf_flat::bitonic_sort<capacity, !greater>::run(val_arr_, idx_arr_);
  }

  // Fill in the primary val_arr_/idx_arr_
  __device__ void add_first_(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    IdxT idx = start + laneId();
    for (int i = 0; i < kMaxArrLen; ++i, idx += WarpSize) {
      if (idx < end) {
        val_arr_[i] = in[idx];
        idx_arr_[i] = in_idx[idx];
      }
    }
    ivf_flat::bitonic_sort<capacity, !greater>::run(val_arr_, idx_arr_);
  }

  // Fill in the secondary val_buf_/idx_buf_
  __device__ void add_extra_(const T* in, IdxT start, IdxT end)
  {
    IdxT idx = start + laneId();
    for (int i = 0; i < kMaxArrLen; ++i, idx += WarpSize) {
      val_buf_[i] = (idx < end) ? in[idx] : dummy_;
      idx_buf_[i] = idx;
    }
    ivf_flat::bitonic_sort<capacity, greater>::run(val_buf_, idx_buf_);
  }

  // Fill in the secondary val_buf_/idx_buf_
  __device__ void add_extra_(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    IdxT idx = start + laneId();
    for (int i = 0; i < kMaxArrLen; ++i, idx += WarpSize) {
      val_buf_[i] = (idx < end) ? in[idx] : dummy_;
      idx_buf_[i] = (idx < end) ? in_idx[idx] : std::numeric_limits<IdxT>::max();
    }
    ivf_flat::bitonic_sort<capacity, greater>::run(val_buf_, idx_buf_);
  }

  __device__ void merge_()
  {
    for (int i = 0; i < kMaxArrLen; ++i) {
      if (is_greater_than<greater>(val_buf_[i], val_arr_[i])) {
        val_arr_[i] = val_buf_[i];
        idx_arr_[i] = idx_buf_[i];
      }
    }
    ivf_flat::bitonic_merge<capacity, !greater>::run(val_arr_, idx_arr_);
  }

  using WarpSort<capacity, greater, T, IdxT>::kMaxArrLen;
  using WarpSort<capacity, greater, T, IdxT>::val_arr_;
  using WarpSort<capacity, greater, T, IdxT>::idx_arr_;
  using WarpSort<capacity, greater, T, IdxT>::k_;
  using WarpSort<capacity, greater, T, IdxT>::dummy_;

  T val_buf_[kMaxArrLen];
  IdxT idx_buf_[kMaxArrLen];
  int buf_len_;
};

/**
 * This one is used for the second pass only:
 *   if the first pass happens in multiple blocks, the output consists of a series
 *   of sorted arrays, length `k` each.
 *   Under this assumption, we can use load_sorted to just do the merging, rather than
 *   the full sort.
 */
template <int capacity, bool greater, typename T, typename IdxT>
class WarpMerge : public WarpSort<capacity, greater, T, IdxT> {
 public:
  __device__ WarpMerge(int k, T dummy) : WarpSort<capacity, greater, T, IdxT>(k, dummy) {}

  // NB: the input is already sorted, because it's the second pass.
  __device__ void add(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    IdxT idx       = start + Pow2<kWarpWidth>::mod(laneId());
    IdxT first_end = (start + k_ < end) ? (start + k_) : end;
    for (int i = 0; i < kMaxArrLen; ++i, idx += kWarpWidth) {
      if (idx < first_end) {
        val_arr_[i] = in[idx];
        idx_arr_[i] = in_idx[idx];
      }
    }

    for (start += k_; start < end; start += k_) {
      load_sorted(in, in_idx, start);
    }
  }

  __device__ void done() {}

 private:
  using WarpSort<capacity, greater, T, IdxT>::kWarpWidth;
  using WarpSort<capacity, greater, T, IdxT>::kMaxArrLen;
  using WarpSort<capacity, greater, T, IdxT>::val_arr_;
  using WarpSort<capacity, greater, T, IdxT>::idx_arr_;
  using WarpSort<capacity, greater, T, IdxT>::k_;
  using WarpSort<capacity, greater, T, IdxT>::dummy_;
};

template <typename T, typename IdxT>
int calc_smem_size_for_block_wide(int num_of_warp, IdxT k)
{
  return Pow2<256>::roundUp(num_of_warp / 2 * sizeof(T) * k) + num_of_warp / 2 * sizeof(IdxT) * k;
}

template <template <int, bool, typename, typename> class WarpSortWarpWide,
          int capacity,
          bool greater,
          typename T,
          typename IdxT>
class WarpSortBlockWide {
 public:
  __device__ WarpSortBlockWide(int k, T dummy, void* smem_buf)
    : queue_(k, dummy), k_(k), dummy_(dummy)
  {
    val_smem_             = static_cast<T*>(smem_buf);
    const int num_of_warp = blockDim.x / WarpSize;
    idx_smem_             = reinterpret_cast<IdxT*>(reinterpret_cast<char*>(smem_buf) +
                                        Pow2<256>::roundUp(num_of_warp / 2 * sizeof(T) * k_));
  }

  __device__ void add(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    // static_assert(std::is_same_v<WarpSortWarpWide<capacity, greater, T, IdxT>,
    //                              WarpMerge<capacity, greater, T, IdxT>>);

    int num_of_warp   = blockDim.x / WarpSize;
    const int warp_id = threadIdx.x / WarpSize;
    IdxT len_per_warp = (end - start - 1) / num_of_warp + 1;
    len_per_warp      = ((len_per_warp - 1) / k_ + 1) * k_;

    IdxT warp_start = start + warp_id * len_per_warp;
    IdxT warp_end   = warp_start + len_per_warp;
    if (warp_end > end) { warp_end = end; }
    queue_.add(in, in_idx, warp_start, warp_end);
  }

  // can't use the form of "in + len" and let the caller pass "in" by setting it to
  // correct offset.
  // It's due to the need to fill idx.
  // so has to pass the correct offset as "start"
  __device__ void add(const T* in, IdxT start, IdxT end)
  {
    if constexpr (std::is_same_v<WarpSortWarpWide<capacity, greater, T, IdxT>,
                                 WarpSelect<capacity, greater, T, IdxT>>) {
      const IdxT end_for_fullwarp = Pow2<WarpSize>::roundUp(end - start) + start;
      for (IdxT i = start + threadIdx.x; i < end_for_fullwarp; i += blockDim.x) {
        T val = (i < end) ? in[i] : dummy_;
        queue_.add(val, i);
      }
    } else if constexpr (std::is_same_v<WarpSortWarpWide<capacity, greater, T, IdxT>,
                                        WarpBitonic<capacity, greater, T, IdxT>>) {
      int num_of_warp   = blockDim.x / WarpSize;
      const int warp_id = threadIdx.x / WarpSize;
      IdxT len_per_warp = (end - start - 1) / num_of_warp + 1;
      len_per_warp      = Pow2<WarpSize>::roundUp(len_per_warp);

      IdxT warp_start = start + warp_id * len_per_warp;
      IdxT warp_end   = warp_start + len_per_warp;
      if (warp_end > end) { warp_end = end; }
      queue_.add(in, warp_start, warp_end);
    }
  }

  __device__ void add(T val, IdxT idx) { queue_.add(val, idx); }

  /**
   * At the point of calling this function, the warp-level queues consumed all input independently.
   * The remaining work to be done is to merge them together.
   *
   * Here we tree-merge the results using the shared memory and block sync.
   */
  __device__ void done()
  {
    queue_.done();

    int num_of_warp   = blockDim.x / WarpSize;
    const int warp_id = threadIdx.x / WarpSize;

    while (num_of_warp > 1) {
      int half_num_of_warp = (num_of_warp + 1) / 2;
      if (warp_id < num_of_warp && warp_id >= half_num_of_warp) {
        int dst_warp_id = warp_id - half_num_of_warp;
        queue_.dump(val_smem_ + dst_warp_id * k_, idx_smem_ + dst_warp_id * k_);
      }
      __syncthreads();

      if (warp_id < num_of_warp / 2) { queue_.load_sorted(val_smem_, idx_smem_, warp_id * k_); }
      __syncthreads();

      num_of_warp = half_num_of_warp;
    }
  }

  __device__ void dump(T* out, IdxT* out_idx) const
  {
    if (threadIdx.x < kWarpWidth) { queue_.dump(out, out_idx); }
  }

 private:
  static constexpr int kWarpWidth = std::min<int>(capacity, WarpSize);

  WarpSortWarpWide<capacity, greater, T, IdxT> queue_;
  int k_;
  T dummy_;
  T* val_smem_;
  IdxT* idx_smem_;
};

/**
 * Uses the `WarpSortClass` to sort chunks of data within one block with no interblock
 * communication. It can be arranged so, that multiple blocks process one line of input; in this
 * case, they output multiple results of length k each. Then, a second pass is needed to merge those
 * into one final output.
 */
template <template <int, bool, typename, typename> class WarpSortClass,
          int capacity,
          bool greater,
          typename T,
          typename IdxT>
__global__ void block_kernel(
  const T* in, const IdxT* in_idx, IdxT batch_size, IdxT len, int k, T* out, IdxT* out_idx, T dummy)
{
  extern __shared__ __align__(sizeof(T) * 256) uint8_t smem_buf_bytes[];
  T* smem_buf = (T*)smem_buf_bytes;

  const int num_of_block        = gridDim.x / batch_size;
  const IdxT len_per_block      = (len - 1) / num_of_block + 1;
  const int batch_id            = blockIdx.x / num_of_block;
  const int block_id_in_a_batch = blockIdx.x % num_of_block;

  IdxT start = block_id_in_a_batch * len_per_block;
  IdxT end   = start + len_per_block;
  if (end >= len) { end = len; }

  WarpSortBlockWide<WarpSortClass, capacity, greater, T, IdxT> queue(k, dummy, smem_buf);
  if constexpr (std::is_same_v<WarpSortClass<capacity, greater, T, IdxT>,
                               WarpMerge<capacity, greater, T, IdxT>>) {
    queue.add(in + batch_id * len, in_idx + batch_id * len, start, end);
  } else {
    if (in_idx == nullptr) {
      queue.add(in + batch_id * len, start, end);
    } else {
      queue.add(in + batch_id * len, in_idx + batch_id * len, start, end);
    }
  }

  queue.done();
  queue.dump(out + blockIdx.x * k, out_idx + blockIdx.x * k);
}

template <template <int, bool, typename, typename> class WarpSortClass,
          typename T,
          typename IdxT,
          int Capacity = kMaxCapacity>
struct launch_setup {
  /**
   * @brief Calculate the best block size and minimum grid size for the given `k`.
   *
   * @param[in] k
   *   The select-top-k parameter
   * @param[out] block_size
   *   Returned block size
   * @param[out] min_grid_size
   *   Returned minimum grid size needed to achieve the best potential occupancy
   */
  static void calc_optimal_params(int k, int* block_size, int* min_grid_size)
  {
    const int capacity = calc_capacity(k);
    if constexpr (Capacity > WarpSize) {  // TODO: replace with `Capacity > 1` to allow small sizes.
      if (capacity < Capacity) {
        return launch_setup<WarpSortClass, T, IdxT, Capacity / 2>::calc_optimal_params(
          capacity, block_size, min_grid_size);
      }
    }
    ASSERT(capacity <= Capacity, "Requested k is too big (%d)", k);
    auto calc_smem = [k](int block_size) {
      int num_of_warp = block_size / WarpSize;
      return calc_smem_size_for_block_wide<T>(num_of_warp, k);
    };
    RAFT_CUDA_TRY(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      min_grid_size, block_size, block_kernel<WarpSortClass, Capacity, true, T, IdxT>, calc_smem));
  }

  static void kernel(int k,
                     bool greater,
                     IdxT batch_size,
                     IdxT len,
                     int num_blocks,
                     int block_dim,
                     int smem_size,
                     const T* in_key,
                     const IdxT* in_idx,
                     T* out_key,
                     IdxT* out_idx,
                     cudaStream_t stream)
  {
    const int capacity = calc_capacity(k);
    if constexpr (Capacity > WarpSize) {  // TODO: replace with `Capacity > 1` to allow small sizes.
      if (capacity < Capacity) {
        return launch_setup<WarpSortClass, T, IdxT, Capacity / 2>::kernel(k,
                                                                          greater,
                                                                          batch_size,
                                                                          len,
                                                                          num_blocks,
                                                                          block_dim,
                                                                          smem_size,
                                                                          in_key,
                                                                          in_idx,
                                                                          out_key,
                                                                          out_idx,
                                                                          stream);
      }
    }
    ASSERT(capacity <= Capacity, "Requested k is too big (%d)", k);
    T dummy = greater ? lower_bound<T>() : upper_bound<T>();
    if (greater) {
      block_kernel<WarpSortClass, Capacity, true>
        <<<batch_size * num_blocks, block_dim, smem_size, stream>>>(
          in_key, in_idx, batch_size, len, k, out_key, out_idx, dummy);
    } else {
      block_kernel<WarpSortClass, Capacity, false>
        <<<batch_size * num_blocks, block_dim, smem_size, stream>>>(
          in_key, in_idx, batch_size, len, k, out_key, out_idx, dummy);
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
};

template <template <int, bool, typename, typename> class WarpSortClass>
struct LaunchThreshold {
};

template <>
struct LaunchThreshold<WarpSelect> {
  static constexpr int len_factor_for_multi_block  = 2;
  static constexpr int len_factor_for_single_block = 32;
};

template <>
struct LaunchThreshold<WarpBitonic> {
  static constexpr int len_factor_for_choosing     = 4;
  static constexpr int len_factor_for_multi_block  = 2;
  static constexpr int len_factor_for_single_block = 4;
};

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
void calc_launch_parameter(int batch_size, IdxT len, int k, int* p_num_of_block, int* p_num_of_warp)
{
  const int capacity = calc_capacity(k);
  int block_size     = 0;
  int min_grid_size  = 0;
  launch_setup<WarpSortClass, T, IdxT>::calc_optimal_params(k, &block_size, &min_grid_size);

  int num_of_warp;
  int num_of_block;
  if (batch_size < min_grid_size) {  // may use multiple blocks
    num_of_warp       = block_size / WarpSize;
    num_of_block      = min_grid_size / batch_size;
    int len_per_block = (len - 1) / num_of_block + 1;
    int len_per_warp  = (len_per_block - 1) / num_of_warp + 1;

    len_per_warp  = Pow2<WarpSize>::roundUp(len_per_warp);
    len_per_block = len_per_warp * num_of_warp;
    num_of_block  = (len - 1) / len_per_block + 1;

    constexpr int len_factor = LaunchThreshold<WarpSortClass>::len_factor_for_multi_block;
    if (len_per_warp < capacity * len_factor) {
      len_per_warp  = capacity * len_factor;
      len_per_block = num_of_warp * len_per_warp;
      if ((IdxT)len_per_block > len) { len_per_block = len; }
      num_of_block = (len - 1) / len_per_block + 1;
      num_of_warp  = (len_per_block - 1) / len_per_warp + 1;
    }
  } else {  // use only single block
    num_of_block = 1;

    // block size could be decreased if batch size is large
    float scale = batch_size / min_grid_size;
    if (scale > 1) {
      // make sure scale > 1 so block_size only decreases not increases
      if (0.8 * scale > 1) { scale = 0.8 * scale; }
      block_size /= scale;
      if (block_size < 1) { block_size = 1; }
      block_size = Pow2<WarpSize>::roundUp(block_size);
    }

    num_of_warp      = block_size / WarpSize;
    int len_per_warp = (len - 1) / num_of_warp + 1;
    len_per_warp     = Pow2<WarpSize>::roundUp(len_per_warp);
    num_of_warp      = (len - 1) / len_per_warp + 1;

    constexpr int len_factor = LaunchThreshold<WarpSortClass>::len_factor_for_single_block;
    if (len_per_warp < capacity * len_factor) {
      len_per_warp = capacity * len_factor;
      num_of_warp  = (len - 1) / len_per_warp + 1;
    }
  }

  *p_num_of_block = num_of_block;
  *p_num_of_warp  = num_of_warp;
}

template <typename T, typename IdxT>
void calc_launch_parameter_for_merge(IdxT len, int k, int* num_of_block, int* num_of_warp)
{
  *num_of_block = 1;

  int block_size    = 0;
  int min_grid_size = 0;
  launch_setup<WarpMerge, T, IdxT>::calc_optimal_params(k, &block_size, &min_grid_size);

  *num_of_warp      = block_size / WarpSize;
  IdxT len_per_warp = (len - 1) / (*num_of_warp) + 1;
  len_per_warp      = ((len_per_warp - 1) / k + 1) * k;
  *num_of_warp      = (len - 1) / len_per_warp + 1;
}

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename IdxT>
void warp_sort_topk_(int num_of_block,
                     int num_of_warp,
                     const T* in,
                     const IdxT* in_idx,
                     size_t batch_size,
                     size_t len,
                     int k,
                     T* out,
                     IdxT* out_idx       = nullptr,
                     bool greater        = true,
                     cudaStream_t stream = 0)
{
  rmm::device_uvector<T> tmp_val(num_of_block * k * batch_size, stream);
  rmm::device_uvector<IdxT> tmp_idx(num_of_block * k * batch_size, stream);

  // printf("#block=%d, #warp=%d\n", num_of_block, num_of_warp);
  int capacity = calc_capacity(k);

  T* result_val    = (num_of_block == 1) ? out : tmp_val.data();
  IdxT* result_idx = (num_of_block == 1) ? out_idx : tmp_idx.data();
  int block_dim    = num_of_warp * WarpSize;
  int smem_size    = calc_smem_size_for_block_wide<T>(num_of_warp, (IdxT)k);
  launch_setup<WarpSortClass, T, IdxT>::kernel((IdxT)k,
                                               greater,
                                               (IdxT)batch_size,
                                               (IdxT)len,
                                               num_of_block,
                                               block_dim,
                                               smem_size,
                                               in,
                                               in_idx,
                                               result_val,
                                               result_idx,
                                               stream);

  if (num_of_block > 1) {
    len = k * num_of_block;
    calc_launch_parameter_for_merge<T>(len, k, &num_of_block, &num_of_warp);
    // printf("#block=%d, #warp=%d\n", num_of_block, num_of_warp);
    block_dim = num_of_warp * WarpSize;
    smem_size = calc_smem_size_for_block_wide<T>(num_of_warp, (IdxT)k);
    launch_setup<WarpMerge, T, IdxT>::kernel((IdxT)k,
                                             greater,
                                             (IdxT)batch_size,
                                             (IdxT)len,
                                             num_of_block,
                                             block_dim,
                                             smem_size,
                                             tmp_val.data(),
                                             tmp_idx.data(),
                                             out,
                                             out_idx,
                                             stream);
  }
}

template <typename T, typename IdxT>
void warp_sort_topk(const T* in,
                    const IdxT* in_idx,
                    size_t batch_size,
                    size_t len,
                    int k,
                    T* out,
                    IdxT* out_idx                = nullptr,
                    bool greater                 = true,
                    rmm::cuda_stream_view stream = 0)
{
  ASSERT(k <= kMaxCapacity, "Current max k is %d (requested %d)", kMaxCapacity, k);

  int capacity     = calc_capacity(k);
  int num_of_block = 0;
  int num_of_warp  = 0;
  calc_launch_parameter<WarpBitonic, T>(batch_size, len, (IdxT)k, &num_of_block, &num_of_warp);
  int len_per_warp = len / (num_of_block * num_of_warp);

  if (len_per_warp <= capacity * LaunchThreshold<WarpBitonic>::len_factor_for_choosing) {
    warp_sort_topk_<WarpBitonic, T, IdxT>(
      num_of_block, num_of_warp, in, in_idx, batch_size, len, k, out, out_idx, greater, stream);
  } else {
    calc_launch_parameter<WarpSelect, T>(batch_size, len, k, &num_of_block, &num_of_warp);
    warp_sort_topk_<WarpSelect, T, IdxT>(
      num_of_block, num_of_warp, in, in_idx, batch_size, len, k, out, out_idx, greater, stream);
  }
}

}  // namespace raft::spatial::knn::detail::ivf_flat
