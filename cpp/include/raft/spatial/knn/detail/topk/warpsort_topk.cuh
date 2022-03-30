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
  Three APIs of different scopes are provided:
    1. host function: warp_sort_topk()
    2. block-wide API: class block_sort
    3. warp-wide API: class warp_sort_filtered and class warp_sort_immediate


  1. warp_sort_topk()
    (see the docstring)

  2. class block_sort
    It can be regarded as a fixed size priority queue for a thread block,
    although the API is not typical.
    class warp_sort_filtered and warp_sort_immediate can be used to instantiate block_sort.

    It uses dynamic shared memory as intermediate buffer.
    So the required shared memory size should be calculated using
    calc_smem_size_for_block_wide() and passed as the 3rd kernel launch parameter.

    Two overload functions can be used to add items to the queue.
    One is load(const T* in, IdxT start, IdxT end) and it adds a range of items,
    namely [start, end) of in. The idx is inferred from start.
    This function should be called only once to add all items, and should not be
    used together with the add().
    The second one is add(T val, IdxT idx), and it adds only one item pair.
    Note that the range [start, end) is for the whole block of threads, that is,
    each thread in the same block should get the same start/end.
    In contrast, the parameters of the second form are for only one thread,
    so each thread must get different val/idx.

    After adding is finished, function done() should be called. And finally,
    store() is used to get the top-k result.

    Example:
      __global__ void kernel() {
        block_sort<warp_sort_immediate, ...> queue(...);

        // way 1, [0, len) is same for the whole block
        queue.load(in, 0, len);
        // way 2, each thread gets its own val/idx pair
        for (IdxT i = threadIdx.x; i < len, i += blockDim.x) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        queue.store(out, out_idx);
     }

     int smem_size = calc_smem_size_for_block_wide<T>(...);
     kernel<<<grid_dim, block_dim, smem_size>>>();


  3. class warp_sort_filtered and class warp_sort_immediate
    These two classes can be regarded as fixed size priority queue for a warp.
    Usage is similar to class block_sort.
    Two types of add() functions are provided, and also note that [start, end) is
    for a whole warp, while val/idx is for a thread.
    No shared memory is needed.

    The host function (warp_sort_topk) uses a heuristic to choose between these two classes for
    sorting, warp_sort_immediate being chosen when the number of inputs per warp is somewhat small
    (see the usage of LaunchThreshold<warp_sort_immediate>::len_factor_for_choosing).

    Example:
      __global__ void kernel() {
        warp_sort_immediate<...> queue(...);
        int warp_id = threadIdx.x / WarpSize;
        int lane_id = threadIdx.x % WarpSize;

        // way 1, [0, len) is same for the whole warp
        queue.load(in, 0, len);
        // way 2, each thread gets its own val/idx pair
        for (IdxT i = lane_id; i < len, i += WarpSize) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        // each warp outputs to a different offset
        queue.store(out+ warp_id * k, out_idx+ warp_id * k);
      }
 */

namespace raft::spatial::knn::detail::topk {

static constexpr int kMaxCapacity = 256;

namespace {

/** Whether 'left` should indeed be on the left w.r.t. `right`. */
template <bool Ascending, typename T>
__device__ __forceinline__ auto is_ordered(T left, T right) -> bool
{
  if constexpr (Ascending) { return left < right; }
  if constexpr (!Ascending) { return left > right; }
}

constexpr auto calc_capacity(int k) -> int
{
  int capacity = isPo2(k) ? k : (1 << (log2(k) + 1));
  if (capacity < WarpSize) { capacity = WarpSize; }  // TODO: remove this to allow small sizes.
  return capacity;
}

}  // namespace

/**
 * A fixed-size warp-level priority queue.
 * By feeding the data through this queue, you get the `k <= Capacity`
 * smallest/greatest values in the data.
 *
 * @tparam Capacity
 *   maximum number of elements in the queue.
 * @tparam Ascending
 *   which comparison to use: `true` means `<`, collect the smallest elements,
 *   `false` means `>`, collect the greatest elements.
 * @tparam T
 *   the type of keys (what is being compared)
 * @tparam IdxT
 *   the type of payload (normally, indices of elements), i.e.
 *   the content sorted alongside the keys.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort {
  static_assert(isPo2(Capacity));

 public:
  /**
   * Construct the warp_sort empty queue.
   *
   * @param k
   *   number of elements to select.
   * @param dummy
   *   the `empty` value for the choosen binary operation,
   *   i.e. `Ascending ? upper_bound<T>() : lower_bound<T>()`.
   *
   */
  __device__ warp_sort(IdxT k, T dummy) : k_(k), dummy_(dummy)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_arr_[i] = dummy_;
    }
  }

  /**
   * Load k values from the pointers at the given position, and merge them in the storage.
   */
  __device__ void load_sorted(const T* in, const IdxT* in_idx)
  {
    IdxT idx = kWarpWidth - 1 - Pow2<kWarpWidth>::mod(laneId());
#pragma unroll
    for (int i = kMaxArrLen - 1; i >= 0; --i, idx += kWarpWidth) {
      if (idx < k_) {
        T t = in[idx];
        if (is_ordered<Ascending>(t, val_arr_[i])) {
          val_arr_[i] = t;
          idx_arr_[i] = in_idx[idx];
        }
      }
    }
    topk::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
  }

  /** Save the content by the pointer location. */
  __device__ void store(T* out, IdxT* out_idx) const
  {
    IdxT idx = Pow2<kWarpWidth>::mod(laneId());
#pragma unroll kMaxArrLen
    for (int i = 0; i < kMaxArrLen && idx < k_; i++, idx += kWarpWidth) {
      out[idx]     = val_arr_[i];
      out_idx[idx] = idx_arr_[i];
    }
  }

 protected:
  static constexpr int kWarpWidth = std::min<int>(Capacity, WarpSize);
  static constexpr int kMaxArrLen = Capacity / kWarpWidth;

  const IdxT k_;
  const T dummy_;
  T val_arr_[kMaxArrLen];
  IdxT idx_arr_[kMaxArrLen];

  /**
   * Merge another array (sorted in the opposite direction) in the queue.
   * Thanks to the other array being sorted in the opposite direction,
   * it's enough to call bitonic.merge once to maintain the valid state
   * of the queue.
   *
   * @tparam PerThreadSizeIn
   *   the size of the other array per-thread (compared to `kMaxArrLen`).
   *
   * @param keys_in
   *   the values to be merged in. Pointers are unique per-thread. The values
   *   must already be sorted in the opposite direction.
   *   The layout of `keys_in` must be the same as the layout of `val_arr_`.
   * @param ids_in
   *   the associated indices of the elements in the same format as `keys_in`.
   */
  template <int PerThreadSizeIn>
  __device__ __forceinline__ void merge_in(const T* __restrict__ keys_in,
                                           const IdxT* __restrict__ ids_in)
  {
#pragma unroll
    for (int i = std::min(kMaxArrLen, PerThreadSizeIn); i > 0; i--) {
      T& key  = val_arr_[kMaxArrLen - i];
      T other = keys_in[PerThreadSizeIn - i];
      if (is_ordered<Ascending>(other, key)) {
        key                      = other;
        idx_arr_[kMaxArrLen - i] = ids_in[PerThreadSizeIn - i];
      }
    }
    topk::bitonic<kMaxArrLen>(Ascending).merge(val_arr_, idx_arr_);
  }
};

/**
 * This version of warp_sort compares each input element against the current
 * estimate of k-th value before adding it to the intermediate sorting buffer.
 * This makes the algorithm do less sorting steps for long input sequences
 * at the cost of extra checks on each step.
 *
 * This implementation is preferred for large len values.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_filtered : public warp_sort<Capacity, Ascending, T, IdxT> {
  static_assert(Capacity >= WarpSize);

 public:
  __device__ warp_sort_filtered(int k, T dummy)
    : warp_sort<Capacity, Ascending, T, IdxT>(k, dummy), buf_len_(0), k_th_(dummy)
  {
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = dummy_;
    }
  }

  __device__ void load(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
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
    if (is_ordered<Ascending>(val, k_th_)) {
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
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k_ - 1);
  }

  __device__ void merge_buf_()
  {
    topk::bitonic<kMaxBufLen>(!Ascending).sort(val_buf_, idx_buf_);
    this->merge_in<kMaxBufLen>(val_buf_, idx_buf_);
    buf_len_ = 0;
    set_k_th_();  // contains warp sync
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = dummy_;
    }
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::k_;
  using warp_sort<Capacity, Ascending, T, IdxT>::dummy_;

  static constexpr int kMaxBufLen = (Capacity <= 64) ? 2 : 4;

  T val_buf_[kMaxBufLen];
  IdxT idx_buf_[kMaxBufLen];
  int buf_len_;

  T k_th_;
};

/**
 * This version of warp_sort adds every input element into the intermediate sorting
 * buffer, and thus does the sorting step every `Capacity` input elements.
 *
 * This implementation is preferred for very small len values.
 */
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_sort_immediate : public warp_sort<Capacity, Ascending, T, IdxT> {
  static_assert(Capacity >= WarpSize);

 public:
  __device__ warp_sort_immediate(int k, T dummy)
    : warp_sort<Capacity, Ascending, T, IdxT>(k, dummy), buf_len_(0)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_buf_[i] = dummy_;
    }
  }

  __device__ void load(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    add_first_(in, in_idx, start, end);
    start += Capacity;
    while (start < end) {
      add_extra_(in, in_idx, start, end);
      this->merge_in<kMaxArrLen>(val_buf_, idx_buf_);
      start += Capacity;
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
      topk::bitonic<kMaxArrLen>(!Ascending).sort(val_buf_, idx_buf_);
      this->merge_in<kMaxArrLen>(val_buf_, idx_buf_);
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
      topk::bitonic<kMaxArrLen>(!Ascending).sort(val_buf_, idx_buf_);
      this->merge_in<kMaxArrLen>(val_buf_, idx_buf_);
    }
  }

 private:
  /** Fill in the primary val_arr_/idx_arr_ */
  __device__ void add_first_(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    IdxT idx = start + laneId();
    for (int i = 0; i < kMaxArrLen; ++i, idx += WarpSize) {
      if (idx < end) {
        val_arr_[i] = in[idx];
        idx_arr_[i] = in_idx[idx];
      }
    }
    topk::bitonic<kMaxArrLen>(Ascending).sort(val_arr_, idx_arr_);
  }

  /** Fill in the secondary val_buf_/idx_buf_ */
  __device__ void add_extra_(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    IdxT idx = start + laneId();
    for (int i = 0; i < kMaxArrLen; ++i, idx += WarpSize) {
      val_buf_[i] = (idx < end) ? in[idx] : dummy_;
      idx_buf_[i] = (idx < end) ? in_idx[idx] : std::numeric_limits<IdxT>::max();
    }
    topk::bitonic<kMaxArrLen>(!Ascending).sort(val_buf_, idx_buf_);
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::k_;
  using warp_sort<Capacity, Ascending, T, IdxT>::dummy_;

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
template <int Capacity, bool Ascending, typename T, typename IdxT>
class warp_merge : public warp_sort<Capacity, Ascending, T, IdxT> {
 public:
  __device__ warp_merge(int k, T dummy) : warp_sort<Capacity, Ascending, T, IdxT>(k, dummy) {}

  // NB: the input is already sorted, because it's the second pass.
  __device__ void load(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    for (; start < end; start += k_) {
      load_sorted(in + start, in_idx + start);
    }
  }

  __device__ void done() {}

 private:
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::k_;
  using warp_sort<Capacity, Ascending, T, IdxT>::dummy_;
};

template <typename T, typename IdxT>
int calc_smem_size_for_block_wide(int num_of_warp, IdxT k)
{
  return Pow2<256>::roundUp(num_of_warp / 2 * sizeof(T) * k) + num_of_warp / 2 * sizeof(IdxT) * k;
}

template <template <int, bool, typename, typename> class WarpSortWarpWide,
          int Capacity,
          bool Ascending,
          typename T,
          typename IdxT>
class block_sort {
 public:
  __device__ block_sort(int k, T dummy, void* smem_buf) : queue_(k, dummy), k_(k), dummy_(dummy)
  {
    val_smem_             = static_cast<T*>(smem_buf);
    const int num_of_warp = blockDim.x / WarpSize;
    idx_smem_             = reinterpret_cast<IdxT*>(reinterpret_cast<char*>(smem_buf) +
                                        Pow2<256>::roundUp(num_of_warp / 2 * sizeof(T) * k_));
  }

  __device__ void load(const T* in, const IdxT* in_idx, IdxT start, IdxT end)
  {
    int num_of_warp   = blockDim.x / WarpSize;
    const int warp_id = threadIdx.x / WarpSize;
    IdxT len_per_warp = ceildiv<IdxT>(end - start, num_of_warp);
    len_per_warp      = alignTo<IdxT>(len_per_warp, k_);

    IdxT warp_start = start + warp_id * len_per_warp;
    IdxT warp_end   = warp_start + len_per_warp;
    if (warp_end > end) { warp_end = end; }
    queue_.load(in, in_idx, warp_start, warp_end);
  }

  __device__ void add(T val, IdxT idx) { queue_.add(val, idx); }

  /**
   * At the point of calling this function, the warp-level queues consumed all input
   * independently. The remaining work to be done is to merge them together.
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
        queue_.store(val_smem_ + dst_warp_id * k_, idx_smem_ + dst_warp_id * k_);
      }
      __syncthreads();

      if (warp_id < num_of_warp / 2) {
        queue_.load_sorted(val_smem_ + warp_id * k_, idx_smem_ + warp_id * k_);
      }
      __syncthreads();

      num_of_warp = half_num_of_warp;
    }
  }

  /** Save the content by the pointer location. */
  __device__ void store(T* out, IdxT* out_idx) const
  {
    if (threadIdx.x < kWarpWidth) { queue_.store(out, out_idx); }
  }

 private:
  static constexpr int kWarpWidth = std::min<int>(Capacity, WarpSize);

  WarpSortWarpWide<Capacity, Ascending, T, IdxT> queue_;
  int k_;
  T dummy_;
  T* val_smem_;
  IdxT* idx_smem_;
};

/**
 * Uses the `WarpSortClass` to sort chunks of data within one block with no interblock
 * communication. It can be arranged so, that multiple blocks process one row of input; in this
 * case, they output multiple results of length k each. Then, a second pass is needed to merge
 * those into one final output.
 */
template <template <int, bool, typename, typename> class WarpSortClass,
          int Capacity,
          bool Ascending,
          typename T,
          typename IdxT>
__global__ void block_kernel(
  const T* in, const IdxT* in_idx, IdxT len, int k, T* out, IdxT* out_idx, T dummy)
{
  extern __shared__ __align__(sizeof(T) * 256) uint8_t smem_buf_bytes[];
  block_sort<WarpSortClass, Capacity, Ascending, T, IdxT> queue(
    k, dummy, reinterpret_cast<T*>(smem_buf_bytes));
  in += blockIdx.y * len;
  in_idx += blockIdx.y * len;

  const IdxT len_per_block = ceildiv<IdxT>(len, gridDim.x);
  queue.load(
    in, in_idx, blockIdx.x * len_per_block, std::min<IdxT>(len, (blockIdx.x + 1) * len_per_block));

  queue.done();
  const int block_id = blockIdx.x + gridDim.x * blockIdx.y;
  queue.store(out + block_id * k, out_idx + block_id * k);
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
                     bool select_min,
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
                                                                          select_min,
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
    T dummy = select_min ? upper_bound<T>() : lower_bound<T>();
    // This is less than cuda's max block dim along Y axis (65535), but it's a
    // power-of-two, which ensures the alignment of batches in memory.
    constexpr IdxT kMaxGridDimY = 32768;
    for (IdxT offset = 0; offset < batch_size; offset += kMaxGridDimY) {
      IdxT batch_chunk = std::min<IdxT>(kMaxGridDimY, batch_size - offset);
      dim3 gs(num_blocks, batch_chunk, 1);
      if (select_min) {
        block_kernel<WarpSortClass, Capacity, true>
          <<<gs, block_dim, smem_size, stream>>>(in_key + offset * len,
                                                 in_idx + offset * len,
                                                 len,
                                                 k,
                                                 out_key + offset * num_blocks * k,
                                                 out_idx + offset * num_blocks * k,
                                                 dummy);
      } else {
        block_kernel<WarpSortClass, Capacity, false>
          <<<gs, block_dim, smem_size, stream>>>(in_key + offset * len,
                                                 in_idx + offset * len,
                                                 len,
                                                 k,
                                                 out_key + offset * num_blocks * k,
                                                 out_idx + offset * num_blocks * k,
                                                 dummy);
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }
};

template <template <int, bool, typename, typename> class WarpSortClass>
struct LaunchThreshold {
};

template <>
struct LaunchThreshold<warp_sort_filtered> {
  static constexpr int len_factor_for_multi_block  = 2;
  static constexpr int len_factor_for_single_block = 32;
};

template <>
struct LaunchThreshold<warp_sort_immediate> {
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
  launch_setup<warp_merge, T, IdxT>::calc_optimal_params(k, &block_size, &min_grid_size);

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
                     IdxT* out_idx,
                     bool select_min,
                     cudaStream_t stream = 0)
{
  rmm::device_uvector<T> tmp_val(num_of_block * k * batch_size, stream);
  rmm::device_uvector<IdxT> tmp_idx(num_of_block * k * batch_size, stream);

  int capacity = calc_capacity(k);

  T* result_val    = (num_of_block == 1) ? out : tmp_val.data();
  IdxT* result_idx = (num_of_block == 1) ? out_idx : tmp_idx.data();
  int block_dim    = num_of_warp * WarpSize;
  int smem_size    = calc_smem_size_for_block_wide<T>(num_of_warp, (IdxT)k);
  launch_setup<WarpSortClass, T, IdxT>::kernel((IdxT)k,
                                               select_min,
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
    // Merge the results across blocks using warp_merge
    len = k * num_of_block;
    calc_launch_parameter_for_merge<T>(len, k, &num_of_block, &num_of_warp);
    block_dim = num_of_warp * WarpSize;
    smem_size = calc_smem_size_for_block_wide<T>(num_of_warp, (IdxT)k);
    launch_setup<warp_merge, T, IdxT>::kernel((IdxT)k,
                                              select_min,
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

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_keys` as a row-major matrix with len columns and
 * batch_size rows, then this function selects k smallest/largest values in each row and fills
 * in the row-major matrix `out` of size (batch_size, k).
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 *
 * @param[in] in
 *   contiguous device array of inputs of size (len * batch_size);
 *   these are compared and selected.
 * @param[in] in_idx
 *   contiguous device array of inputs of size (len * batch_size);
 *   typically, these are indices of the corresponding in_keys.
 * @param[in] batch_size
 *   number of input rows, i.e. the batch size.
 * @param[in] len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: len >= k.
 * @param[in] k
 *   the number of outputs to select in each input row.
 * @param[out] out
 *   contiguous device array of outputs of size (k * batch_size);
 *   the k smallest/largest values from each row of the `in_keys`.
 * @param[out] out_idx
 *   contiguous device array of outputs of size (k * batch_size);
 *   the payload selected together with `out`.
 * @param[in] select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param[in] stream
 */
template <typename T, typename IdxT>
void warp_sort_topk(const T* in,
                    const IdxT* in_idx,
                    size_t batch_size,
                    size_t len,
                    int k,
                    T* out,
                    IdxT* out_idx,
                    bool select_min,
                    rmm::cuda_stream_view stream = 0)
{
  ASSERT(k <= kMaxCapacity, "Current max k is %d (requested %d)", kMaxCapacity, k);

  int capacity     = calc_capacity(k);
  int num_of_block = 0;
  int num_of_warp  = 0;
  calc_launch_parameter<warp_sort_immediate, T>(
    batch_size, len, (IdxT)k, &num_of_block, &num_of_warp);
  int len_per_warp = len / (num_of_block * num_of_warp);

  if (len_per_warp <= capacity * LaunchThreshold<warp_sort_immediate>::len_factor_for_choosing) {
    warp_sort_topk_<warp_sort_immediate, T, IdxT>(
      num_of_block, num_of_warp, in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
  } else {
    calc_launch_parameter<warp_sort_filtered, T>(batch_size, len, k, &num_of_block, &num_of_warp);
    warp_sort_topk_<warp_sort_filtered, T, IdxT>(
      num_of_block, num_of_warp, in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
  }
}

}  // namespace raft::spatial::knn::detail::topk
