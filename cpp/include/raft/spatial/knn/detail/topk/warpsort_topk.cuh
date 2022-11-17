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

#include <raft/core/logger.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/pow2_utils.cuh>

#include <algorithm>
#include <functional>
#include <type_traits>

#include <rmm/device_vector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

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

    It uses dynamic shared memory as an intermediate buffer.
    So the required shared memory size should be calculated using
    calc_smem_size_for_block_wide() and passed as the 3rd kernel launch parameter.

    To add elements to the queue, use add(T val, IdxT idx) with unique values per-thread.
    Use WarpSortClass<...>::kDummy constant for the threads outside of input bounds.

    After adding is finished, function done() should be called. And finally, store() is used to get
    the top-k result.

    Example:
      __global__ void kernel() {
        block_sort<warp_sort_immediate, ...> queue(...);

        for (IdxT i = threadIdx.x; i < len, i += blockDim.x) {
          queue.add(in[i], in_idx[i]);
        }

        queue.done();
        queue.store(out, out_idx);
     }

     int smem_size = calc_smem_size_for_block_wide<T>(...);
     kernel<<<grid_dim, block_dim, smem_size>>>();


  3. class warp_sort_filtered and class warp_sort_immediate
    These two classes can be regarded as fixed size priority queue for a warp.
    Usage is similar to class block_sort. No shared memory is needed.

    The host function (warp_sort_topk) uses a heuristic to choose between these two classes for
    sorting, warp_sort_immediate being chosen when the number of inputs per warp is somewhat small
    (see the usage of LaunchThreshold<warp_sort_immediate>::len_factor_for_choosing).

    Example:
      __global__ void kernel() {
        warp_sort_immediate<...> queue(...);
        int warp_id = threadIdx.x / WarpSize;
        int lane_id = threadIdx.x % WarpSize;

        for (IdxT i = lane_id; i < len, i += WarpSize) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        // each warp outputs to a different offset
        queue.store(out + warp_id * k, out_idx + warp_id * k);
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
  static_assert(std::is_default_constructible_v<IdxT>);

 public:
  /**
   *  The `empty` value for the chosen binary operation,
   *  i.e. `Ascending ? upper_bound<T>() : lower_bound<T>()`.
   */
  static constexpr T kDummy = Ascending ? upper_bound<T>() : lower_bound<T>();
  /** Width of the subwarp. */
  static constexpr int kWarpWidth = std::min<int>(Capacity, WarpSize);
  /** The number of elements to select. */
  const int k;

  /**
   * Construct the warp_sort empty queue.
   *
   * @param k
   *   number of elements to select.
   */
  __device__ warp_sort(int k) : k(k)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_arr_[i] = kDummy;
      idx_arr_[i] = IdxT{};
    }
  }

  /**
   * Load k values from the pointers at the given position, and merge them in the storage.
   *
   * When it actually loads the values, it always performs some collective warp operations in the
   * end, thus enforcing warp sync. This means, it's safe to call `store` with the same arguments
   * after `load_sorted` without extra sync. Note, however, that this is not necessarily true for
   * the reverse order, because the access patterns of `store` and `load_sorted` are different.
   *
   * @param[in] in
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] in_idx
   *    a device pointer to a contiguous array, unique per-subwarp
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[in] do_merge
   *    must be the same for all threads within a subwarp of size `kWarpWidth`.
   *    It serves as a conditional; when `false` the function does nothing.
   *    We need it to ensure threads within a full warp don't diverge calling `bitonic::merge()`.
   */
  __device__ void load_sorted(const T* in, const IdxT* in_idx, bool do_merge = true)
  {
    if (do_merge) {
      int idx = Pow2<kWarpWidth>::mod(laneId()) ^ Pow2<kWarpWidth>::Mask;
#pragma unroll
      for (int i = kMaxArrLen - 1; i >= 0; --i, idx += kWarpWidth) {
        if (idx < k) {
          T t = in[idx];
          if (is_ordered<Ascending>(t, val_arr_[i])) {
            val_arr_[i] = t;
            idx_arr_[i] = in_idx[idx];
          }
        }
      }
    }
    if (kWarpWidth < WarpSize || do_merge) {
      topk::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
    }
  }

  /**
   *  Save the content by the pointer location.
   *
   * @param[out] out
   *   device pointer to a contiguous array, unique per-subwarp of size `kWarpWidth`
   *    (length: k <= kWarpWidth * kMaxArrLen).
   * @param[out] out_idx
   *   device pointer to a contiguous array, unique per-subwarp of size `kWarpWidth`
   *    (length: k <= kWarpWidth * kMaxArrLen).
   */
  __device__ void store(T* out, IdxT* out_idx) const
  {
    int idx = Pow2<kWarpWidth>::mod(laneId());
#pragma unroll kMaxArrLen
    for (int i = 0; i < kMaxArrLen && idx < k; i++, idx += kWarpWidth) {
      out[idx]     = val_arr_[i];
      out_idx[idx] = idx_arr_[i];
    }
  }

 protected:
  static constexpr int kMaxArrLen = Capacity / kWarpWidth;

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
    topk::bitonic<kMaxArrLen>(Ascending, kWarpWidth).merge(val_arr_, idx_arr_);
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
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;

  __device__ warp_sort_filtered(int k)
    : warp_sort<Capacity, Ascending, T, IdxT>(k), buf_len_(0), k_th_(kDummy)
  {
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
      idx_buf_[i] = IdxT{};
    }
  }

  __device__ void add(T val, IdxT idx)
  {
    // comparing for k_th should reduce the total amount of updates:
    // `false` means the input value is surely not in the top-k values.
    bool do_add = is_ordered<Ascending>(val, k_th_);
    // merge the buf if it's full and we cannot add an element anymore.
    if (any(buf_len_ + do_add > kMaxBufLen)) {
      // still, add an element before merging if possible for this thread
      if (do_add && buf_len_ < kMaxBufLen) {
        add_to_buf_(val, idx);
        do_add = false;
      }
      merge_buf_();
    }
    // add an element if necessary and haven't already.
    if (do_add) { add_to_buf_(val, idx); }
  }

  __device__ void done()
  {
    if (any(buf_len_ != 0)) { merge_buf_(); }
  }

 private:
  __device__ __forceinline__ void set_k_th_()
  {
    // NB on using srcLane: it's ok if it is outside the warp size / width;
    //                      the modulo op will be done inside the __shfl_sync.
    k_th_ = shfl(val_arr_[kMaxArrLen - 1], k - 1, kWarpWidth);
  }

  __device__ __forceinline__ void merge_buf_()
  {
    topk::bitonic<kMaxBufLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
    this->merge_in<kMaxBufLen>(val_buf_, idx_buf_);
    buf_len_ = 0;
    set_k_th_();  // contains warp sync
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      val_buf_[i] = kDummy;
    }
  }

  __device__ __forceinline__ void add_to_buf_(T val, IdxT idx)
  {
    // NB: the loop is used here to ensure the constant indexing,
    //     to not force the buffers spill into the local memory.
#pragma unroll
    for (int i = 0; i < kMaxBufLen; i++) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }
    buf_len_++;
  }

  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

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
 public:
  using warp_sort<Capacity, Ascending, T, IdxT>::kDummy;
  using warp_sort<Capacity, Ascending, T, IdxT>::kWarpWidth;
  using warp_sort<Capacity, Ascending, T, IdxT>::k;

  __device__ warp_sort_immediate(int k) : warp_sort<Capacity, Ascending, T, IdxT>(k), buf_len_(0)
  {
#pragma unroll
    for (int i = 0; i < kMaxArrLen; i++) {
      val_buf_[i] = kDummy;
      idx_buf_[i] = IdxT{};
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
      topk::bitonic<kMaxArrLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
      this->merge_in<kMaxArrLen>(val_buf_, idx_buf_);
#pragma unroll
      for (int i = 0; i < kMaxArrLen; i++) {
        val_buf_[i] = kDummy;
      }
      buf_len_ = 0;
    }
  }

  __device__ void done()
  {
    if (buf_len_ != 0) {
      topk::bitonic<kMaxArrLen>(!Ascending, kWarpWidth).sort(val_buf_, idx_buf_);
      this->merge_in<kMaxArrLen>(val_buf_, idx_buf_);
    }
  }

 private:
  using warp_sort<Capacity, Ascending, T, IdxT>::kMaxArrLen;
  using warp_sort<Capacity, Ascending, T, IdxT>::val_arr_;
  using warp_sort<Capacity, Ascending, T, IdxT>::idx_arr_;

  T val_buf_[kMaxArrLen];
  IdxT idx_buf_[kMaxArrLen];
  int buf_len_;
};

template <typename T, typename IdxT>
auto calc_smem_size_for_block_wide(int num_of_warp, int k) -> int
{
  return Pow2<256>::roundUp(ceildiv(num_of_warp, 2) * sizeof(T) * k) +
         ceildiv(num_of_warp, 2) * sizeof(IdxT) * k;
}

template <template <int, bool, typename, typename> class WarpSortWarpWide,
          int Capacity,
          bool Ascending,
          typename T,
          typename IdxT>
class block_sort {
 public:
  using queue_t = WarpSortWarpWide<Capacity, Ascending, T, IdxT>;

  __device__ block_sort(int k, uint8_t* smem_buf) : queue_(k)
  {
    val_smem_             = reinterpret_cast<T*>(smem_buf);
    const int num_of_warp = subwarp_align::div(blockDim.x);
    idx_smem_             = reinterpret_cast<IdxT*>(
      smem_buf + Pow2<256>::roundUp(ceildiv(num_of_warp, 2) * sizeof(T) * k));
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

    const int warp_id = subwarp_align::div(threadIdx.x);
    // NB: there is no need for the second __synchthreads between .load_sorted and .store:
    //     we shift the pointers every iteration, such that individual warps either access the same
    //     locations or do not overlap with any of the other warps. The access patterns within warps
    //     are different for the two functions, but .load_sorted implies warp sync at the end, so
    //     there is no need for __syncwarp either.
    for (int shift_mask = ~0, nwarps = subwarp_align::div(blockDim.x), split = (nwarps + 1) >> 1;
         nwarps > 1;
         nwarps = split, split = (nwarps + 1) >> 1) {
      if (warp_id < nwarps && warp_id >= split) {
        int dst_warp_shift = (warp_id - (split & shift_mask)) * queue_.k;
        queue_.store(val_smem_ + dst_warp_shift, idx_smem_ + dst_warp_shift);
      }
      __syncthreads();

      shift_mask = ~shift_mask;  // invert the mask
      {
        int src_warp_shift = (warp_id + (split & shift_mask)) * queue_.k;
        // The last argument serves as a condition for loading
        //  -- to make sure threads within a full warp do not diverge on `bitonic::merge()`
        queue_.load_sorted(
          val_smem_ + src_warp_shift, idx_smem_ + src_warp_shift, warp_id < nwarps - split);
      }
    }
  }

  /** Save the content by the pointer location. */
  __device__ void store(T* out, IdxT* out_idx) const
  {
    if (threadIdx.x < subwarp_align::Value) { queue_.store(out, out_idx); }
  }

 private:
  using subwarp_align = Pow2<queue_t::kWarpWidth>;
  queue_t queue_;
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
__launch_bounds__(256) __global__
  void block_kernel(const T* in, const IdxT* in_idx, IdxT len, int k, T* out, IdxT* out_idx)
{
  extern __shared__ __align__(256) uint8_t smem_buf_bytes[];
  block_sort<WarpSortClass, Capacity, Ascending, T, IdxT> queue(k, smem_buf_bytes);
  in += blockIdx.y * len;
  if (in_idx != nullptr) { in_idx += blockIdx.y * len; }

  const IdxT stride         = gridDim.x * blockDim.x;
  const IdxT per_thread_lim = len + laneId();
  for (IdxT i = threadIdx.x + blockIdx.x * blockDim.x; i < per_thread_lim; i += stride) {
    queue.add(i < len ? __ldcs(in + i) : WarpSortClass<Capacity, Ascending, T, IdxT>::kDummy,
              (i < len && in_idx != nullptr) ? __ldcs(in_idx + i) : i);
  }

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
   * @param[in] block_size_limit
   *   Forcefully limit the block size (optional)
   */
  static void calc_optimal_params(int k,
                                  int* block_size,
                                  int* min_grid_size,
                                  int block_size_limit = 0)
  {
    const int capacity = calc_capacity(k);
    if constexpr (Capacity > 1) {
      if (capacity < Capacity) {
        return launch_setup<WarpSortClass, T, IdxT, Capacity / 2>::calc_optimal_params(
          capacity, block_size, min_grid_size, block_size_limit);
      }
    }
    ASSERT(capacity <= Capacity, "Requested k is too big (%d)", k);
    auto calc_smem = [k](int block_size) {
      int num_of_warp = block_size / std::min<int>(WarpSize, Capacity);
      return calc_smem_size_for_block_wide<T, IdxT>(num_of_warp, k);
    };
    RAFT_CUDA_TRY(cudaOccupancyMaxPotentialBlockSizeVariableSMem(
      min_grid_size,
      block_size,
      block_kernel<WarpSortClass, Capacity, true, T, IdxT>,
      calc_smem,
      block_size_limit));
  }

  static void kernel(int k,
                     bool select_min,
                     size_t batch_size,
                     size_t len,
                     int num_blocks,
                     int block_dim,
                     int smem_size,
                     const T* in_key,
                     const IdxT* in_idx,
                     T* out_key,
                     IdxT* out_idx,
                     rmm::cuda_stream_view stream)
  {
    const int capacity = calc_capacity(k);
    if constexpr (Capacity > 1) {
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

    // This is less than cuda's max block dim along Y axis (65535), but it's a
    // power-of-two, which ensures the alignment of batches in memory.
    constexpr size_t kMaxGridDimY = 32768;
    for (size_t offset = 0; offset < batch_size; offset += kMaxGridDimY) {
      size_t batch_chunk = std::min<size_t>(kMaxGridDimY, batch_size - offset);
      dim3 gs(num_blocks, batch_chunk, 1);
      if (select_min) {
        block_kernel<WarpSortClass, Capacity, true, T, IdxT>
          <<<gs, block_dim, smem_size, stream>>>(in_key, in_idx, IdxT(len), k, out_key, out_idx);
      } else {
        block_kernel<WarpSortClass, Capacity, false, T, IdxT>
          <<<gs, block_dim, smem_size, stream>>>(in_key, in_idx, IdxT(len), k, out_key, out_idx);
      }
      RAFT_CUDA_TRY(cudaPeekAtLastError());
      out_key += batch_chunk * num_blocks * k;
      out_idx += batch_chunk * num_blocks * k;
      in_key += batch_chunk * len;
      if (in_idx != nullptr) { in_idx += batch_chunk * len; }
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
void calc_launch_parameter(
  size_t batch_size, size_t len, int k, int* p_num_of_block, int* p_num_of_warp)
{
  const int capacity               = calc_capacity(k);
  const int capacity_per_full_warp = std::max(capacity, WarpSize);
  int block_size                   = 0;
  int min_grid_size                = 0;
  launch_setup<WarpSortClass, T, IdxT>::calc_optimal_params(k, &block_size, &min_grid_size);
  block_size = Pow2<WarpSize>::roundDown(block_size);

  int num_of_warp;
  int num_of_block;
  if (batch_size < size_t(min_grid_size)) {  // may use multiple blocks
    num_of_warp       = block_size / WarpSize;
    num_of_block      = min_grid_size / int(batch_size);
    int len_per_block = int(ceildiv<size_t>(len, num_of_block));
    int len_per_warp  = ceildiv(len_per_block, num_of_warp);

    len_per_warp  = Pow2<WarpSize>::roundUp(len_per_warp);
    len_per_block = len_per_warp * num_of_warp;
    num_of_block  = int(ceildiv<size_t>(len, len_per_block));

    constexpr int kLenFactor = LaunchThreshold<WarpSortClass>::len_factor_for_multi_block;
    if (len_per_warp < capacity_per_full_warp * kLenFactor) {
      len_per_warp  = capacity_per_full_warp * kLenFactor;
      len_per_block = num_of_warp * len_per_warp;
      if (size_t(len_per_block) > len) { len_per_block = len; }
      num_of_block = int(ceildiv<size_t>(len, len_per_block));
      num_of_warp  = ceildiv(len_per_block, len_per_warp);
    }
  } else {  // use only single block
    num_of_block = 1;

    auto adjust_block_size = [len, capacity_per_full_warp](int bs) {
      int warps_per_block = bs / WarpSize;
      int len_per_warp    = int(ceildiv<size_t>(len, warps_per_block));
      len_per_warp        = Pow2<WarpSize>::roundUp(len_per_warp);
      warps_per_block     = int(ceildiv<size_t>(len, len_per_warp));

      constexpr int kLenFactor = LaunchThreshold<WarpSortClass>::len_factor_for_single_block;
      if (len_per_warp < capacity_per_full_warp * kLenFactor) {
        len_per_warp    = capacity_per_full_warp * kLenFactor;
        warps_per_block = int(ceildiv<size_t>(len, len_per_warp));
      }

      return warps_per_block * WarpSize;
    };

    // gradually reduce the block size while the batch size allows and the len is not big enough
    // to occupy a single block well.
    block_size = adjust_block_size(block_size);
    do {
      num_of_warp               = block_size / WarpSize;
      int another_block_size    = 0;
      int another_min_grid_size = 0;
      launch_setup<WarpSortClass, T, IdxT>::calc_optimal_params(
        k, &another_block_size, &another_min_grid_size, block_size);
      another_block_size = adjust_block_size(another_block_size);
      if (batch_size >= size_t(another_min_grid_size)  // still have enough work
          && another_block_size < block_size           // protect against an infinite loop
          && another_min_grid_size * another_block_size >
               min_grid_size * block_size  // improve occupancy
      ) {
        block_size    = another_block_size;
        min_grid_size = another_min_grid_size;
      } else {
        break;
      }
    } while (block_size > WarpSize);
    num_of_warp = std::max(1, num_of_warp);
  }

  *p_num_of_block = num_of_block;
  *p_num_of_warp  = num_of_warp * capacity_per_full_warp / capacity;
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
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr = nullptr)
{
  auto pool_guard = raft::get_pool_memory_resource(
    mr, num_of_block * k * batch_size * 2 * std::max(sizeof(T), sizeof(IdxT)));
  if (pool_guard) {
    RAFT_LOG_DEBUG("warp_sort_topk: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::device_uvector<T> tmp_val(num_of_block * k * batch_size, stream, mr);
  rmm::device_uvector<IdxT> tmp_idx(num_of_block * k * batch_size, stream, mr);

  int capacity   = calc_capacity(k);
  int warp_width = std::min(capacity, WarpSize);

  T* result_val    = (num_of_block == 1) ? out : tmp_val.data();
  IdxT* result_idx = (num_of_block == 1) ? out_idx : tmp_idx.data();
  int block_dim    = num_of_warp * warp_width;
  int smem_size    = calc_smem_size_for_block_wide<T, IdxT>(num_of_warp, k);

  launch_setup<WarpSortClass, T, IdxT>::kernel(k,
                                               select_min,
                                               batch_size,
                                               len,
                                               num_of_block,
                                               block_dim,
                                               smem_size,
                                               in,
                                               in_idx,
                                               result_val,
                                               result_idx,
                                               stream);

  if (num_of_block > 1) {
    // a second pass to merge the results if necessary
    launch_setup<WarpSortClass, T, IdxT>::kernel(k,
                                                 select_min,
                                                 batch_size,
                                                 k * num_of_block,
                                                 1,
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
 * @param batch_size
 *   number of input rows, i.e. the batch size.
 * @param len
 *   length of a single input array (row); also sometimes referred as n_cols.
 *   Invariant: len >= k.
 * @param k
 *   the number of outputs to select in each input row.
 * @param[out] out
 *   contiguous device array of outputs of size (k * batch_size);
 *   the k smallest/largest values from each row of the `in_keys`.
 * @param[out] out_idx
 *   contiguous device array of outputs of size (k * batch_size);
 *   the payload selected together with `out`.
 * @param select_min
 *   whether to select k smallest (true) or largest (false) keys.
 * @param stream
 * @param mr an optional memory resource to use across the calls (you can provide a large enough
 *           memory pool here to avoid memory allocations within the call).
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
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr = nullptr)
{
  ASSERT(k <= kMaxCapacity, "Current max k is %d (requested %d)", kMaxCapacity, k);
  ASSERT(len <= size_t(std::numeric_limits<IdxT>::max()),
         "The `len` (%zu) does not fit the indexing type",
         len);

  int capacity     = calc_capacity(k);
  int num_of_block = 0;
  int num_of_warp  = 0;
  calc_launch_parameter<warp_sort_immediate, T, IdxT>(
    batch_size, len, k, &num_of_block, &num_of_warp);
  int len_per_thread = len / (num_of_block * num_of_warp * std::min(capacity, WarpSize));

  if (len_per_thread <= LaunchThreshold<warp_sort_immediate>::len_factor_for_choosing) {
    warp_sort_topk_<warp_sort_immediate, T, IdxT>(num_of_block,
                                                  num_of_warp,
                                                  in,
                                                  in_idx,
                                                  batch_size,
                                                  len,
                                                  k,
                                                  out,
                                                  out_idx,
                                                  select_min,
                                                  stream,
                                                  mr);
  } else {
    calc_launch_parameter<warp_sort_filtered, T, IdxT>(
      batch_size, len, k, &num_of_block, &num_of_warp);
    warp_sort_topk_<warp_sort_filtered, T, IdxT>(num_of_block,
                                                 num_of_warp,
                                                 in,
                                                 in_idx,
                                                 batch_size,
                                                 len,
                                                 k,
                                                 out,
                                                 out_idx,
                                                 select_min,
                                                 stream,
                                                 mr);
  }
}

}  // namespace raft::spatial::knn::detail::topk
