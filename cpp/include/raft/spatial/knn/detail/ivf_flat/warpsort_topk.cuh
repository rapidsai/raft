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
    One is add(const T* in, idxT start, idxT end) and it adds a range of items,
    namely [start, end) of in. The idx is inferred from start.
    This function should be called only once to add all items, and should not be
    used together with the second form of add().
    The second one is add(T val, idxT idx), and it adds only one item pair.
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
        for (idxT i = threadIdx.x; i < len, i += blockDim.x) {
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

    Example:
      __global__ void kernel() {
        WarpBitonic<...> queue(...);
        int warp_id = threadIdx.x / WarpSize;
        int lane_id = threadIdx.x % WarpSize;

        // way 1, [0, len) is same for the whole warp
        queue.add(in, 0, len);
        // way 2, each thread gets its own val/idx pair
        for (idxT i = lane_id; i < len, i += WarpSize) {
          queue.add(in[i], idx[i]);
        }

        queue.done();
        // each warp outputs to a different offset
        queue.dump(out+ warp_id * k * sizeof(T), out_idx+ warp_id * k * sizeof(idxT));
      }
 */

namespace raft::spatial::knn::detail::ivf_flat {

namespace {

template <typename T>
constexpr T get_lower_bound()
{
  if (std::numeric_limits<T>::has_infinity && std::numeric_limits<T>::is_signed) {
    return -std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::lowest();
  }
}

template <typename T>
constexpr T get_upper_bound()
{
  if (std::numeric_limits<T>::has_infinity) {
    return std::numeric_limits<T>::infinity();
  } else {
    return std::numeric_limits<T>::max();
  }
}

template <typename T>
constexpr T get_dummy(bool greater)
{
  return greater ? get_lower_bound<T>() : get_upper_bound<T>();
}

template <bool greater, typename T>
__device__ inline bool is_greater_than(T val, T baseline)
{
  if constexpr (greater) { return val > baseline; }
  if constexpr (!greater) { return val < baseline; }
}

template <typename T>
constexpr HDI T nextHighestPowerOf2(T v)
{
  /**
   * TODO: Not entirely sure if this is what we need in the code of this file.
   *       It returns `r`, such that r > v, r <= v*2, and r is power of two.
   */
  return isPo2(v) ? (v << (T)1) : ((T)1 << (log2(v) + 1));
}

int calc_capacity(int k)
{
  int capacity = nextHighestPowerOf2(k);
  if (capacity < WarpSize) { capacity = WarpSize; }
  return capacity;
}
}  // namespace
template <int capacity, bool greater, typename T, typename idxT>
class WarpSort {
 public:
  __device__ WarpSort(idxT k, T dummy) : lane_(threadIdx.x % WarpSize), k_(k), dummy_(dummy)
  {
    static_assert(capacity >= WarpSize && isPo2(capacity));

    for (int i = 0; i < max_arr_len_; ++i) {
      val_arr_[i] = dummy_;
    }
  }

  // load and merge k sorted values
  __device__ void load_sorted(const T* in, const idxT* in_idx, idxT start)
  {
    idxT idx = start + WarpSize - 1 - lane_;
    for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WarpSize) {
      if (idx < start + k_) {
        T t = in[idx];
        if (is_greater_than<greater>(t, val_arr_[i])) {
          val_arr_[i] = t;
          idx_arr_[i] = in_idx[idx];
        }
      }
    }

    bitonic_merge<capacity, !greater>::run(val_arr_, idx_arr_);
  }

  __device__ void dump(T* out, idxT* out_idx) const
  {
    for (int i = 0; i < max_arr_len_; ++i) {
      idxT out_i = i * WarpSize + lane_;
      if (out_i < k_) {
        out[out_i]     = val_arr_[i];
        out_idx[out_i] = idx_arr_[i];
      }
    }
  }

 protected:
  static constexpr int max_arr_len_ = capacity / WarpSize;

  T val_arr_[max_arr_len_];
  idxT idx_arr_[max_arr_len_];

  const int lane_;
  const idxT k_;
  const T dummy_;
};

template <int capacity, bool greater, typename T, typename idxT>
class WarpSelect : public WarpSort<capacity, greater, T, idxT> {
 public:
  __device__ WarpSelect(idxT k, T dummy)
    : WarpSort<capacity, greater, T, idxT>(k, dummy),
      buf_len_(0),
      k_th_(dummy),
      k_th_lane_((k - 1) % WarpSize)
  {
    for (int i = 0; i < max_buf_len_; ++i) {
      val_buf_[i] = dummy_;
    }
  }

  __device__ void add(const T* in, idxT start, idxT end)
  {
    const idxT end_for_fullwarp = Pow2<WarpSize>::roundUp(end - start) + start;
    for (idxT i = start + lane_; i < end_for_fullwarp; i += WarpSize) {
      T val = (i < end) ? in[i] : dummy_;
      add(val, i);
    }
  }

  __device__ void add(T val, idxT idx)
  {
    if (is_greater_than<greater>(val, k_th_)) {
      for (int i = 0; i < max_buf_len_ - 1; ++i) {
        val_buf_[i] = val_buf_[i + 1];
        idx_buf_[i] = idx_buf_[i + 1];
      }
      val_buf_[max_buf_len_ - 1] = val;
      idx_buf_[max_buf_len_ - 1] = idx;

      ++buf_len_;
    }

    if (any(buf_len_ == max_buf_len_)) { merge_buf_(); }
  }

  __device__ void done()
  {
    if (any(buf_len_ != 0)) { merge_buf_(); }
  }

 private:
  __device__ void set_k_th_()
  {
    // it's the best we can do, should use "val_arr_[k_th_row_]"
    k_th_ = shfl(val_arr_[max_arr_len_ - 1], k_th_lane_);
  }

  __device__ void merge_buf_()
  {
    bitonic_sort<max_buf_len_ * WarpSize, greater>::run(val_buf_, idx_buf_);

    if (max_arr_len_ > max_buf_len_) {
      for (int i = 0; i < max_buf_len_; ++i) {
        T& val = val_arr_[max_arr_len_ - max_buf_len_ + i];
        T& buf = val_buf_[i];
        if (is_greater_than<greater>(buf, val)) {
          val                                       = buf;
          idx_arr_[max_arr_len_ - max_buf_len_ + i] = idx_buf_[i];
        }
      }
    } else if (max_arr_len_ < max_buf_len_) {
      for (int i = 0; i < max_arr_len_; ++i) {
        T& val = val_arr_[i];
        T& buf = val_buf_[max_buf_len_ - max_arr_len_ + i];
        if (is_greater_than<greater>(buf, val)) {
          val         = buf;
          idx_arr_[i] = idx_buf_[max_buf_len_ - max_arr_len_ + i];
        }
      }
    } else {
      for (int i = 0; i < max_arr_len_; ++i) {
        if (is_greater_than<greater>(val_buf_[i], val_arr_[i])) {
          val_arr_[i] = val_buf_[i];
          idx_arr_[i] = idx_buf_[i];
        }
      }
    }

    bitonic_merge<capacity, !greater>::run(val_arr_, idx_arr_);

    buf_len_ = 0;
    set_k_th_();  // contains sync
    for (int i = 0; i < max_buf_len_; ++i) {
      val_buf_[i] = dummy_;
    }
  }

  using WarpSort<capacity, greater, T, idxT>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT>::val_arr_;
  using WarpSort<capacity, greater, T, idxT>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT>::lane_;
  using WarpSort<capacity, greater, T, idxT>::k_;
  using WarpSort<capacity, greater, T, idxT>::dummy_;

  static constexpr int max_buf_len_ = (capacity <= 64) ? 2 : 4;

  T val_buf_[max_buf_len_];
  idxT idx_buf_[max_buf_len_];
  int buf_len_;

  T k_th_;
  const int k_th_lane_;
};

template <int capacity, bool greater, typename T, typename idxT>
class WarpBitonic : public WarpSort<capacity, greater, T, idxT> {
 public:
  __device__ WarpBitonic(idxT k, T dummy)
    : WarpSort<capacity, greater, T, idxT>(k, dummy), buf_len_(0)
  {
    for (int i = 0; i < max_arr_len_; ++i) {
      val_buf_[i] = dummy_;
    }
  }

  __device__ void add(const T* in, idxT start, idxT end)
  {
    add_first_(in, start, end);
    start += capacity;
    while (start < end) {
      add_extra_(in, start, end);
      merge_();
      start += capacity;
    }
  }

  __device__ void add(T val, idxT idx)
  {
    for (int i = 0; i < max_arr_len_; ++i) {
      if (i == buf_len_) {
        val_buf_[i] = val;
        idx_buf_[i] = idx;
      }
    }

    ++buf_len_;
    if (buf_len_ == max_arr_len_) {
      bitonic_sort<capacity, greater>::run(val_buf_, idx_buf_);
      merge_();

      for (int i = 0; i < max_arr_len_; ++i) {
        val_buf_[i] = dummy_;
      }
      buf_len_ = 0;
    }
  }

  __device__ void done()
  {
    if (buf_len_ != 0) {
      bitonic_sort<capacity, greater>::run(val_buf_, idx_buf_);
      merge_();
    }
  }

 private:
  __device__ void add_first_(const T* in, idxT start, idxT end)
  {
    idxT idx = start + lane_;
    for (int i = 0; i < max_arr_len_; ++i, idx += WarpSize) {
      if (idx < end) {
        val_arr_[i] = in[idx];
        idx_arr_[i] = idx;
      }
    }
    bitonic_sort<capacity, !greater>::run(val_arr_, idx_arr_);
  }

  __device__ void add_extra_(const T* in, idxT start, idxT end)
  {
    idxT idx = start + lane_;
    for (int i = 0; i < max_arr_len_; ++i, idx += WarpSize) {
      val_buf_[i] = (idx < end) ? in[idx] : dummy_;
      idx_buf_[i] = idx;
    }
    bitonic_sort<capacity, greater>::run(val_buf_, idx_buf_);
  }

  __device__ void merge_()
  {
    for (int i = 0; i < max_arr_len_; ++i) {
      if (is_greater_than<greater>(val_buf_[i], val_arr_[i])) {
        val_arr_[i] = val_buf_[i];
        idx_arr_[i] = idx_buf_[i];
      }
    }
    bitonic_merge<capacity, !greater>::run(val_arr_, idx_arr_);
  }

  using WarpSort<capacity, greater, T, idxT>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT>::val_arr_;
  using WarpSort<capacity, greater, T, idxT>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT>::lane_;
  using WarpSort<capacity, greater, T, idxT>::k_;
  using WarpSort<capacity, greater, T, idxT>::dummy_;

  T val_buf_[max_arr_len_];
  idxT idx_buf_[max_arr_len_];
  int buf_len_;
};

template <int capacity, bool greater, typename T, typename idxT>
class WarpMerge : public WarpSort<capacity, greater, T, idxT> {
 public:
  __device__ WarpMerge(idxT k, T dummy) : WarpSort<capacity, greater, T, idxT>(k, dummy) {}

  __device__ void add(const T* in, const idxT* in_idx, idxT start, idxT end)
  {
    idxT idx       = start + lane_;
    idxT first_end = (start + k_ < end) ? (start + k_) : end;
    for (int i = 0; i < max_arr_len_; ++i, idx += WarpSize) {
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
  using WarpSort<capacity, greater, T, idxT>::max_arr_len_;
  using WarpSort<capacity, greater, T, idxT>::val_arr_;
  using WarpSort<capacity, greater, T, idxT>::idx_arr_;
  using WarpSort<capacity, greater, T, idxT>::lane_;
  using WarpSort<capacity, greater, T, idxT>::k_;
  using WarpSort<capacity, greater, T, idxT>::dummy_;
};

template <typename T, typename idxT>
int calc_smem_size_for_block_wide(int num_of_warp, idxT k)
{
  return Pow2<256>::roundUp(num_of_warp / 2 * sizeof(T) * k) + num_of_warp / 2 * sizeof(idxT) * k;
}

template <template <int, bool, typename, typename> class WarpSortWarpWide,
          int capacity,
          bool greater,
          typename T,
          typename idxT>
class WarpSortBlockWide {
 public:
  __device__ WarpSortBlockWide(idxT k, T dummy, void* smem_buf)
    : queue_(k, dummy), k_(k), dummy_(dummy)
  {
    val_smem_             = static_cast<T*>(smem_buf);
    const int num_of_warp = blockDim.x / WarpSize;
    idx_smem_             = reinterpret_cast<idxT*>(reinterpret_cast<char*>(smem_buf) +
                                        Pow2<256>::roundUp(num_of_warp / 2 * sizeof(T) * k_));
  }

  __device__ void add(const T* in, const idxT* in_idx, idxT start, idxT end)
  {
    static_assert(std::is_same_v<WarpSortWarpWide<capacity, greater, T, idxT>,
                                 WarpMerge<capacity, greater, T, idxT>>);

    int num_of_warp   = blockDim.x / WarpSize;
    const int warp_id = threadIdx.x / WarpSize;
    idxT len_per_warp = (end - start - 1) / num_of_warp + 1;
    len_per_warp      = ((len_per_warp - 1) / k_ + 1) * k_;

    idxT warp_start = start + warp_id * len_per_warp;
    idxT warp_end   = warp_start + len_per_warp;
    if (warp_end > end) { warp_end = end; }
    queue_.add(in, in_idx, warp_start, warp_end);
  }

  // can't use the form of "in + len" and let the caller pass "in" by setting it to
  // correct offset.
  // It's due to the need to fill idx.
  // so has to pass the correct offset as "start"
  __device__ void add(const T* in, idxT start, idxT end)
  {
    if constexpr (std::is_same_v<WarpSortWarpWide<capacity, greater, T, idxT>,
                                 WarpSelect<capacity, greater, T, idxT>>) {
      const idxT end_for_fullwarp = Pow2<WarpSize>::roundUp(end - start) + start;
      for (idxT i = start + threadIdx.x; i < end_for_fullwarp; i += blockDim.x) {
        T val = (i < end) ? in[i] : dummy_;
        queue_.add(val, i);
      }
    } else if constexpr (std::is_same_v<WarpSortWarpWide<capacity, greater, T, idxT>,
                                        WarpBitonic<capacity, greater, T, idxT>>) {
      int num_of_warp   = blockDim.x / WarpSize;
      const int warp_id = threadIdx.x / WarpSize;
      idxT len_per_warp = (end - start - 1) / num_of_warp + 1;
      len_per_warp      = Pow2<WarpSize>::roundUp(len_per_warp);

      idxT warp_start = start + warp_id * len_per_warp;
      idxT warp_end   = warp_start + len_per_warp;
      if (warp_end > end) { warp_end = end; }
      queue_.add(in, warp_start, warp_end);
    }
  }

  __device__ void add(T val, idxT idx) { queue_.add(val, idx); }

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

  __device__ void dump(T* out, idxT* out_idx) const
  {
    if (threadIdx.x < WarpSize) { queue_.dump(out, out_idx); }
  }

 private:
  WarpSortWarpWide<capacity, greater, T, idxT> queue_;
  int k_;
  T dummy_;
  T* val_smem_;
  idxT* idx_smem_;
};

template <template <int, bool, typename, typename> class WarpSortClass,
          int capacity,
          bool greater,
          typename T,
          typename idxT>
__global__ void block_kernel(const T* in,
                             const idxT* in_idx,
                             idxT batch_size,
                             idxT len,
                             idxT k,
                             T* out,
                             idxT* out_idx,
                             T dummy)
{
  extern __shared__ __align__(sizeof(T) * 256) uint8_t smem_buf_bytes[];
  T* smem_buf = (T*)smem_buf_bytes;

  const int num_of_block        = gridDim.x / batch_size;
  const idxT len_per_block      = (len - 1) / num_of_block + 1;
  const int batch_id            = blockIdx.x / num_of_block;
  const int block_id_in_a_batch = blockIdx.x % num_of_block;

  idxT start = block_id_in_a_batch * len_per_block;
  idxT end   = start + len_per_block;
  if (end >= len) { end = len; }

  WarpSortBlockWide<WarpSortClass, capacity, greater, T, idxT> queue(k, dummy, smem_buf);
  if constexpr (std::is_same_v<WarpSortClass<capacity, greater, T, idxT>,
                               WarpMerge<capacity, greater, T, idxT>>) {
    queue.add(in + batch_id * len, in_idx + batch_id * len, start, end);
  } else {
    queue.add(in + batch_id * len, start, end);
  }

  queue.done();
  queue.dump(out + blockIdx.x * k, out_idx + blockIdx.x * k);
}

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename idxT>
void calc_launch_parameter_by_occupancy(idxT k, int* block_size, int* min_grid_size)
{
  const int capacity                                             = calc_capacity(k);
  decltype(&block_kernel<WarpSortClass, 32, true, T, idxT>) func = nullptr;
  if (capacity == 32) {
    func = block_kernel<WarpSortClass, 32, true, T, idxT>;
  } else if (capacity == 64) {
    func = block_kernel<WarpSortClass, 64, true, T, idxT>;
  } else if (capacity == 128) {
    func = block_kernel<WarpSortClass, 128, true, T, idxT>;
  } else if (capacity == 256) {
    func = block_kernel<WarpSortClass, 256, true, T, idxT>;
  } else {
    ASSERT(false, "Requested capacity is too big (%d)", capacity);
  }

  auto calc_smem = [k](int block_size) {
    int num_of_warp = block_size / WarpSize;
    return calc_smem_size_for_block_wide<T>(num_of_warp, k);
  };

  cudaOccupancyMaxPotentialBlockSizeVariableSMem(min_grid_size, block_size, func, calc_smem);
}

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

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename idxT>
void calc_launch_parameter(
  int batch_size, idxT len, idxT k, int* p_num_of_block, int* p_num_of_warp)
{
  const int capacity = calc_capacity(k);
  int block_size     = 0;
  int min_grid_size  = 0;
  calc_launch_parameter_by_occupancy<WarpSortClass, T, idxT>(k, &block_size, &min_grid_size);

  int num_of_warp;
  int num_of_block;
  if (batch_size < min_grid_size) {  // may use multiple blocks
    num_of_warp        = block_size / WarpSize;
    num_of_block       = min_grid_size / batch_size;
    idxT len_per_block = (len - 1) / num_of_block + 1;
    idxT len_per_warp  = (len_per_block - 1) / num_of_warp + 1;

    len_per_warp  = Pow2<WarpSize>::roundUp(len_per_warp);
    len_per_block = len_per_warp * num_of_warp;
    num_of_block  = (len - 1) / len_per_block + 1;

    constexpr int len_factor = LaunchThreshold<WarpSortClass>::len_factor_for_multi_block;
    if (len_per_warp < capacity * len_factor) {
      len_per_warp  = capacity * len_factor;
      len_per_block = num_of_warp * len_per_warp;
      if (len_per_block > len) { len_per_block = len; }
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

    num_of_warp       = block_size / WarpSize;
    idxT len_per_warp = (len - 1) / num_of_warp + 1;
    len_per_warp      = Pow2<WarpSize>::roundUp(len_per_warp);
    num_of_warp       = (len - 1) / len_per_warp + 1;

    constexpr int len_factor = LaunchThreshold<WarpSortClass>::len_factor_for_single_block;
    if (len_per_warp < capacity * len_factor) {
      len_per_warp = capacity * len_factor;
      num_of_warp  = (len - 1) / len_per_warp + 1;
    }
  }

  *p_num_of_block = num_of_block;
  *p_num_of_warp  = num_of_warp;
}

template <typename T, typename idxT>
void calc_launch_parameter_for_merge(idxT len, idxT k, int* num_of_block, int* num_of_warp)
{
  *num_of_block = 1;

  int block_size    = 0;
  int min_grid_size = 0;
  calc_launch_parameter_by_occupancy<WarpMerge, T, idxT>(k, &block_size, &min_grid_size);

  *num_of_warp      = block_size / WarpSize;
  idxT len_per_warp = (len - 1) / (*num_of_warp) + 1;
  len_per_warp      = ((len_per_warp - 1) / k + 1) * k;
  *num_of_warp      = (len - 1) / len_per_warp + 1;
}

#define BLOCK_CASE(WarpSortClass, capacity, in_val, in_idx, out_val, out_idx) \
  case capacity:                                                              \
    if (greater) {                                                            \
      block_kernel<WarpSortClass, capacity, true>                             \
        <<<batch_size * num_of_block, block_dim, smem_size, stream>>>(        \
          in_val, in_idx, batch_size, len, k, out_val, out_idx, dummy);       \
    } else {                                                                  \
      block_kernel<WarpSortClass, capacity, false>                            \
        <<<batch_size * num_of_block, block_dim, smem_size, stream>>>(        \
          in_val, in_idx, batch_size, len, k, out_val, out_idx, dummy);       \
    }                                                                         \
    break

template <template <int, bool, typename, typename> class WarpSortClass, typename T, typename idxT>
void warp_sort_topk_(int num_of_block,
                     int num_of_warp,
                     void* buf,
                     size_t& buf_size,
                     const T* in,
                     idxT batch_size,
                     idxT len,
                     idxT k,
                     T* out,
                     idxT* out_idx       = nullptr,
                     bool greater        = true,
                     cudaStream_t stream = 0)
{
  T* tmp_val    = nullptr;
  idxT* tmp_idx = nullptr;

  if (num_of_block > 1) {
    std::vector<size_t> sizes = {sizeof(T) * num_of_block * k * batch_size,
                                 sizeof(idxT) * num_of_block * k * batch_size};
    size_t total_size         = calc_aligned_size(sizes);
    if (!buf) {
      buf_size = total_size;
      return;
    }
    std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
    tmp_val                             = static_cast<T*>(aligned_pointers[0]);
    tmp_idx                             = static_cast<idxT*>(aligned_pointers[1]);
  } else if (!buf) {
    // although don't need buf when num_of_block==1, but can't set buf_size=0
    // otherwise, cudaMalloc(&buf, 0) can result in buf==nullptr
    // then the next call of topk() won't do anything but set buf_size again
    // so set buf_size to 1 here to avoid such case
    buf_size = 1;
    return;
  }

  // printf("#block=%d, #warp=%d\n", num_of_block, num_of_warp);
  T dummy      = get_dummy<T>(greater);
  int capacity = calc_capacity(k);

  T* result_val    = (num_of_block == 1) ? out : tmp_val;
  idxT* result_idx = (num_of_block == 1) ? out_idx : tmp_idx;
  int block_dim    = num_of_warp * WarpSize;
  int smem_size    = calc_smem_size_for_block_wide<T>(num_of_warp, k);
  switch (capacity) {
    BLOCK_CASE(WarpSortClass, 32, in, static_cast<idxT*>(nullptr), result_val, result_idx);
    BLOCK_CASE(WarpSortClass, 64, in, static_cast<idxT*>(nullptr), result_val, result_idx);
    BLOCK_CASE(WarpSortClass, 128, in, static_cast<idxT*>(nullptr), result_val, result_idx);
    BLOCK_CASE(WarpSortClass, 256, in, static_cast<idxT*>(nullptr), result_val, result_idx);
    default: ASSERT(false, "Requested capacity is too big (%d)", capacity);
  }
  // CUDA_CHECK_LAST_ERROR();

  if (num_of_block > 1) {
    len = k * num_of_block;
    calc_launch_parameter_for_merge<T>(len, k, &num_of_block, &num_of_warp);
    // printf("#block=%d, #warp=%d\n", num_of_block, num_of_warp);
    block_dim = num_of_warp * WarpSize;
    smem_size = calc_smem_size_for_block_wide<T>(num_of_warp, k);
    switch (capacity) {
      BLOCK_CASE(WarpMerge, 32, tmp_val, tmp_idx, out, out_idx);
      BLOCK_CASE(WarpMerge, 64, tmp_val, tmp_idx, out, out_idx);
      BLOCK_CASE(WarpMerge, 128, tmp_val, tmp_idx, out, out_idx);
      BLOCK_CASE(WarpMerge, 256, tmp_val, tmp_idx, out, out_idx);
      default: ASSERT(false, "Requested capacity is too big (%d)", capacity);
    }
    // CUDA_CHECK_LAST_ERROR();
  }
}

template <typename T, typename idxT>
void warp_sort_topk(void* buf,
                    size_t& buf_size,
                    const T* in,
                    const idxT*,
                    idxT batch_size,
                    idxT len,
                    idxT k,
                    T* out,
                    idxT* out_idx       = nullptr,
                    bool greater        = true,
                    cudaStream_t stream = 0)
{
  ASSERT(k <= 256, "Current max k is 256 (requested %d)", k);

  int capacity     = calc_capacity(k);
  int num_of_block = 0;
  int num_of_warp  = 0;
  calc_launch_parameter<WarpBitonic, T>(batch_size, len, k, &num_of_block, &num_of_warp);
  int len_per_warp = len / (num_of_block * num_of_warp);

  if (len_per_warp <= capacity * LaunchThreshold<WarpBitonic>::len_factor_for_choosing) {
    warp_sort_topk_<WarpBitonic, T, idxT>(num_of_block,
                                          num_of_warp,
                                          buf,
                                          buf_size,
                                          in,
                                          batch_size,
                                          len,
                                          k,
                                          out,
                                          out_idx,
                                          greater,
                                          stream);
  } else {
    calc_launch_parameter<WarpSelect, T>(batch_size, len, k, &num_of_block, &num_of_warp);
    warp_sort_topk_<WarpSelect, T, idxT>(num_of_block,
                                         num_of_warp,
                                         buf,
                                         buf_size,
                                         in,
                                         batch_size,
                                         len,
                                         k,
                                         out,
                                         out_idx,
                                         greater,
                                         stream);
  }
}

}  // namespace raft::spatial::knn::detail::ivf_flat
