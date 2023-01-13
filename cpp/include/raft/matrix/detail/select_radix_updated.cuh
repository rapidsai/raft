/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <raft/core/cudart_utils.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::spatial::knn::detail::topk {
namespace radix_impl {

constexpr int VECTORIZED_READ_SIZE = 16;
constexpr int LAZY_WRITING_FACTOR  = 4;

template <int BitsPerPass>
__host__ __device__ constexpr int calc_num_buckets()
{
  return 1 << BitsPerPass;
}

template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calc_num_passes()
{
  return ceildiv<int>(sizeof(T) * 8, BitsPerPass);
}

/**
 * Bit 0 is the least significant (rightmost);
 * this implementation processes input from the most to the least significant bit.
 * This way, we can skip some passes in the end at the cost of having an unsorted output.
 *
 * NB: Use pass=-1 for calc_mask().
 */
template <typename T, int BitsPerPass>
__device__ constexpr int calc_start_bit(int pass)
{
  int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BitsPerPass;
  if (start_bit < 0) { start_bit = 0; }
  return start_bit;
}

template <typename T, int BitsPerPass>
__device__ constexpr unsigned calc_mask(int pass)
{
  static_assert(BitsPerPass <= 31);
  int num_bits = calc_start_bit<T, BitsPerPass>(pass - 1) - calc_start_bit<T, BitsPerPass>(pass);
  return (1 << num_bits) - 1;
}

/**
 * Use cub to twiddle bits - so that we can correctly compare bits of floating-point values as well
 * as of integers.
 */
template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddle_in(T key, bool select_min)
{
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
  bits      = cub::Traits<T>::TwiddleIn(bits);
  if (!select_min) { bits = ~bits; }
  return bits;
}

template <typename T>
__device__ T twiddle_out(typename cub::Traits<T>::UnsignedBits bits, bool select_min)
{
  if (!select_min) { bits = ~bits; }
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool select_min)
{
  static_assert(BitsPerPass <= sizeof(int) * 8 - 1,
                "BitsPerPass is too large that the result type could not be int");
  return (twiddle_in(x, select_min) >> start_bit) & mask;
}

/**
 * Map a Func over the input data, using vectorized load instructions if possible.
 *
 * NB: in future, we should move this to cpp/include/raft/linalg/detail/unary_op.cuh, which
 *     currently does not support the second lambda argument (index of an element)
 *
 * @tparam T element type
 * @tparam IdxT indexing type
 * @tparam Func void (T x, IdxT idx)
 *
 * @param in the input data
 * @param len the number of elements to read
 * @param f the lambda taking two arguments (T x, IdxT idx)
 */
template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(const T* in, IdxT len, Func f)
{
  const IdxT stride = blockDim.x * gridDim.x;
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (sizeof(T) >= VECTORIZED_READ_SIZE || VECTORIZED_READ_SIZE % sizeof(T) != 0) {
    for (IdxT i = tid; i < len; i += stride) {
      f(in[i], i);
    }
  } else {
    using wide_t      = TxN_t<T, VECTORIZED_READ_SIZE / sizeof(T)>;
    using align_bytes = Pow2<(size_t)VECTORIZED_READ_SIZE>;
    using align_elems = Pow2<wide_t::Ratio>;
    wide_t wide;

    // how many elements to skip in order to do aligned vectorized load
    const IdxT skip_cnt_left = std::min<IdxT>((IdxT)(align_bytes::roundUp(in) - in), len);

    // The main loop: process all aligned data
    for (IdxT i = tid * wide_t::Ratio + skip_cnt_left; i + wide_t::Ratio <= len;
         i += stride * wide_t::Ratio) {
      wide.load(in, i);
#pragma unroll
      for (int j = 0; j < wide_t::Ratio; ++j) {
        f(wide.val.data[j], i + j);
      }
    }

    static_assert(WarpSize >= wide_t::Ratio);
    // Processes the skipped elements on the left
    if (tid < skip_cnt_left) { f(in[tid], tid); }
    // Processes the skipped elements on the right
    const IdxT skip_cnt_right = align_elems::mod(len - skip_cnt_left);
    const IdxT remain_i       = len - skip_cnt_right + tid;
    if (remain_i < len) { f(in[remain_i], remain_i); }
  }
}

// sync_width should >= WarpSize
template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(const T* in, IdxT len, Func f, int sync_width)
{
  using WideT       = float4;
  const IdxT stride = blockDim.x * gridDim.x;
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = tid; i < len; i += stride) {
      f(in[i], i, true);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    union {
      WideT scalar;
      T array[items_per_scalar];
    } wide;

    int skip_cnt = (reinterpret_cast<size_t>(in) % sizeof(WideT))
                     ? ((sizeof(WideT) - reinterpret_cast<size_t>(in) % sizeof(WideT)) / sizeof(T))
                     : 0;
    if (skip_cnt > len) { skip_cnt = len; }
    const WideT* in_cast = reinterpret_cast<decltype(in_cast)>(in + skip_cnt);
    const IdxT len_cast  = (len - skip_cnt) / items_per_scalar;

    const IdxT len_cast_for_sync = ((len_cast - 1) / sync_width + 1) * sync_width;
    for (IdxT i = tid; i < len_cast_for_sync; i += stride) {
      bool valid = i < len_cast;
      if (valid) { wide.scalar = in_cast[i]; }
      const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j, valid);
      }
    }

    static_assert(WarpSize >= items_per_scalar);
    // need at most one warp for skipped and remained elements,
    // and sync_width >= WarpSize
    if (tid < sync_width) {
      bool valid = tid < skip_cnt;
      T value    = valid ? in[tid] : T();
      f(value, tid, valid);

      const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + tid;
      valid               = remain_i < len;
      value               = valid ? in[remain_i] : T();
      f(value, remain_i, valid);
    }
  }
}

template <typename T, typename IdxT>
struct alignas(128) Counter {
  IdxT k;
  IdxT len;
  IdxT previous_len;
  typename cub::Traits<T>::UnsignedBits kth_value_bits;

  alignas(128) IdxT filter_cnt;
  alignas(128) unsigned int finished_block_cnt;
  alignas(128) IdxT out_cnt;
  alignas(128) IdxT out_back_cnt;
};

// not actually used since the specialization for FilterAndHistogram doesn't use this
// implementation
template <typename T, typename IdxT, int>
class DirectStore {
 public:
  __device__ void store(T value, IdxT index, bool valid, T* out, IdxT* out_idx, IdxT* p_out_cnt)
  {
    if (!valid) { return; }
    IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
    out[pos]     = value;
    out_idx[pos] = index;
  }

  __device__ void flush(T*, IdxT*, IdxT*) {}
};

template <typename T, typename IdxT, int BlockSize>
class BufferedStore {
 public:
  __device__ BufferedStore()
  {
    const int warp_id = threadIdx.x >> 5;
    lane_id_          = threadIdx.x % WarpSize;

    __shared__ T value_smem[BlockSize];
    __shared__ IdxT index_smem[BlockSize];

    value_smem_ = value_smem + (warp_id << 5);
    index_smem_ = index_smem + (warp_id << 5);
    warp_pos_   = 0;
  }

  __device__ void store(T value, IdxT index, bool valid, T* out, IdxT* out_idx, IdxT* p_out_cnt)
  {
    unsigned int valid_mask = __ballot_sync(FULL_WARP_MASK_, valid);
    if (valid_mask == 0) { return; }

    int pos = __popc(valid_mask & ((0x1u << lane_id_) - 1)) + warp_pos_;
    if (valid && pos < WarpSize) {
      value_smem_[pos] = value;
      index_smem_[pos] = index;
    }

    warp_pos_ += __popc(valid_mask);
    // Check if the buffer is full
    if (warp_pos_ >= WarpSize) {
      IdxT pos_smem;
      if (lane_id_ == 0) { pos_smem = atomicAdd(p_out_cnt, static_cast<IdxT>(WarpSize)); }
      pos_smem = __shfl_sync(FULL_WARP_MASK_, pos_smem, 0);

      __syncwarp();
      out[pos_smem + lane_id_]     = value_smem_[lane_id_];
      out_idx[pos_smem + lane_id_] = index_smem_[lane_id_];
      __syncwarp();
      // Now the buffer is clean
      if (valid && pos >= WarpSize) {
        pos -= WarpSize;
        value_smem_[pos] = value;
        index_smem_[pos] = index;
      }

      warp_pos_ -= WarpSize;
    }
  }

  __device__ void flush(T* out, IdxT* out_idx, IdxT* p_out_cnt)
  {
    if (warp_pos_ > 0) {
      IdxT pos_smem;
      if (lane_id_ == 0) { pos_smem = atomicAdd(p_out_cnt, static_cast<IdxT>(warp_pos_)); }
      pos_smem = __shfl_sync(FULL_WARP_MASK_, pos_smem, 0);

      __syncwarp();
      if (lane_id_ < warp_pos_) {
        out[pos_smem + lane_id_]     = value_smem_[lane_id_];
        out_idx[pos_smem + lane_id_] = index_smem_[lane_id_];
      }
    }
  }

 private:
  const unsigned FULL_WARP_MASK_{0xffffffff};
  T* value_smem_;
  IdxT* index_smem_;
  IdxT lane_id_;  //@TODO: Can be const variable
  int warp_pos_;
};

template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          template <typename, typename, int>
          class Store>
class FilterAndHistogram {
 public:
  __device__ void operator()(const T* in_buf,
                             const IdxT* in_idx_buf,
                             T* out_buf,
                             IdxT* out_idx_buf,
                             T* out,
                             IdxT* out_idx,
                             IdxT previous_len,
                             Counter<T, IdxT>* counter,
                             IdxT* histogram,
                             bool select_min,
                             int pass,
                             bool early_stop)
  {
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    __shared__ IdxT histogram_smem[num_buckets];
    for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x) {
      histogram_smem[i] = 0;
    }
    Store<T, IdxT, BlockSize> store;
    __syncthreads();

    const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
    const unsigned mask = calc_mask<T, BitsPerPass>(pass);

    if (pass == 0) {
      auto f = [select_min, start_bit, mask](T value, IdxT) {
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
      };
      vectorized_process(in_buf, previous_len, f);
    } else {
      IdxT* p_filter_cnt           = &counter->filter_cnt;
      IdxT* p_out_cnt              = &counter->out_cnt;
      const auto kth_value_bits    = counter->kth_value_bits;
      const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

      auto f = [in_idx_buf,
                out_buf,
                out_idx_buf,
                out,
                out_idx,
                select_min,
                start_bit,
                mask,
                previous_start_bit,
                kth_value_bits,
                p_filter_cnt,
                p_out_cnt,
                early_stop,
                &store](T value, IdxT i, bool valid) {
        const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                   << previous_start_bit;

        if (valid && previous_bits == kth_value_bits) {
          if (early_stop) {
            IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
            out[pos]     = value;
            out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
          } else {
            if (out_buf) {
              IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
              out_buf[pos]     = value;
              out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
            }

            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
            atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
          }
        }

        if (out_buf || early_stop) {
          store.store(value,
                      in_idx_buf ? in_idx_buf[i] : i,
                      valid && previous_bits < kth_value_bits,
                      out,
                      out_idx,
                      p_out_cnt);
        }
      };
      vectorized_process(in_buf, previous_len, f, WarpSize);
      store.flush(out, out_idx, p_out_cnt);
    }
    if (early_stop) { return; }

    __syncthreads();
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
      if (histogram_smem[i] != 0) { atomicAdd(histogram + i, histogram_smem[i]); }
    }
  }
};

/**
 * Fused filtering of the current phase and building histogram for the next phase
 * (see steps 4-1 in `radix_kernel` description).
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
class FilterAndHistogram<T, IdxT, BitsPerPass, BlockSize, DirectStore> {
 public:
  __device__ void operator()(const T* in_buf,
                             const IdxT* in_idx_buf,
                             T* out_buf,
                             IdxT* out_idx_buf,
                             T* out,
                             IdxT* out_idx,
                             IdxT previous_len,
                             Counter<T, IdxT>* counter,
                             IdxT* histogram,
                             bool select_min,
                             int pass,
                             bool early_stop)
  {
    constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
    __shared__ IdxT histogram_smem[num_buckets];
    for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x) {
      histogram_smem[i] = 0;
    }
    __syncthreads();

    const int start_bit = calc_start_bit<T, BitsPerPass>(pass);
    const unsigned mask = calc_mask<T, BitsPerPass>(pass);

    if (pass == 0) {
      // Passed to vectorized_process, this function executes in all blocks in parallel,
      // i.e. the work is split along the input (both, in batches and chunks of a single row).
      // Later, the histograms are merged using atomicAdd.
      auto f = [select_min, start_bit, mask](T value, IdxT) {
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
      };
      vectorized_process(in_buf, previous_len, f);
    } else {
      IdxT* p_filter_cnt           = &counter->filter_cnt;
      IdxT* p_out_cnt              = &counter->out_cnt;
      const auto kth_value_bits    = counter->kth_value_bits;
      const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

      // See the remark above on the distributed execution of `f` using vectorized_process.
      auto f = [in_idx_buf,
                out_buf,
                out_idx_buf,
                out,
                out_idx,
                select_min,
                start_bit,
                mask,
                previous_start_bit,
                kth_value_bits,
                p_filter_cnt,
                p_out_cnt,
                early_stop](T value, IdxT i) {
        const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                   << previous_start_bit;
        if (previous_bits == kth_value_bits) {
          if (early_stop) {
            IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
            out[pos]     = value;
            out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
          } else {
            if (out_buf) {
              IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
              out_buf[pos]     = value;
              out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;
            }

            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
            atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
          }
        }
        // '(out_buf || early_stop)' is a little tricky:
        // If we skip writing to 'out_buf' (when 'out_buf' is false), we should skip writing to
        // 'out' too. So we won't write the same value to 'out' multiple times. And if we keep
        // skipping the writing, values will be written in last_filter_kernel at last.
        // But when 'early_stop' is true, we need to write to 'out' since it's the last chance.
        else if ((out_buf || early_stop) && previous_bits < kth_value_bits) {
          IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
          out[pos]     = value;
          out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
        }
      };
      vectorized_process(in_buf, previous_len, f);
    }
    if (early_stop) { return; }
    __syncthreads();

    // merge histograms produced by individual blocks
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
      if (histogram_smem[i] != 0) { atomicAdd(histogram + i, histogram_smem[i]); }
    }
  }
};

/**
 * Replace a part of the histogram with its own prefix sum, starting from the `start` and adding
 * `current` to each entry of the result.
 * (step 2 in `radix_kernel` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(volatile IdxT* histogram)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  if constexpr (num_buckets >= BlockSize) {
    static_assert(num_buckets % BlockSize == 0);
    constexpr int items_per_thread = num_buckets / BlockSize;
    typedef cub::BlockLoad<IdxT, BlockSize, items_per_thread, cub::BLOCK_LOAD_TRANSPOSE> BlockLoad;
    typedef cub::BlockStore<IdxT, BlockSize, items_per_thread, cub::BLOCK_STORE_TRANSPOSE>
      BlockStore;
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;

    __shared__ union {
      typename BlockLoad::TempStorage load;
      typename BlockScan::TempStorage scan;
      typename BlockStore::TempStorage store;
    } temp_storage;
    IdxT thread_data[items_per_thread];

    BlockLoad(temp_storage.load).Load(histogram, thread_data);
    __syncthreads();

    BlockScan(temp_storage.scan).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    BlockStore(temp_storage.store).Store(histogram, thread_data);
  } else {
    typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;

    IdxT thread_data = 0;
    if (threadIdx.x < num_buckets) { thread_data = histogram[threadIdx.x]; }

    BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
    __syncthreads();

    if (threadIdx.x < num_buckets) { histogram[threadIdx.x] = thread_data; }
  }
}

/**
 * Calculate in which bucket the k-th value will fall
 *  (steps 3 in `radix_kernel` description)
 */
template <typename T, typename IdxT, int BitsPerPass>
__device__ void choose_bucket(Counter<T, IdxT>* counter,
                              const IdxT* histogram,
                              const IdxT k,
                              const int pass)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    IdxT prev = (i == 0) ? 0 : histogram[i - 1];
    IdxT cur  = histogram[i];

    // one and only one thread will satisfy this condition, so only write once
    if (prev < k && cur >= k) {
      counter->k   = k - prev;    // how many values still are there to find
      counter->len = cur - prev;  // number of values in `index` bucket
      typename cub::Traits<T>::UnsignedBits bucket = i;
      int start_bit                                = calc_start_bit<T, BitsPerPass>(pass);
      counter->kth_value_bits |= bucket << start_bit;
    }
  }
}

// For one-block version, last_filter() could be called when pass < num_passes - 1.
// So pass could not be constexpr
template <typename T, typename IdxT, int BitsPerPass>
__device__ void last_filter(const T* out_buf,
                            const IdxT* out_idx_buf,
                            T* out,
                            IdxT* out_idx,
                            IdxT current_len,
                            IdxT k,
                            Counter<T, IdxT>* counter,
                            const bool select_min,
                            const int pass)
{
  const auto kth_value_bits = counter->kth_value_bits;
  const int start_bit       = calc_start_bit<T, BitsPerPass>(pass);

  // changed in choose_bucket(), need to reload
  const IdxT needed_num_of_kth = counter->k;
  IdxT* p_out_cnt              = &counter->out_cnt;
  IdxT* p_out_back_cnt         = &counter->out_back_cnt;
  for (IdxT i = threadIdx.x; i < current_len; i += blockDim.x) {
    const T value   = out_buf[i];
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      out[pos] = value;
      // for one-block version, 'out_idx_buf' could be nullptr at pass 0;
      // and for dynamic version, 'out_idx_buf' could be nullptr if 'out_buf' is
      // 'in'
      out_idx[pos] = out_idx_buf ? out_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < needed_num_of_kth) {
        IdxT pos     = k - 1 - back_pos;
        out[pos]     = value;
        out_idx[pos] = out_idx_buf ? out_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass>
__global__ void last_filter_kernel(const T* in,
                                   const IdxT* in_idx,
                                   const T* in_buf,
                                   const IdxT* in_idx_buf,
                                   T* out,
                                   IdxT* out_idx,
                                   IdxT len,
                                   IdxT k,
                                   Counter<T, IdxT>* counters,
                                   const bool select_min)
{
  const int batch_id = blockIdx.y;

  Counter<T, IdxT>* counter = counters + batch_id;
  IdxT previous_len         = counter->previous_len;
  if (previous_len == 0) { return; }
  if (previous_len > len / LAZY_WRITING_FACTOR) {
    in_buf       = in;
    in_idx_buf   = in_idx;
    previous_len = len;
  }

  in_buf += batch_id * len;
  if (in_idx_buf) { in_idx_buf += batch_id * len; }
  out += batch_id * k;
  out_idx += batch_id * k;

  constexpr int pass      = calc_num_passes<T, BitsPerPass>() - 1;
  constexpr int start_bit = calc_start_bit<T, BitsPerPass>(pass);

  const auto kth_value_bits    = counter->kth_value_bits;
  const IdxT needed_num_of_kth = counter->k;
  IdxT* p_out_cnt              = &counter->out_cnt;
  IdxT* p_out_back_cnt         = &counter->out_back_cnt;

  auto f = [k,
            select_min,
            kth_value_bits,
            needed_num_of_kth,
            p_out_cnt,
            p_out_back_cnt,
            in_idx_buf,
            out,
            out_idx](T value, IdxT i) {
    const auto bits = (twiddle_in(value, select_min) >> start_bit) << start_bit;
    if (bits < kth_value_bits) {
      IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
      out[pos]     = value;
      out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
    } else if (bits == kth_value_bits) {
      IdxT back_pos = atomicAdd(p_out_back_cnt, static_cast<IdxT>(1));
      if (back_pos < needed_num_of_kth) {
        IdxT pos     = k - 1 - back_pos;
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  };

  vectorized_process(in_buf, previous_len, f);
}

/**
 *
 * It is expected to call this kernel multiple times (passes), in each pass we process a radix,
 * going from the most significant towards the least significant bits (MSD).
 *
 * Conceptually, each pass consists of 4 steps:
 *
 * 1. Calculate histogram
 *      First, transform bits into a digit, the value of which is in the range
 *      [0, 2^{BITS_PER_PASS}-1]. Then count the frequency of each digit value and the result is a
 *      histogram. That is, histogram[i] contains the count of inputs having value i.
 *
 * 2. Scan the histogram
 *      Inclusive prefix sum is computed for the histogram. After this step, histogram[i] contains
 *      the count of inputs having value <= i.
 *
 * 3. Find the bucket j of the histogram that the k-th value falls into
 *
 * 4. Filtering
 *      Input elements whose digit value <j are the top-k elements. We put them into the result
 *      array out. The number of such elements is histogram[j-1]. Since the k-th value must be in
 *      the bucket j, we write all elements in bucket j into a intermediate buffer out_buf. For the
 *      next pass, these elements are used as input, and we would like to find the
 *      (k - histogram[j-1])-th value among them. That is, the k in the next pass is set to
 *      (k - histogram[j-1]).
 *
 * In the implementation, the filtering step is delayed to the next pass so the filtering and
 * histogram computation are fused. In this way, inputs are read once rather than twice.
 */
template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          bool use_dynamic,
          template <typename, typename, int>
          class Store>
__global__ void radix_kernel(const T* in,
                             const IdxT* in_idx,
                             const T* in_buf,
                             const IdxT* in_idx_buf,
                             T* out_buf,
                             IdxT* out_idx_buf,
                             T* out,
                             IdxT* out_idx,
                             Counter<T, IdxT>* counters,
                             IdxT* histograms,
                             const IdxT len,
                             const IdxT k,
                             const bool select_min,
                             const int pass)
{
  const int batch_id = blockIdx.y;
  auto counter       = counters + batch_id;
  IdxT current_k;
  IdxT previous_len;
  IdxT current_len;
  if (pass == 0) {
    current_k    = k;
    previous_len = len;
    // Need to do this so setting counter->previous_len for the next pass is correct.
    // This value is meaningless for pass 0, but it's fine because pass 0 won't be the
    // last pass in current implementation so pass 0 won't hit the "if (pass ==
    // num_passes - 1)" branch.
    // Maybe it's better to reload counter->previous_len and use it rather than
    // current_len in last_filter()
    current_len = len;
  } else {
    current_k    = counter->k;
    current_len  = counter->len;
    previous_len = counter->previous_len;
  }
  if (current_len == 0) { return; }
  bool early_stop = (current_len == current_k);

  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  constexpr int num_passes  = calc_num_passes<T, BitsPerPass>();

  if constexpr (use_dynamic) {
    // Figure out if the previous pass writes buffer
    if (previous_len > len / LAZY_WRITING_FACTOR) {
      previous_len = len;
      in_buf       = in;
      in_idx_buf   = in_idx;
    }
    // Figure out if this pass need to write buffer
    if (current_len > len / LAZY_WRITING_FACTOR) {
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    }
  }
  in_buf += batch_id * len;
  if (in_idx_buf) { in_idx_buf += batch_id * len; }
  if (out_buf) { out_buf += batch_id * len; }
  if (out_idx_buf) { out_idx_buf += batch_id * len; }
  if (out) {
    out += batch_id * k;
    out_idx += batch_id * k;
  }
  auto histogram = histograms + batch_id * num_buckets;

  FilterAndHistogram<T, IdxT, BitsPerPass, BlockSize, Store>()(in_buf,
                                                               in_idx_buf,
                                                               out_buf,
                                                               out_idx_buf,
                                                               out,
                                                               out_idx,
                                                               previous_len,
                                                               counter,
                                                               histogram,
                                                               select_min,
                                                               pass,
                                                               early_stop);
  __threadfence();

  bool isLastBlock = false;
  if (threadIdx.x == 0) {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    isLastBlock           = (finished == (gridDim.x - 1));
  }

  if (__syncthreads_or(isLastBlock)) {
    if (early_stop) {
      if (threadIdx.x == 0) {
        // last_filter_kernel from dynamic version requires setting previous_len
        counter->previous_len = 0;
        counter->len          = 0;
      }
      return;
    }

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();
    choose_bucket<T, IdxT, BitsPerPass>(counter, histogram, current_k, pass);
    __syncthreads();

    // reset for next pass
    if (pass != num_passes - 1) {
      for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        histogram[i] = 0;
      }
    }
    if (threadIdx.x == 0) {
      // last_filter_kernel requires setting previous_len even in the last pass
      counter->previous_len = current_len;
      // not necessary for the last pass, but put it here anyway
      counter->filter_cnt = 0;
    }

    if constexpr (!use_dynamic) {
      if (pass == num_passes - 1) {
        last_filter<T, IdxT, BitsPerPass>(
          out_buf, out_idx_buf, out, out_idx, current_len, k, counter, select_min, pass);
      }
    }
  }
}

template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          template <typename, typename, int>
          class Store>
unsigned calc_grid_dim(int batch_size, IdxT len, int sm_cnt, bool use_dynamic)
{
  static_assert(VECTORIZED_READ_SIZE / sizeof(T) >= 1);

  int active_blocks;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks,
    use_dynamic ? radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, Store>
                : radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, Store>,
    BlockSize,
    0));
  active_blocks *= sm_cnt;

  IdxT best_num_blocks         = 0;
  float best_tail_wave_penalty = 1.0f;
  const IdxT max_num_blocks    = (len - 1) / (VECTORIZED_READ_SIZE / sizeof(T) * BlockSize) + 1;
  for (int num_waves = 1;; ++num_waves) {
    IdxT num_blocks = std::min(
      max_num_blocks, static_cast<IdxT>(std::max(num_waves * active_blocks / batch_size, 1)));
    IdxT items_per_thread = (len - 1) / (num_blocks * BlockSize) + 1;
    items_per_thread      = (items_per_thread - 1) / (VECTORIZED_READ_SIZE / sizeof(T)) + 1;
    items_per_thread *= VECTORIZED_READ_SIZE / sizeof(T);
    num_blocks             = (len - 1) / (items_per_thread * BlockSize) + 1;
    float actual_num_waves = static_cast<float>(num_blocks) * batch_size / active_blocks;
    float tail_wave_penalty =
      (ceilf(actual_num_waves) - actual_num_waves) / ceilf(actual_num_waves);

    // 0.15 is determined experimentally. It also ensures breaking the loop early,
    // e.g. when num_waves > 7, tail_wave_penalty will always <0.15
    if (tail_wave_penalty < 0.15) {
      best_num_blocks = num_blocks;
      break;
    } else if (tail_wave_penalty < best_tail_wave_penalty) {
      best_num_blocks        = num_blocks;
      best_tail_wave_penalty = tail_wave_penalty;
    }

    if (num_blocks == max_num_blocks) { break; }
  }
  return best_num_blocks;
}

template <typename T,
          typename IdxT,
          int BitsPerPass,
          int BlockSize,
          template <typename, typename, int>
          class Store>
void radix_topk(const T* in,
                const IdxT* in_idx,
                int batch_size,
                IdxT len,
                IdxT k,
                T* out,
                IdxT* out_idx,
                bool select_min,
                bool use_dynamic,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr)
{
  // TODO: is it possible to relax this restriction?
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

  auto pool_guard =
    raft::get_pool_memory_resource(mr,
                                   batch_size * (sizeof(Counter<T, IdxT>)      // counters
                                                 + sizeof(IdxT) * num_buckets  // histograms
                                                 + sizeof(T) * len * 2         // T bufs
                                                 + sizeof(IdxT) * len * 2      // IdxT bufs
                                                 ) +
                                     256 * 6);  // might need extra memory for alignment
  if (pool_guard) {
    RAFT_LOG_DEBUG("radix_topk: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::device_uvector<Counter<T, IdxT>> counters(batch_size, stream, mr);
  rmm::device_uvector<IdxT> histograms(num_buckets * batch_size, stream, mr);
  rmm::device_uvector<T> buf1(len * batch_size, stream, mr);
  rmm::device_uvector<IdxT> idx_buf1(len * batch_size, stream, mr);
  rmm::device_uvector<T> buf2(len * batch_size, stream, mr);
  rmm::device_uvector<IdxT> idx_buf2(len * batch_size, stream, mr);

  RAFT_CUDA_TRY(
    cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(Counter<T, IdxT>), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(histograms.data(), 0, histograms.size() * sizeof(IdxT), stream));

  const T* in_buf        = nullptr;
  const IdxT* in_idx_buf = nullptr;
  T* out_buf             = nullptr;
  IdxT* out_idx_buf      = nullptr;

  int sm_cnt;
  {
    int dev;
    RAFT_CUDA_TRY(cudaGetDevice(&dev));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sm_cnt, cudaDevAttrMultiProcessorCount, dev));
  }
  dim3 blocks(
    calc_grid_dim<T, IdxT, BitsPerPass, BlockSize, Store>(batch_size, len, sm_cnt, use_dynamic),
    batch_size);

  constexpr int num_passes = calc_num_passes<T, BitsPerPass>();

  for (int pass = 0; pass < num_passes; ++pass) {
    if (pass == 0) {
      in_buf      = in;
      in_idx_buf  = nullptr;
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    } else if (pass == 1) {
      in_buf      = in;
      in_idx_buf  = in_idx;
      out_buf     = buf1.data();
      out_idx_buf = idx_buf1.data();
    } else if (pass % 2 == 0) {
      in_buf      = buf1.data();
      in_idx_buf  = idx_buf1.data();
      out_buf     = buf2.data();
      out_idx_buf = idx_buf2.data();
    } else {
      in_buf      = buf2.data();
      in_idx_buf  = idx_buf2.data();
      out_buf     = buf1.data();
      out_idx_buf = idx_buf1.data();
    }

    if (!use_dynamic) {
      radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, Store>
        <<<blocks, BlockSize, 0, stream>>>(in,
                                           in_idx,
                                           in_buf,
                                           in_idx_buf,
                                           out_buf,
                                           out_idx_buf,
                                           out,
                                           out_idx,
                                           counters.data(),
                                           histograms.data(),
                                           len,
                                           k,
                                           select_min,
                                           pass);
    } else {
      radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, Store>
        <<<blocks, BlockSize, 0, stream>>>(in,
                                           in_idx,
                                           in_buf,
                                           in_idx_buf,
                                           out_buf,
                                           out_idx_buf,
                                           out,
                                           out_idx,
                                           counters.data(),
                                           histograms.data(),
                                           len,
                                           k,
                                           select_min,
                                           pass);
    }
  }

  if (use_dynamic) {
    dim3 blocks((len / (VECTORIZED_READ_SIZE / sizeof(T)) - 1) / BlockSize + 1, batch_size);
    last_filter_kernel<T, IdxT, BitsPerPass><<<blocks, BlockSize, 0, stream>>>(
      in, in_idx, out_buf, out_idx_buf, out, out_idx, len, k, counters.data(), select_min);
  }
}

template <typename T, typename IdxT, int BitsPerPass>
__device__ void filter_and_histogram(const T* in_buf,
                                     const IdxT* in_idx_buf,
                                     T* out_buf,
                                     IdxT* out_idx_buf,
                                     T* out,
                                     IdxT* out_idx,
                                     Counter<T, IdxT>* counter,
                                     IdxT* histogram,
                                     bool select_min,
                                     int pass)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram[i] = 0;
  }
  IdxT* p_filter_cnt = &counter->filter_cnt;
  if (threadIdx.x == 0) { *p_filter_cnt = 0; }
  __syncthreads();

  const int start_bit     = calc_start_bit<T, BitsPerPass>(pass);
  const unsigned mask     = calc_mask<T, BitsPerPass>(pass);
  const IdxT previous_len = counter->previous_len;

  if (pass == 0) {
    // Could not use vectorized_process() as in FilterAndHistogram because
    // vectorized_process() assumes multi-block, e.g. uses gridDim.x
    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      T value    = in_buf[i];
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
      atomicAdd(histogram + bucket, static_cast<IdxT>(1));
    }
  } else {
    IdxT* p_out_cnt              = &counter->out_cnt;
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value            = in_buf[i];
      const auto previous_bits = (twiddle_in(value, select_min) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
        out_buf[pos]     = value;
        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;

        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, select_min);
        atomicAdd(histogram + bucket, static_cast<IdxT>(1));
      } else if (previous_bits < kth_value_bits) {
        IdxT pos     = atomicAdd(p_out_cnt, static_cast<IdxT>(1));
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
__global__ void radix_topk_one_block_kernel(const T* in,
                                            const IdxT* in_idx,
                                            const IdxT len,
                                            const IdxT k,
                                            T* out,
                                            IdxT* out_idx,
                                            const bool select_min,
                                            T* buf1,
                                            IdxT* idx_buf1,
                                            T* buf2,
                                            IdxT* idx_buf2)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  __shared__ Counter<T, IdxT> counter;
  __shared__ IdxT histogram[num_buckets];

  if (threadIdx.x == 0) {
    counter.k              = k;
    counter.len            = len;
    counter.previous_len   = len;
    counter.kth_value_bits = 0;
    counter.out_cnt        = 0;
    counter.out_back_cnt   = 0;
  }
  __syncthreads();

  in += blockIdx.x * len;
  if (in_idx) { in_idx += blockIdx.x * len; }
  out += blockIdx.x * k;
  out_idx += blockIdx.x * k;
  buf1 += blockIdx.x * len;
  idx_buf1 += blockIdx.x * len;
  buf2 += blockIdx.x * len;
  idx_buf2 += blockIdx.x * len;
  const T* in_buf        = nullptr;
  const IdxT* in_idx_buf = nullptr;
  T* out_buf             = nullptr;
  IdxT* out_idx_buf      = nullptr;

  constexpr int num_passes = calc_num_passes<T, BitsPerPass>();
  for (int pass = 0; pass < num_passes; ++pass) {
    if (pass == 0) {
      in_buf      = in;
      in_idx_buf  = nullptr;
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    } else if (pass == 1) {
      in_buf      = in;
      in_idx_buf  = in_idx;
      out_buf     = buf1;
      out_idx_buf = idx_buf1;
    } else if (pass % 2 == 0) {
      in_buf      = buf1;
      in_idx_buf  = idx_buf1;
      out_buf     = buf2;
      out_idx_buf = idx_buf2;
    } else {
      in_buf      = buf2;
      in_idx_buf  = idx_buf2;
      out_buf     = buf1;
      out_idx_buf = idx_buf1;
    }
    IdxT current_len = counter.len;
    IdxT current_k   = counter.k;

    filter_and_histogram<T, IdxT, BitsPerPass>(in_buf,
                                               in_idx_buf,
                                               out_buf,
                                               out_idx_buf,
                                               out,
                                               out_idx,
                                               &counter,
                                               histogram,
                                               select_min,
                                               pass);
    __syncthreads();

    scan<IdxT, BitsPerPass, BlockSize>(histogram);
    __syncthreads();

    choose_bucket<T, IdxT, BitsPerPass>(&counter, histogram, current_k, pass);
    if (threadIdx.x == 0) { counter.previous_len = current_len; }
    __syncthreads();

    if (counter.len == counter.k || pass == num_passes - 1) {
      last_filter<T, IdxT, BitsPerPass>(pass == 0 ? in : out_buf,
                                        pass == 0 ? in_idx : out_idx_buf,
                                        out,
                                        out_idx,
                                        current_len,
                                        k,
                                        &counter,
                                        select_min,
                                        pass);
      break;
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void radix_topk_one_block(const T* in,
                          const IdxT* in_idx,
                          int batch_size,
                          IdxT len,
                          IdxT k,
                          T* out,
                          IdxT* out_idx,
                          bool select_min,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr)
{
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);

  auto pool_guard =
    raft::get_pool_memory_resource(mr,
                                   batch_size * (sizeof(T) * len * 2       // T bufs
                                                 + sizeof(IdxT) * len * 2  // IdxT bufs
                                                 ) +
                                     256 * 4);  // might need extra memory for alignment
  if (pool_guard) {
    RAFT_LOG_DEBUG("radix_topk: using pool memory resource with initial size %zu bytes",
                   pool_guard->pool_size());
  }

  rmm::device_uvector<T> buf1(len * batch_size, stream, mr);
  rmm::device_uvector<IdxT> idx_buf1(len * batch_size, stream, mr);
  rmm::device_uvector<T> buf2(len * batch_size, stream, mr);
  rmm::device_uvector<IdxT> idx_buf2(len * batch_size, stream, mr);

  radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize>
    <<<batch_size, BlockSize, 0, stream>>>(in,
                                           in_idx,
                                           len,
                                           k,
                                           out,
                                           out_idx,
                                           select_min,
                                           buf1.data(),
                                           idx_buf1.data(),
                                           buf2.data(),
                                           idx_buf2.data());
}

}  // namespace radix_impl

/**
 * Select k smallest or largest key/values from each row in the input data.
 *
 * If you think of the input data `in_keys` as a row-major matrix with len columns and
 * batch_size rows, then this function selects k smallest/largest values in each row and fills
 * in the row-major matrix `out` of size (batch_size, k).
 *
 * Note, the output is NOT sorted within the groups of `k` selected elements.
 *
 * @tparam T
 *   the type of the keys (what is being compared).
 * @tparam IdxT
 *   the index type (what is being selected together with the keys).
 * @tparam BitsPerPass
 *   The size of the radix;
 *   it affects the number of passes and number of buckets.
 * @tparam BlockSize
 *   Number of threads in a kernel thread block.
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
 * @param use_dynamic
 *   whether to use the dynamic implementation, which is favorable if the leading bits of input data
 *   are almost the same.
 * @param stream
 * @param mr an optional memory resource to use across the calls (you can provide a large enough
 *           memory pool here to avoid memory allocations within the call).
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void radix_topk_updated(const T* in,
                        const IdxT* in_idx,
                        int batch_size,
                        IdxT len,
                        IdxT k,
                        T* out,
                        IdxT* out_idx,
                        bool select_min,
                        bool use_dynamic,
                        rmm::cuda_stream_view stream,
                        rmm::mr::device_memory_resource* mr = nullptr)
{
  constexpr int items_per_thread = 32;

  if (len <= BlockSize * items_per_thread) {
    radix_impl::radix_topk_one_block<T, IdxT, BitsPerPass, BlockSize>(
      in, in_idx, batch_size, len, k, out, out_idx, select_min, stream, mr);
  } else if (len < 100.0 * k / batch_size + 0.01) {
    radix_impl::radix_topk<T, IdxT, BitsPerPass, BlockSize, radix_impl::BufferedStore>(
      in, in_idx, batch_size, len, k, out, out_idx, select_min, use_dynamic, stream, mr);
  } else {
    radix_impl::radix_topk<T, IdxT, BitsPerPass, BlockSize, radix_impl::DirectStore>(
      in, in_idx, batch_size, len, k, out, out_idx, select_min, use_dynamic, stream, mr);
  }
}

}  // namespace raft::spatial::knn::detail::topk
