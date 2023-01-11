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

#include <raft/core/cudart_utils.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/device_atomics.cuh>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/vectorized.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>

#include <rmm/device_vector.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>

namespace raft::spatial::knn::detail::topk {
namespace radix_impl {

constexpr int BLOCK_DIM           = 512;
using WideT                       = float4;
constexpr int LAZY_WRITING_FACTOR = 4;

template <int BitsPerPass>
__host__ __device__ constexpr int calc_num_buckets()
{
  return 1 << BitsPerPass;
}

template <typename T, int BitsPerPass>
__host__ __device__ constexpr int calc_num_passes()
{
  return (sizeof(T) * 8 - 1) / BitsPerPass + 1;
}

// bit 0 is the least significant (rightmost) bit
// this function works even when pass=-1, which is used in calc_mask()
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

template <typename T>
__device__ typename cub::Traits<T>::UnsignedBits twiddle_in(T key, bool greater)
{
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
  bits      = cub::Traits<T>::TwiddleIn(bits);
  if (greater) { bits = ~bits; }
  return bits;
}

template <typename T>
__device__ T twiddle_out(typename cub::Traits<T>::UnsignedBits bits, bool greater)
{
  if (greater) { bits = ~bits; }
  bits = cub::Traits<T>::TwiddleOut(bits);
  return reinterpret_cast<T&>(bits);
}

template <typename T, int BitsPerPass>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool greater)
{
  static_assert(BitsPerPass <= sizeof(int) * 8 - 1,
                "BitsPerPass is too large that the result type could not be int");
  return (twiddle_in(x, greater) >> start_bit) & mask;
}

template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(const T* in, IdxT len, Func f)
{
  const IdxT stride = blockDim.x * gridDim.x;
  const int tid     = blockIdx.x * blockDim.x + threadIdx.x;
  if constexpr (sizeof(T) >= sizeof(WideT)) {
    for (IdxT i = tid; i < len; i += stride) {
      f(in[i], i);
    }
  } else {
    static_assert(sizeof(WideT) % sizeof(T) == 0);
    constexpr int items_per_scalar = sizeof(WideT) / sizeof(T);
    // TODO: it's UB
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

    for (IdxT i = tid; i < len_cast; i += stride) {
      wide.scalar       = in_cast[i];
      const IdxT real_i = skip_cnt + i * items_per_scalar;
#pragma unroll
      for (int j = 0; j < items_per_scalar; ++j) {
        f(wide.array[j], real_i + j);
      }
    }

    static_assert(WarpSize >= items_per_scalar);
    // and because items_per_scalar > skip_cnt, WarpSize > skip_cnt
    // no need to use loop
    if (tid < skip_cnt) { f(in[tid], tid); }
    // because len_cast = (len - skip_cnt) / items_per_scalar,
    // len_cast * items_per_scalar + items_per_scalar > len - skip_cnt;
    // and so
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <= WarpSize
    // no need to use loop
    const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + tid;
    if (remain_i < len) { f(in[remain_i], remain_i); }
  }
}

// sync_width should >= WarpSize
template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(const T* in, IdxT len, Func f, int sync_width)
{
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
    unsigned int valid_mask = __ballot_sync(FULL_WARP_MASK, valid);
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
      pos_smem = __shfl_sync(FULL_WARP_MASK, pos_smem, 0);

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
      pos_smem = __shfl_sync(FULL_WARP_MASK, pos_smem, 0);

      __syncwarp();
      if (lane_id_ < warp_pos_) {
        out[pos_smem + lane_id_]     = value_smem_[lane_id_];
        out_idx[pos_smem + lane_id_] = index_smem_[lane_id_];
      }
    }
  }

 private:
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
                             bool greater,
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
      auto f = [greater, start_bit, mask](T value, IdxT) {
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
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
                greater,
                start_bit,
                mask,
                previous_start_bit,
                kth_value_bits,
                p_filter_cnt,
                p_out_cnt,
                early_stop,
                &store](T value, IdxT i, bool valid) {
        const auto previous_bits = (twiddle_in(value, greater) >> previous_start_bit)
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

            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
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
                             bool greater,
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
      auto f = [greater, start_bit, mask](T value, IdxT) {
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
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
                greater,
                start_bit,
                mask,
                previous_start_bit,
                kth_value_bits,
                p_filter_cnt,
                p_out_cnt,
                early_stop](T value, IdxT i) {
        const auto previous_bits = (twiddle_in(value, greater) >> previous_start_bit)
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

            int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
            atomicAdd(histogram_smem + bucket, static_cast<IdxT>(1));
          }
        }
        // '(out_buf || early_stop)':
        // If we skip writing to 'out_buf' (when !out_buf), we should skip
        // writing to 'out' too. So we won't write the same value to 'out'
        // multiple times. And if we keep skipping the writing, values will be
        // written in last_filter_kernel at last. But when 'early_stop' is true,
        // we need to write to 'out' since it's the last chance.
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
    for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
      if (histogram_smem[i] != 0) { atomicAdd(histogram + i, histogram_smem[i]); }
    }
  }
};

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
      counter->k                                   = k - prev;
      counter->len                                 = cur - prev;
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
                            const bool greater,
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
    const auto bits = (twiddle_in(value, greater) >> start_bit) << start_bit;
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
                                   const bool greater)
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
            greater,
            kth_value_bits,
            needed_num_of_kth,
            p_out_cnt,
            p_out_back_cnt,
            in_idx_buf,
            out,
            out_idx](T value, IdxT i) {
    const auto bits = (twiddle_in(value, greater) >> start_bit) << start_bit;
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
                             const bool greater,
                             const int pass)
{
  __shared__ bool isLastBlock;

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
                                                               greater,
                                                               pass,
                                                               early_stop);
  __threadfence();

  if (threadIdx.x == 0) {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    isLastBlock           = (finished == (gridDim.x - 1));
  }

  // Synchronize to make sure that each thread reads the correct value of isLastBlock.
  __syncthreads();
  if (isLastBlock) {
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
          out_buf, out_idx_buf, out, out_idx, current_len, k, counter, greater, pass);
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
  static_assert(sizeof(WideT) / sizeof(T) >= 1);

  int active_blocks;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &active_blocks,
    use_dynamic ? radix_kernel<T, IdxT, BitsPerPass, BlockSize, false, Store>
                : radix_kernel<T, IdxT, BitsPerPass, BlockSize, true, Store>,
    BlockSize,
    0));
  active_blocks *= sm_cnt;

  IdxT best_num_blocks         = 0;
  float best_tail_wave_penalty = 1.0f;
  const IdxT max_num_blocks    = (len - 1) / (sizeof(WideT) / sizeof(T) * BlockSize) + 1;
  for (int num_waves = 1;; ++num_waves) {
    int num_blocks = std::min(max_num_blocks, std::max(num_waves * active_blocks / batch_size, 1));
    IdxT items_per_thread = (len - 1) / (num_blocks * BlockSize) + 1;
    items_per_thread      = (items_per_thread - 1) / (sizeof(WideT) / sizeof(T)) + 1;
    items_per_thread *= sizeof(WideT) / sizeof(T);
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
void radix_topk(void* buf,
                size_t& buf_size,
                const T* in,
                const IdxT* in_idx,
                int batch_size,
                IdxT len,
                IdxT k,
                T* out,
                IdxT* out_idx,
                bool greater,
                cudaStream_t stream,
                bool use_dynamic = false)
{
  // TODO: is it possible to relax this restriction?
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

  Counter<T, IdxT>* counters = nullptr;
  IdxT* histograms           = nullptr;
  T* buf1                    = nullptr;
  IdxT* idx_buf1             = nullptr;
  T* buf2                    = nullptr;
  IdxT* idx_buf2             = nullptr;
  {
    std::vector<size_t> sizes = {sizeof(*counters) * batch_size,
                                 sizeof(*histograms) * num_buckets * batch_size,
                                 sizeof(*buf1) * len * batch_size,
                                 sizeof(*idx_buf1) * len * batch_size,
                                 sizeof(*buf2) * len * batch_size,
                                 sizeof(*idx_buf2) * len * batch_size};
    size_t total_size         = calc_aligned_size(sizes);
    if (!buf) {
      buf_size = total_size;
      return;
    }

    std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
    counters                            = static_cast<decltype(counters)>(aligned_pointers[0]);
    histograms                          = static_cast<decltype(histograms)>(aligned_pointers[1]);
    buf1                                = static_cast<decltype(buf1)>(aligned_pointers[2]);
    idx_buf1                            = static_cast<decltype(idx_buf1)>(aligned_pointers[3]);
    buf2                                = static_cast<decltype(buf2)>(aligned_pointers[4]);
    idx_buf2                            = static_cast<decltype(idx_buf2)>(aligned_pointers[5]);

    RAFT_CUDA_TRY(cudaMemsetAsync(
      buf,
      0,
      static_cast<char*>(aligned_pointers[2]) - static_cast<char*>(aligned_pointers[0]),
      stream));
  }

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
                                           counters,
                                           histograms,
                                           len,
                                           k,
                                           greater,
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
                                           counters,
                                           histograms,
                                           len,
                                           k,
                                           greater,
                                           pass);
    }
  }

  if (use_dynamic) {
    dim3 blocks((len / (sizeof(WideT) / sizeof(T)) - 1) / BlockSize + 1, batch_size);
    last_filter_kernel<T, IdxT, BitsPerPass><<<blocks, BlockSize, 0, stream>>>(
      in, in_idx, out_buf, out_idx_buf, out, out_idx, len, k, counters, greater);
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
                                     bool greater,
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
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
      atomicAdd(histogram + bucket, static_cast<IdxT>(1));
    }
  } else {
    IdxT* p_out_cnt              = &counter->out_cnt;
    const auto kth_value_bits    = counter->kth_value_bits;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);

    for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
      const T value            = in_buf[i];
      const auto previous_bits = (twiddle_in(value, greater) >> previous_start_bit)
                                 << previous_start_bit;
      if (previous_bits == kth_value_bits) {
        IdxT pos         = atomicAdd(p_filter_cnt, static_cast<IdxT>(1));
        out_buf[pos]     = value;
        out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i;

        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
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
                                            const bool greater,
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

    filter_and_histogram<T, IdxT, BitsPerPass>(
      in_buf, in_idx_buf, out_buf, out_idx_buf, out, out_idx, &counter, histogram, greater, pass);
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
                                        greater,
                                        pass);
      break;
    }
  }
}

template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void radix_topk_one_block(void* buf,
                          size_t& buf_size,
                          const T* in,
                          const IdxT* in_idx,
                          int batch_size,
                          IdxT len,
                          IdxT k,
                          T* out,
                          IdxT* out_idx,
                          bool greater,
                          cudaStream_t stream)
{
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);

  T* buf1        = nullptr;
  IdxT* idx_buf1 = nullptr;
  T* buf2        = nullptr;
  IdxT* idx_buf2 = nullptr;
  {
    std::vector<size_t> sizes = {sizeof(*buf1) * len * batch_size,
                                 sizeof(*idx_buf1) * len * batch_size,
                                 sizeof(*buf2) * len * batch_size,
                                 sizeof(*idx_buf2) * len * batch_size};
    size_t total_size         = calc_aligned_size(sizes);
    if (!buf) {
      buf_size = total_size;
      return;
    }

    std::vector<void*> aligned_pointers = calc_aligned_pointers(buf, sizes);
    buf1                                = static_cast<decltype(buf1)>(aligned_pointers[0]);
    idx_buf1                            = static_cast<decltype(idx_buf1)>(aligned_pointers[1]);
    buf2                                = static_cast<decltype(buf2)>(aligned_pointers[2]);
    idx_buf2                            = static_cast<decltype(idx_buf2)>(aligned_pointers[3]);
  }

  radix_topk_one_block_kernel<T, IdxT, BitsPerPass, BlockSize>
    <<<batch_size, BlockSize, 0, stream>>>(
      in, in_idx, len, k, out, out_idx, greater, buf1, idx_buf1, buf2, idx_buf2);
}

}  // namespace radix_impl

template <typename T, typename IdxT>
void radix_topk_11bits(void* buf,
                       size_t& buf_size,
                       const T* in,
                       const IdxT* in_idx,
                       int batch_size,
                       IdxT len,
                       IdxT k,
                       T* out,
                       IdxT* out_idx       = nullptr,
                       bool greater        = true,
                       cudaStream_t stream = 0)
{
  constexpr int items_per_thread = 32;
  if (len <= radix_impl::BLOCK_DIM * items_per_thread) {
    radix_impl::radix_topk_one_block<T, IdxT, 11, radix_impl::BLOCK_DIM>(
      buf, buf_size, in, in_idx, batch_size, len, k, out, out_idx, greater, stream);
  } else if (len < 100.0 * k / batch_size + 0.01) {
    radix_impl::radix_topk<T, IdxT, 11, radix_impl::BLOCK_DIM, radix_impl::BufferedStore>(
      buf, buf_size, in, in_idx, batch_size, len, k, out, out_idx, greater, stream);
  } else {
    radix_impl::radix_topk<T, IdxT, 11, radix_impl::BLOCK_DIM, radix_impl::DirectStore>(
      buf, buf_size, in, in_idx, batch_size, len, k, out, out_idx, greater, stream);
  }
}

template <typename T, typename IdxT>
void radix_topk_11bits_dynamic(void* buf,
                               size_t& buf_size,
                               const T* in,
                               const IdxT* in_idx,
                               int batch_size,
                               IdxT len,
                               IdxT k,
                               T* out,
                               IdxT* out_idx       = nullptr,
                               bool greater        = true,
                               cudaStream_t stream = 0)
{
  constexpr bool use_dynamic = true;

  constexpr int items_per_thread = 32;
  if (len <= radix_impl::BLOCK_DIM * items_per_thread) {
    radix_impl::radix_topk_one_block<T, IdxT, 11, radix_impl::BLOCK_DIM>(
      buf, buf_size, in, in_idx, batch_size, len, k, out, out_idx, greater, stream);
  } else if (len < 100.0 * k / batch_size + 0.01) {
    radix_impl::radix_topk<T, IdxT, 11, radix_impl::BLOCK_DIM, radix_impl::BufferedStore>(
      buf, buf_size, in, in_idx, batch_size, len, k, out, out_idx, greater, stream, use_dynamic);
  } else {
    radix_impl::radix_topk<T, IdxT, 11, radix_impl::BLOCK_DIM, radix_impl::DirectStore>(
      buf, buf_size, in, in_idx, batch_size, len, k, out, out_idx, greater, stream, use_dynamic);
  }
}

}  // namespace raft::spatial::knn::detail::topk
