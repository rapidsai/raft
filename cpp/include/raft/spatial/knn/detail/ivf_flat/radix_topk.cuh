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

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/radix_rank_sort_operations.cuh>

#include <raft/cudart_utils.h>
#include <raft/device_atomics.cuh>

/*
  Two implementations:

  (1) radix select (select + filter):
      first select the k-th value by going through radix passes,
      then filter out all wanted data from original data

  (2) radix topk:
      filter out wanted data directly while going through radix passes
*/

namespace raft::spatial::knn::detail::ivf_flat {

constexpr int BLOCK_DIM       = 512;
constexpr int ITEM_PER_THREAD = 32;

template <int BITS_PER_PASS>
__host__ __device__ constexpr int calc_num_buckets()
{
  return 1 << BITS_PER_PASS;
}

template <typename T, int BITS_PER_PASS>
__host__ __device__ constexpr int calc_num_passes()
{
  return (sizeof(T) * 8 - 1) / BITS_PER_PASS + 1;
}

// bit 0 is the least significant (rightmost) bit
// this function works even when pass=-1, which is used in calc_mask()
template <typename T, int BITS_PER_PASS>
__device__ constexpr int calc_start_bit(int pass)
{
  int start_bit = static_cast<int>(sizeof(T) * 8) - (pass + 1) * BITS_PER_PASS;
  if (start_bit < 0) { start_bit = 0; }
  return start_bit;
}

template <typename T, int BITS_PER_PASS>
__device__ constexpr unsigned calc_mask(int pass)
{
  static_assert(BITS_PER_PASS <= 31);
  int num_bits =
    calc_start_bit<T, BITS_PER_PASS>(pass - 1) - calc_start_bit<T, BITS_PER_PASS>(pass);
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

template <typename T, int BITS_PER_PASS>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool greater)
{
  static_assert(BITS_PER_PASS <= sizeof(int) * 8 - 1);  // so return type can be int
  return (twiddle_in(x, greater) >> start_bit) & mask;
}

template <typename T, typename IdxT, typename Func>
__device__ void vectorized_process(const T* in, IdxT len, Func f)
{
  using WideT = float4;

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
    // len - (skip_cnt + len_cast * items_per_scalar) < items_per_scalar <=
    // WarpSize no need to use loop
    const IdxT remain_i = skip_cnt + len_cast * items_per_scalar + tid;
    if (remain_i < len) { f(in[remain_i], remain_i); }
  }
}

template <typename T, typename IdxT>
struct Counter {
  IdxT k;
  IdxT len;
  IdxT previous_len;
  int bucket;

  IdxT filter_cnt;
  unsigned int finished_block_cnt;
  IdxT out_cnt;
  IdxT out_back_cnt;
  T kth_value;
};

template <typename T, typename IdxT, int BITS_PER_PASS>
__device__ void filter_and_histogram(const T* in_buf,
                                     const IdxT* in_idx_buf,
                                     T* out_buf,
                                     IdxT* out_idx_buf,
                                     T* out,
                                     IdxT* out_idx,
                                     IdxT len,
                                     Counter<T, IdxT>* counter,
                                     IdxT* histogram,
                                     bool greater,
                                     int pass,
                                     int k)
{
  constexpr int num_buckets = calc_num_buckets<BITS_PER_PASS>();
  __shared__ IdxT histogram_smem[num_buckets];
  for (IdxT i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    histogram_smem[i] = 0;
  }
  __syncthreads();

  const int start_bit = calc_start_bit<T, BITS_PER_PASS>(pass);
  const unsigned mask = calc_mask<T, BITS_PER_PASS>(pass);

  if (pass == 0) {
    auto f = [greater, start_bit, mask](T value, IdxT) {
      int bucket = calc_bucket<T, BITS_PER_PASS>(value, start_bit, mask, greater);
      atomicAdd(histogram_smem + bucket, IdxT(1));
    };
    vectorized_process(in_buf, len, f);
  } else {
    const IdxT previous_len      = counter->previous_len;
    const int want_bucket        = counter->bucket;
    IdxT& filter_cnt             = counter->filter_cnt;
    IdxT& out_cnt                = counter->out_cnt;
    T& kth_value                 = counter->kth_value;
    const IdxT counter_len       = counter->len;
    const int previous_start_bit = calc_start_bit<T, BITS_PER_PASS>(pass - 1);
    const unsigned previous_mask = calc_mask<T, BITS_PER_PASS>(pass - 1);

    auto f = [in_idx_buf,
              out_buf,
              out_idx_buf,
              out,
              out_idx,
              greater,
              k,
              start_bit,
              mask,
              previous_start_bit,
              previous_mask,
              want_bucket,
              &filter_cnt,
              &out_cnt,
              &kth_value,
              counter_len](T value, IdxT i) {
      int prev_bucket =
        calc_bucket<T, BITS_PER_PASS>(value, previous_start_bit, previous_mask, greater);
      if (prev_bucket == want_bucket) {
        IdxT pos     = atomicAdd(&filter_cnt, IdxT(1));
        out_buf[pos] = value;
        if (out_idx_buf) { out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i; }
        int bucket = calc_bucket<T, BITS_PER_PASS>(value, start_bit, mask, greater);
        atomicAdd(histogram_smem + bucket, IdxT(1));

        if (counter_len == 1) {
          if (out) {
            out[k - 1]     = value;
            out_idx[k - 1] = in_idx_buf ? in_idx_buf[i] : i;
          } else {
            kth_value = value;
          }
        }
      } else if (out && prev_bucket < want_bucket) {
        IdxT pos     = atomicAdd(&out_cnt, IdxT(1));
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    };

    vectorized_process(in_buf, previous_len, f);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    if (histogram_smem[i] != 0) { atomicAdd(histogram + i, histogram_smem[i]); }
  }
}

template <typename IdxT, int BITS_PER_PASS, int NUM_THREAD>
__device__ void scan(volatile IdxT* histogram,
                     const int start,
                     const int num_buckets,
                     const IdxT current)
{
  typedef cub::BlockScan<IdxT, NUM_THREAD> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;

  IdxT thread_data = 0;
  int index        = start + threadIdx.x;
  if (index < num_buckets) { thread_data = histogram[index]; }

  BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
  __syncthreads();
  if (index < num_buckets) { histogram[index] = thread_data + current; }
  __syncthreads();  // This sync is necessary, as the content of histogram needs
                    // to be read after
}

template <typename T, typename IdxT, int BITS_PER_PASS, int NUM_THREAD>
__device__ void choose_bucket(Counter<T, IdxT>* counter, IdxT* histogram, const IdxT k)
{
  constexpr int num_buckets = calc_num_buckets<BITS_PER_PASS>();
  int index                 = threadIdx.x;
  IdxT current_value        = 0;
  int num_pass              = 1;
  if constexpr (num_buckets >= NUM_THREAD) {
    static_assert(num_buckets % NUM_THREAD == 0);
    num_pass = num_buckets / NUM_THREAD;
  }

  for (int i = 0; i < num_pass && (current_value < k); i++) {
    scan<IdxT, BITS_PER_PASS, NUM_THREAD>(histogram, i * NUM_THREAD, num_buckets, current_value);
    if (index < num_buckets) {
      IdxT prev = (index == 0) ? 0 : histogram[index - 1];
      IdxT cur  = histogram[index];

      // one and only one thread will satisfy this condition, so only write once
      if (prev < k && cur >= k) {
        counter->k            = k - prev;
        counter->previous_len = counter->len;
        counter->len          = cur - prev;
        counter->bucket       = index;
      }
    }
    index += NUM_THREAD;
    current_value = histogram[(i + 1) * NUM_THREAD - 1];
  }
}

template <typename T, typename IdxT, int BITS_PER_PASS, int NUM_THREAD>
__global__ void radix_kernel(const T* in_buf,
                             const IdxT* in_idx_buf,
                             T* out_buf,
                             IdxT* out_idx_buf,
                             T* out,
                             IdxT* out_idx,
                             Counter<T, IdxT>* counters,
                             IdxT* histograms,
                             const IdxT len,
                             const int k,
                             const bool greater,
                             const int pass)
{
  __shared__ bool isLastBlockDone;

  constexpr int num_buckets = calc_num_buckets<BITS_PER_PASS>();
  constexpr int num_passes  = calc_num_passes<T, BITS_PER_PASS>();
  const int batch_id        = blockIdx.y;
  in_buf += batch_id * len;
  out_buf += batch_id * len;
  if (in_idx_buf) { in_idx_buf += batch_id * len; }
  if (out_idx_buf) { out_idx_buf += batch_id * len; }
  if (out) {
    out += batch_id * k;
    out_idx += batch_id * k;
  }
  auto counter   = counters + batch_id;
  auto histogram = histograms + batch_id * num_buckets;

  filter_and_histogram<T, IdxT, BITS_PER_PASS>(in_buf,
                                               in_idx_buf,
                                               out_buf,
                                               out_idx_buf,
                                               out,
                                               out_idx,
                                               len,
                                               counter,
                                               histogram,
                                               greater,
                                               pass,
                                               k);
  __threadfence();

  if (threadIdx.x == 0) {
    unsigned int finished = atomicInc(&counter->finished_block_cnt, gridDim.x - 1);
    isLastBlockDone       = (finished == (gridDim.x - 1));
  }

  // Synchronize to make sure that each thread reads the correct value of
  // isLastBlockDone.
  __syncthreads();
  if (isLastBlockDone) {
    if (counter->len == 1 && threadIdx.x == 0) {
      counter->previous_len = 0;
      counter->len          = 0;
    }
    // init counter, other members of counter is initialized with 0 by
    // cudaMemset()
    if (pass == 0 && threadIdx.x == 0) {
      counter->k   = k;
      counter->len = len;
      if (out) { counter->out_back_cnt = 0; }
    }
    __syncthreads();

    IdxT ori_k = counter->k;

    if (counter->len > 0) {
      choose_bucket<T, IdxT, BITS_PER_PASS, NUM_THREAD>(counter, histogram, ori_k);
    }

    __syncthreads();
    if (pass == num_passes - 1) {
      const IdxT previous_len = counter->previous_len;
      const int want_bucket   = counter->bucket;
      int start_bit           = calc_start_bit<T, BITS_PER_PASS>(pass);
      unsigned mask           = calc_mask<T, BITS_PER_PASS>(pass);

      if (!out) {  // radix select
        for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
          const T value = out_buf[i];
          int bucket    = calc_bucket<T, BITS_PER_PASS>(value, start_bit, mask, greater);
          if (bucket == want_bucket) {
            // TODO: UB
            // could use atomicExch, but it's not defined for T=half
            counter->kth_value = value;
            break;
          }
        }
      } else {  // radix topk
        IdxT& out_cnt = counter->out_cnt;
        for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
          const T value = out_buf[i];
          int bucket    = calc_bucket<T, BITS_PER_PASS>(value, start_bit, mask, greater);
          if (bucket < want_bucket) {
            IdxT pos     = atomicAdd(&out_cnt, IdxT(1));
            out[pos]     = value;
            out_idx[pos] = out_idx_buf[i];
          } else if (bucket == want_bucket) {
            IdxT needed_num_of_kth = counter->k;
            IdxT back_pos          = atomicAdd(&(counter->out_back_cnt), IdxT(1));
            if (back_pos < needed_num_of_kth) {
              IdxT pos     = k - 1 - back_pos;
              out[pos]     = value;
              out_idx[pos] = out_idx_buf[i];
            }
          }
        }
        __syncthreads();
      }
    } else {
      // reset for next pass
      for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        histogram[i] = 0;
      }
      if (threadIdx.x == 0) { counter->filter_cnt = 0; }
    }
  }
}

template <typename T, typename IdxT, int BITS_PER_PASS, int NUM_THREAD>
void radix_topk(const T* in,
                const IdxT* in_idx,
                size_t batch_size,
                size_t len,
                int k,
                T* out,
                IdxT* out_idx,
                bool greater,
                rmm::cuda_stream_view stream)
{
  // TODO: is it possible to relax this restriction?
  static_assert(calc_num_passes<T, BITS_PER_PASS>() > 1);
  constexpr int num_buckets = calc_num_buckets<BITS_PER_PASS>();

  rmm::device_uvector<Counter<T, IdxT>> counters(batch_size, stream);
  rmm::device_uvector<IdxT> histograms(num_buckets * batch_size, stream);
  rmm::device_uvector<T> buf1(len * batch_size, stream);
  rmm::device_uvector<IdxT> idx_buf1(len * batch_size, stream);
  rmm::device_uvector<T> buf2(len * batch_size, stream);
  rmm::device_uvector<IdxT> idx_buf2(len * batch_size, stream);

  RAFT_CUDA_TRY(
    cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(Counter<T, IdxT>), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(histograms.data(), 0, histograms.size() * sizeof(IdxT), stream));

  const T* in_buf        = nullptr;
  const IdxT* in_idx_buf = nullptr;
  T* out_buf             = nullptr;
  IdxT* out_idx_buf      = nullptr;

  dim3 blocks((len - 1) / (NUM_THREAD * ITEM_PER_THREAD) + 1, batch_size);

  constexpr int num_passes = calc_num_passes<T, BITS_PER_PASS>();

  for (int pass = 0; pass < num_passes; ++pass) {
    if (pass == 0) {
      in_buf      = in;
      in_idx_buf  = nullptr;
      out_buf     = nullptr;
      out_idx_buf = nullptr;
    } else if (pass == 1) {
      in_buf      = in;
      in_idx_buf  = in_idx ? in_idx : nullptr;
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

    radix_kernel<T, IdxT, BITS_PER_PASS, NUM_THREAD>
      <<<blocks, NUM_THREAD, 0, stream>>>(in_buf,
                                          in_idx_buf,
                                          out_buf,
                                          out_idx_buf,
                                          out,
                                          out_idx,
                                          counters.data(),
                                          histograms.data(),
                                          len,
                                          k,
                                          greater,
                                          pass);
  }
}

}  // namespace raft::spatial::knn::detail::ivf_flat
