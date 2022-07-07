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
#include <raft/vectorized.cuh>

namespace raft::spatial::knn::detail::topk {

constexpr int ITEM_PER_THREAD      = 32;
constexpr int VECTORIZED_READ_SIZE = 16;

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

// Minimum reasonable block size for the given radix size.
template <int BitsPerPass>
__host__ __device__ constexpr int calc_min_block_size()
{
  return 1 << std::max<int>(BitsPerPass - 4, Pow2<WarpSize>::Log2 + 1);
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
__device__ typename cub::Traits<T>::UnsignedBits twiddle_in(T key, bool greater)
{
  auto bits = reinterpret_cast<typename cub::Traits<T>::UnsignedBits&>(key);
  bits      = cub::Traits<T>::TwiddleIn(bits);
  if (greater) { bits = ~bits; }
  return bits;
}

template <typename T, int BitsPerPass>
__device__ int calc_bucket(T x, int start_bit, unsigned mask, bool greater)
{
  static_assert(BitsPerPass <= sizeof(int) * 8 - 1);  // so return type can be int
  return (twiddle_in(x, greater) >> start_bit) & mask;
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
};

/**
 * Fused filtering of the current phase and building histogram for the next phase
 * (see steps 4-1 in `radix_kernel` description).
 */
template <typename T, typename IdxT, int BitsPerPass>
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
    auto f = [greater, start_bit, mask](T value, IdxT) {
      int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
      atomicAdd(histogram_smem + bucket, IdxT(1));
    };
    vectorized_process(in_buf, len, f);
  } else {
    const IdxT previous_len      = counter->previous_len;
    const int want_bucket        = counter->bucket;
    IdxT& filter_cnt             = counter->filter_cnt;
    IdxT& out_cnt                = counter->out_cnt;
    const IdxT counter_len       = counter->len;
    const int previous_start_bit = calc_start_bit<T, BitsPerPass>(pass - 1);
    const unsigned previous_mask = calc_mask<T, BitsPerPass>(pass - 1);

    // See the remark above on the distributed execution of `f` using vectorized_process.
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
              counter_len](T value, IdxT i) {
      int prev_bucket =
        calc_bucket<T, BitsPerPass>(value, previous_start_bit, previous_mask, greater);
      if (prev_bucket == want_bucket) {
        IdxT pos     = atomicAdd(&filter_cnt, IdxT(1));
        out_buf[pos] = value;
        if (out_idx_buf) { out_idx_buf[pos] = in_idx_buf ? in_idx_buf[i] : i; }
        int bucket = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
        atomicAdd(histogram_smem + bucket, IdxT(1));

        if (counter_len == 1) {
          out[k - 1]     = value;
          out_idx[k - 1] = in_idx_buf ? in_idx_buf[i] : i;
        }
      } else if (prev_bucket < want_bucket) {
        IdxT pos     = atomicAdd(&out_cnt, IdxT(1));
        out[pos]     = value;
        out_idx[pos] = in_idx_buf ? in_idx_buf[i] : i;
      }
    };

    vectorized_process(in_buf, previous_len, f);
  }
  __syncthreads();

  // merge histograms produced by individual blocks
  for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
    if (histogram_smem[i] != 0) { atomicAdd(histogram + i, histogram_smem[i]); }
  }
}

/**
 * Replace a part of the histogram with its own prefix sum, starting from the `start` and adding
 * `current` to each entry of the result.
 * (step 2 in `radix_kernel` description)
 */
template <typename IdxT, int BitsPerPass, int BlockSize>
__device__ void scan(volatile IdxT* histogram,
                     const int start,
                     const int num_buckets,
                     const IdxT current)
{
  typedef cub::BlockScan<IdxT, BlockSize> BlockScan;
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

/**
 * Calculate in which bucket the k-th value will fall
 *  (steps 2-3 in `radix_kernel` description)
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
__device__ void choose_bucket(Counter<T, IdxT>* counter, IdxT* histogram, const IdxT k)
{
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  int index                 = threadIdx.x;
  IdxT last_prefix_sum      = 0;
  int num_pass              = 1;
  if constexpr (num_buckets >= BlockSize) {
    static_assert(num_buckets % BlockSize == 0);
    num_pass = num_buckets / BlockSize;
  }

  for (int i = 0; i < num_pass && (last_prefix_sum < k); i++) {
    // Turn the i-th chunk of the histogram into its prefix sum.
    scan<IdxT, BitsPerPass, BlockSize>(histogram, i * BlockSize, num_buckets, last_prefix_sum);
    if (index < num_buckets) {
      // Number of values in the previous `index-1` buckets (see the `scan` op above)
      IdxT prev = (index == 0) ? 0 : histogram[index - 1];
      // Number of values in `index` buckets
      IdxT cur = histogram[index];

      // one and only one thread will satisfy this condition, so only write once
      if (prev < k && cur >= k) {
        counter->k            = k - prev;  // how many values still are there to find
        counter->previous_len = counter->len;
        counter->len          = cur - prev;  // number of values in `index` bucket
        counter->bucket       = index;
      }
    }
    index += BlockSize;
    // this will break the loop when the counter is set (cur >= k), because last_prefix_sum >= cur
    last_prefix_sum = histogram[(i + 1) * BlockSize - 1];
  }
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
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
__global__ void __launch_bounds__(BlockSize) radix_kernel(const T* in_buf,
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

  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();
  constexpr int num_passes  = calc_num_passes<T, BitsPerPass>();
  const int batch_id        = blockIdx.y;
  in_buf += batch_id * len;
  out_buf += batch_id * len;
  out += batch_id * k;
  out_idx += batch_id * k;
  if (in_idx_buf) { in_idx_buf += batch_id * len; }
  if (out_idx_buf) { out_idx_buf += batch_id * len; }

  auto counter   = counters + batch_id;
  auto histogram = histograms + batch_id * num_buckets;

  filter_and_histogram<T, IdxT, BitsPerPass>(in_buf,
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
      counter->k            = k;
      counter->len          = len;
      counter->out_back_cnt = 0;
    }
    __syncthreads();

    IdxT ori_k = counter->k;

    if (counter->len > 0) {
      choose_bucket<T, IdxT, BitsPerPass, BlockSize>(counter, histogram, ori_k);
    }

    __syncthreads();
    if (pass == num_passes - 1) {
      const IdxT previous_len = counter->previous_len;
      const int want_bucket   = counter->bucket;
      int start_bit           = calc_start_bit<T, BitsPerPass>(pass);
      unsigned mask           = calc_mask<T, BitsPerPass>(pass);

      // radix topk
      IdxT& out_cnt = counter->out_cnt;
      for (IdxT i = threadIdx.x; i < previous_len; i += blockDim.x) {
        const T value = out_buf[i];
        int bucket    = calc_bucket<T, BitsPerPass>(value, start_bit, mask, greater);
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
    } else {
      // reset for next pass
      for (int i = threadIdx.x; i < num_buckets; i += blockDim.x) {
        histogram[i] = 0;
      }
      if (threadIdx.x == 0) { counter->filter_cnt = 0; }
    }
  }
}

/**
 * Calculate the minimal batch size, such that GPU is still fully occupied.
 */
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
inline dim3 get_optimal_grid_size(size_t req_batch_size, size_t len)
{
  int dev_id, sm_count, occupancy, max_grid_dim_y;
  RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_id));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&max_grid_dim_y, cudaDevAttrMaxGridDimY, dev_id));
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &occupancy, radix_kernel<T, IdxT, BitsPerPass, BlockSize>, BlockSize, 0));

  // number of block we'd use if the batch size is enough to occupy the gpu in any case
  size_t blocks_per_row = ceildiv<size_t>(len, BlockSize * ITEM_PER_THREAD);

  // fully occupy GPU
  size_t opt_batch_size = ceildiv<size_t>(sm_count * occupancy, blocks_per_row);
  // round it up to the closest pow-of-two for better data alignment
  opt_batch_size = isPo2(opt_batch_size) ? opt_batch_size : (1 << (log2(opt_batch_size) + 1));
  // Take a max possible pow-of-two grid_dim_y
  max_grid_dim_y = isPo2(max_grid_dim_y) ? max_grid_dim_y : (1 << log2(max_grid_dim_y));
  // If the optimal batch size is very small compared to the requested batch size, we know
  // the extra required memory is not significant and we can increase the batch size for
  // better occupancy when the grid size is not multiple of the SM count.
  // Also don't split the batch size when there is not much work overall.
  const size_t safe_enlarge_factor = 9;
  const size_t min_grid_size       = 1024;
  while ((opt_batch_size << safe_enlarge_factor) < req_batch_size ||
         blocks_per_row * opt_batch_size < min_grid_size) {
    opt_batch_size <<= 1;
  }

  // Do not exceed the max grid size.
  opt_batch_size = std::min<size_t>(opt_batch_size, size_t(max_grid_dim_y));
  // Don't do more work than needed
  opt_batch_size = std::min<size_t>(opt_batch_size, req_batch_size);
  // Let more blocks share one row if the required batch size is too small.
  while (opt_batch_size * blocks_per_row < size_t(sm_count * occupancy) &&
         // Ensure we still can read data somewhat efficiently
         len * sizeof(T) > 2 * VECTORIZED_READ_SIZE * BlockSize * blocks_per_row) {
    blocks_per_row <<= 1;
  }

  return dim3(blocks_per_row, opt_batch_size);
}

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
template <typename T, typename IdxT, int BitsPerPass, int BlockSize>
void radix_topk(const T* in,
                const IdxT* in_idx,
                size_t batch_size,
                size_t len,
                int k,
                T* out,
                IdxT* out_idx,
                bool select_min,
                rmm::cuda_stream_view stream)
{
  // reduce the block size if the input length is too small.
  if constexpr (BlockSize > calc_min_block_size<BitsPerPass>()) {
    if (BlockSize * ITEM_PER_THREAD > len) {
      return radix_topk<T, IdxT, BitsPerPass, BlockSize / 2>(
        in, in_idx, batch_size, len, k, out, out_idx, select_min, stream);
    }
  }

  // TODO: is it possible to relax this restriction?
  static_assert(calc_num_passes<T, BitsPerPass>() > 1);
  constexpr int num_buckets = calc_num_buckets<BitsPerPass>();

  dim3 blocks           = get_optimal_grid_size<T, IdxT, BitsPerPass, BlockSize>(batch_size, len);
  size_t max_chunk_size = blocks.y;

  rmm::device_uvector<Counter<T, IdxT>> counters(max_chunk_size, stream);
  rmm::device_uvector<IdxT> histograms(num_buckets * max_chunk_size, stream);
  rmm::device_uvector<T> buf1(len * max_chunk_size, stream);
  rmm::device_uvector<IdxT> idx_buf1(len * max_chunk_size, stream);
  rmm::device_uvector<T> buf2(len * max_chunk_size, stream);
  rmm::device_uvector<IdxT> idx_buf2(len * max_chunk_size, stream);

  for (size_t offset = 0; offset < batch_size; offset += max_chunk_size) {
    blocks.y = std::min(max_chunk_size, batch_size - offset);

    RAFT_CUDA_TRY(
      cudaMemsetAsync(counters.data(), 0, counters.size() * sizeof(Counter<T, IdxT>), stream));
    RAFT_CUDA_TRY(cudaMemsetAsync(histograms.data(), 0, histograms.size() * sizeof(IdxT), stream));

    const T* in_buf        = nullptr;
    const IdxT* in_idx_buf = nullptr;
    T* out_buf             = nullptr;
    IdxT* out_idx_buf      = nullptr;

    constexpr int num_passes = calc_num_passes<T, BitsPerPass>();

    for (int pass = 0; pass < num_passes; ++pass) {
      if (pass == 0) {
        in_buf      = in + offset * len;
        in_idx_buf  = nullptr;
        out_buf     = nullptr;
        out_idx_buf = nullptr;
      } else if (pass == 1) {
        in_buf      = in + offset * len;
        in_idx_buf  = in_idx ? in_idx + offset * len : nullptr;
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

      radix_kernel<T, IdxT, BitsPerPass, BlockSize>
        <<<blocks, BlockSize, 0, stream>>>(in_buf,
                                           in_idx_buf,
                                           out_buf,
                                           out_idx_buf,
                                           out + offset * k,
                                           out_idx + offset * k,
                                           counters.data(),
                                           histograms.data(),
                                           len,
                                           k,
                                           !select_min,
                                           pass);
      RAFT_CUDA_TRY(cudaPeekAtLastError());
    }
  }
}

}  // namespace raft::spatial::knn::detail::topk
