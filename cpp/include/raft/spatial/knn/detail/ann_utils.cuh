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
#include <raft/cudart_utils.h>
#include <raft/distance/distance.cuh>
#include <raft/distance/distance_type.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace raft::spatial::knn::detail::utils {

/** Whether pointers are accessible on the device or on the host. */
enum class pointer_residency {
  /** Some of the pointers are on the device, some on the host. */
  mixed,
  /** All pointers accessible from both the device and the host. */
  host_and_device,
  /** All pointers are host accessible. */
  host_only,
  /** All poitners are device accessible. */
  device_only
};

template <typename... Types>
struct pointer_residency_count {
};

template <>
struct pointer_residency_count<> {
  static inline auto run() -> std::tuple<int, int> { return std::make_tuple(0, 0); }
};

template <typename Type, typename... Types>
struct pointer_residency_count<Type, Types...> {
  static inline auto run(const Type* ptr, const Types*... ptrs) -> std::tuple<int, int>
  {
    auto [on_device, on_host] = pointer_residency_count<Types...>::run(ptrs...);
    cudaPointerAttributes attr;
    RAFT_CUDA_TRY(cudaPointerGetAttributes(&attr, ptr));
    switch (attr.type) {
      case cudaMemoryTypeUnregistered:
      case cudaMemoryTypeHost: return std::make_tuple(on_device, on_host + 1);
      case cudaMemoryTypeDevice: return std::make_tuple(on_device + 1, on_host);
      case cudaMemoryTypeManaged: return std::make_tuple(on_device + 1, on_host + 1);
      default: return std::make_tuple(on_device, on_host);
    }
  }
};

/** Check if all argument pointers reside on the host or on the device. */
template <typename... Types>
auto check_pointer_residency(const Types*... ptrs) -> pointer_residency
{
  auto [on_device, on_host] = pointer_residency_count<Types...>::run(ptrs...);
  int n_args                = sizeof...(Types);
  if (on_device == n_args && on_host == n_args) { return pointer_residency::host_and_device; }
  if (on_device == n_args) { return pointer_residency::device_only; }
  if (on_host == n_args) { return pointer_residency::host_only; }
  return pointer_residency::mixed;
}

template <typename T>
struct config {
};

template <>
struct config<float> {
  using value_t                    = float;
  static constexpr double kDivisor = 1.0;
};
template <>
struct config<uint8_t> {
  using value_t                    = uint32_t;
  static constexpr double kDivisor = 256.0;
};
template <>
struct config<int8_t> {
  using value_t                    = int32_t;
  static constexpr double kDivisor = 128.0;
};

/**
 * @brief Converting values between the types taking into account scaling factors
 * for the integral types.
 *
 * @tparam T target type of the mapping.
 */
template <typename T>
struct mapping {
  /**
   * @defgroup
   * @brief Cast and possibly scale a value of the source type `S` to the target type `T`.
   *
   * @tparam S source type
   * @param x source value
   * @{
   */
  template <typename S>
  HDI auto operator()(const S& x) -> std::enable_if_t<std::is_same_v<S, T>, T>
  {
    return x;
  };

  template <typename S>
  HDI auto operator()(const S& x) -> std::enable_if_t<!std::is_same_v<S, T>, T>
  {
    constexpr double kMult = config<T>::kDivisor / config<S>::kDivisor;
    if constexpr (std::is_floating_point_v<S>) { return static_cast<T>(x * static_cast<S>(kMult)); }
    if constexpr (std::is_floating_point_v<T>) { return static_cast<T>(x) * static_cast<T>(kMult); }
    return static_cast<T>(static_cast<float>(x) * static_cast<float>(kMult));
  };
  /** @} */
};

/**
 * @brief Sets the first num bytes of the block of memory pointed by ptr to the specified value.
 *
 * @param[out] ptr host or device pointer
 * @param[in] value
 * @param[in] n_bytes
 */
template <typename T>
inline void memzero(T* ptr, size_t n_elems, rmm::cuda_stream_view stream)
{
  switch (check_pointer_residency(ptr)) {
    case pointer_residency::host_and_device:
    case pointer_residency::device_only: {
      RAFT_CUDA_TRY(cudaMemsetAsync(ptr, 0, n_elems * sizeof(T), stream));
    } break;
    case pointer_residency::host_only: {
      stream.synchronize();
      ::memset(ptr, 0, n_elems * sizeof(T));
    } break;
    default: RAFT_FAIL("memset: unreachable code");
  }
}

__global__ void argmin_along_rows_kernel(uint32_t n_rows,
                                         uint32_t n_cols,
                                         const float* a,
                                         uint32_t* out)
{
  __shared__ uint32_t shm_ids[1024];  // NOLINT
  __shared__ float shm_vals[1024];    // NOLINT
  uint32_t i = blockIdx.x;
  if (i >= n_rows) return;
  uint32_t min_idx = n_cols;
  float min_val    = raft::upper_bound<float>();
  for (uint32_t j = threadIdx.x; j < n_cols; j += blockDim.x) {
    if (min_val > a[j + n_cols * i]) {
      min_val = a[j + n_cols * i];
      min_idx = j;
    }
  }
  shm_vals[threadIdx.x] = min_val;
  shm_ids[threadIdx.x]  = min_idx;
  __syncthreads();
  for (uint32_t offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if (threadIdx.x < offset) {
      if (shm_vals[threadIdx.x] < shm_vals[threadIdx.x + offset]) {
      } else if (shm_vals[threadIdx.x] > shm_vals[threadIdx.x + offset]) {
        shm_vals[threadIdx.x] = shm_vals[threadIdx.x + offset];
        shm_ids[threadIdx.x]  = shm_ids[threadIdx.x + offset];
      } else if (shm_ids[threadIdx.x] > shm_ids[threadIdx.x + offset]) {
        shm_ids[threadIdx.x] = shm_ids[threadIdx.x + offset];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { out[i] = shm_ids[0]; }
}

/**
 * @brief Find index of the smallest element in each row.
 *
 * NB: device-only function
 * TODO: specialize select_k for the case of `k == 1` and use that one instead.
 *
 * @param n_rows
 * @param n_cols
 * @param[in] a device pointer to the row-major matrix [n_rows, n_cols]
 * @param[out] out device pointer to the vector of selected indices [n_rows]
 * @param stream
 */
inline void argmin_along_rows(
  uint32_t n_rows, uint32_t n_cols, const float* a, uint32_t* out, rmm::cuda_stream_view stream)
{
  uint32_t block_dim = 1024;
  while (block_dim > n_cols) {
    block_dim /= 2;
  }
  block_dim = max(block_dim, 128);
  argmin_along_rows_kernel<<<n_rows, block_dim, 0, stream>>>(n_rows, n_cols, a, out);
}

__global__ void dots_along_rows_kernel(uint32_t n_rows, uint32_t n_cols, const float* a, float* out)
{
  uint64_t i = threadIdx.y + (blockDim.y * blockIdx.x);
  if (i >= n_rows) return;

  float sqsum = 0.0;
  for (uint64_t j = threadIdx.x; j < n_cols; j += blockDim.x) {
    float val = a[j + (n_cols * i)];
    sqsum += val * val;
  }
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 1);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 2);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 4);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 8);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 16);
  if (threadIdx.x == 0) { out[i] = sqsum; }
}

/**
 * @brief Square sum of values in each row (row-major matrix).
 *
 * NB: device-only function
 *
 * @param n_rows
 * @param n_cols
 * @param[in] a device pointer to the row-major matrix [n_rows, n_cols]
 * @param[out] out device pointer to the vector of dot-products [n_rows]
 * @param stream
 */
inline void dots_along_rows(
  uint32_t n_rows, uint32_t n_cols, const float* a, float* out, rmm::cuda_stream_view stream)
{
  dim3 threads(32, 4, 1);
  dim3 blocks(ceildiv(n_rows, threads.y), 1, 1);
  dots_along_rows_kernel<<<blocks, threads, 0, stream>>>(n_rows, n_cols, a, out);
  /**
   * TODO: this can be replaced with the rowNorm helper as shown below.
   * However, the rowNorm helper seems to incur a significant performance penalty
   * (example case ann-search slowed down from 150ms to 186ms).
   *
   * raft::linalg::rowNorm(out, a, n_cols, n_rows, raft::linalg::L2Norm, true, stream);
   */
}

template <typename T>
__global__ void accumulate_into_selected_kernel(uint32_t n_rows,
                                                uint32_t n_cols,
                                                float* output,
                                                uint32_t* selection_counters,
                                                const T* input,
                                                const uint32_t* row_ids)
{
  uint64_t gid = threadIdx.x + (blockDim.x * blockIdx.x);
  uint64_t j   = gid % n_cols;
  uint64_t i   = gid / n_cols;
  if (i >= n_rows) return;
  uint64_t l = row_ids[i];
  if (j == 0) { atomicAdd(&(selection_counters[l]), 1); }
  atomicAdd(&(output[j + n_cols * l]), mapping<float>{}(input[gid]));
}

/**
 * @brief Add all rows of input matrix into a selection of rows in the output matrix
 * (cast and possibly scale the data input type). Count the number of times every output
 * row was selected along the way.
 *
 * @tparam T
 *
 * @param n_cols number of columns in all matrices
 * @param[out] output output matrix [..., n_cols]
 * @param[inout] selection_counters number of occurrences of each row id in row_ids [..., n_cols]
 * @param n_rows number of rows in the input
 * @param[in] input row-major input matrix [n_rows, n_cols]
 * @param[in] row_ids row indices in the output matrix [n_rows]
 */
template <typename T>
void accumulate_into_selected(size_t n_rows,
                              uint32_t n_cols,
                              float* output,
                              uint32_t* selection_counters,
                              const T* input,
                              const uint32_t* row_ids,
                              rmm::cuda_stream_view stream)
{
  switch (check_pointer_residency(output, input, selection_counters, row_ids)) {
    case pointer_residency::host_and_device:
    case pointer_residency::device_only: {
      uint32_t block_dim = 128;
      auto grid_dim =
        static_cast<uint32_t>(ceildiv<size_t>(n_rows * static_cast<size_t>(n_cols), block_dim));
      accumulate_into_selected_kernel<T><<<grid_dim, block_dim, 0, stream>>>(
        n_rows, n_cols, output, selection_counters, input, row_ids);
    } break;
    case pointer_residency::host_only: {
      stream.synchronize();
      for (size_t i = 0; i < n_rows; i++) {
        uint32_t l = row_ids[i];
        selection_counters[l]++;
        for (uint32_t j = 0; j < n_cols; j++) {
          output[j + n_cols * l] += mapping<float>{}(input[j + n_cols * i]);
        }
      }
      stream.synchronize();
    } break;
    default: RAFT_FAIL("All pointers must reside on the same side, host or device.");
  }
}

__global__ void normalize_rows_kernel(uint32_t n_rows, uint32_t n_cols, float* a)
{
  uint64_t i = threadIdx.y + (blockDim.y * blockIdx.x);
  if (i >= n_rows) return;

  float sqsum = 0.0;
  for (uint32_t j = threadIdx.x; j < n_cols; j += blockDim.x) {
    float val = a[j + (n_cols * i)];
    sqsum += val * val;
  }
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 1);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 2);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 4);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 8);
  sqsum += __shfl_xor_sync(0xffffffff, sqsum, 16);
  if (sqsum <= 1e-8) return;
  sqsum = rsqrtf(sqsum);  // reciprocal of the square root
  for (uint32_t j = threadIdx.x; j < n_cols; j += blockDim.x) {
    a[j + n_cols * i] *= sqsum;
  }
}

/**
 * @brief Divide rows by their L2 norm (square root of sum of squares).
 *
 * NB: device-only function
 *
 * @param[in] n_rows
 * @param[in] n_cols
 * @param[inout] a device pointer to a row-major matrix [n_rows, n_cols]
 * @param stream
 */
inline void normalize_rows(uint32_t n_rows, uint32_t n_cols, float* a, rmm::cuda_stream_view stream)
{
  dim3 threads(32, 4, 1);  // DO NOT CHANGE
  dim3 blocks(ceildiv(n_rows, threads.y), 1, 1);
  normalize_rows_kernel<<<blocks, threads, 0, stream>>>(n_rows, n_cols, a);
}

template <typename Lambda>
__global__ void map_along_rows_kernel(
  uint32_t n_rows, uint32_t n_cols, float* a, const uint32_t* d, Lambda map)
{
  uint64_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  uint64_t i   = gid / n_cols;
  if (i >= n_rows) return;
  float& x = a[gid];
  x        = map(x, d[i]);
}

/**
 * @brief Map a binary function over a matrix and a vector element-wise, broadcasting the vector
 * values along rows: `m[i, j] = op(m[i,j], v[i])`
 *
 * NB: device-only function
 *
 * @tparam Lambda
 *
 * @param n_rows
 * @param n_cols
 * @param[inout] m device pointer to a row-major matrix [n_rows, n_cols]
 * @param[in] v device pointer to a vector [n_rows]
 * @param op the binary operation to apply on every element of matrix rows and of the vector
 */
template <typename Lambda>
inline void map_along_rows(uint32_t n_rows,
                           uint32_t n_cols,
                           float* m,
                           const uint32_t* v,
                           Lambda op,
                           rmm::cuda_stream_view stream)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(
    ceildiv<uint64_t>(static_cast<uint64_t>(n_rows) * static_cast<uint64_t>(n_cols), threads.x),
    1,
    1);
  map_along_rows_kernel<<<blocks, threads, 0, stream>>>(n_rows, n_cols, m, v, op);
}

template <typename T>
__global__ void outer_add_kernel(const T* a, uint32_t len_a, const T* b, uint32_t len_b, T* c)
{
  uint64_t gid = threadIdx.x + blockDim.x * blockIdx.x;
  uint64_t i   = gid / len_b;
  uint64_t j   = gid % len_b;
  if (i >= len_a) return;
  c[gid] = (a == nullptr ? T(0) : a[i]) + (b == nullptr ? T(0) : b[j]);
}

template <typename T, typename IdxT>
__global__ void block_copy_kernel(const IdxT* in_offsets,
                                  const IdxT* out_offsets,
                                  IdxT n_blocks,
                                  const T* in_data,
                                  T* out_data,
                                  IdxT n_mult)
{
  IdxT i = IdxT(blockDim.x) * IdxT(blockIdx.x) + threadIdx.x;
  // find the source offset using the binary search.
  uint32_t l     = 0;
  uint32_t r     = n_blocks;
  IdxT in_offset = 0;
  if (in_offsets[r] * n_mult <= i) return;
  while (l + 1 < r) {
    uint32_t c = (l + r) >> 1;
    IdxT o     = in_offsets[c] * n_mult;
    if (o <= i) {
      l         = c;
      in_offset = o;
    } else {
      r = c;
    }
  }
  // copy the data
  out_data[out_offsets[l] * n_mult - in_offset + i] = in_data[i];
}

/**
 * Copy chunks of data from one array to another at given offsets.
 *
 * @tparam T element type
 * @tparam IdxT index type
 *
 * @param[in] in_offsets
 * @param[in] out_offsets
 * @param n_blocks size of the offset arrays minus one.
 * @param[in] in_data
 * @param[out] out_data
 * @param n_mult constant multiplier for offset values (such as e.g. `dim`)
 * @param stream
 */
template <typename T, typename IdxT>
void block_copy(const IdxT* in_offsets,
                const IdxT* out_offsets,
                IdxT n_blocks,
                const T* in_data,
                T* out_data,
                IdxT n_mult,
                rmm::cuda_stream_view stream)
{
  IdxT in_size;
  update_host(&in_size, in_offsets + n_blocks, 1, stream);
  stream.synchronize();
  dim3 threads(128, 1, 1);
  dim3 blocks(ceildiv<IdxT>(in_size * n_mult, threads.x), 1, 1);
  block_copy_kernel<<<blocks, threads, 0, stream>>>(
    in_offsets, out_offsets, n_blocks, in_data, out_data, n_mult);
}

/**
 * @brief Fill matrix `c` with all combinations of sums of vectors `a` and `b`.
 *
 * NB: device-only function
 *
 * @tparam T element type
 *
 * @param[in] a device pointer to a vector [len_a]
 * @param len_a number of elements in `a`
 * @param[in] b device pointer to a vector [len_b]
 * @param len_b number of elements in `b`
 * @param[out] c row-major matrix [len_a, len_b]
 * @param stream
 */
template <typename T>
void outer_add(
  const T* a, uint32_t len_a, const T* b, uint32_t len_b, T* c, rmm::cuda_stream_view stream)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(
    ceildiv<uint64_t>(static_cast<uint64_t>(len_a) * static_cast<uint64_t>(len_b), threads.x),
    1,
    1);
  outer_add_kernel<<<blocks, threads, 0, stream>>>(a, len_a, b, len_b, c);
}

template <typename T, typename S, typename IdxT>
__global__ void copy_selected_kernel(uint64_t n_rows,
                                     uint64_t n_cols,
                                     const S* src,
                                     const IdxT* row_ids,
                                     uint64_t ld_src,
                                     T* dst,
                                     uint64_t ld_dst)
{
  uint64_t gid   = threadIdx.x + uint64_t{blockDim.x} * uint64_t{blockIdx.x};
  uint64_t j     = gid % n_cols;
  uint64_t i_dst = gid / n_cols;
  if (i_dst >= n_rows) return;
  auto i_src              = static_cast<uint64_t>(row_ids[i_dst]);
  dst[ld_dst * i_dst + j] = mapping<T>{}(src[ld_src * i_src + j]);
}

/**
 * @brief Copy selected rows of a matrix while mapping the data from the source to the target
 * type.
 *
 * @tparam T target type
 * @tparam S source type
 *
 * @param n_rows
 * @param n_cols
 * @param[in] src input matrix [..., ld_src]
 * @param[in] row_ids selection of rows to be copied [n_rows]
 * @param ld_src number of cols in the input (ld_src >= n_cols)
 * @param[out] dst output matrix [n_rows, ld_dst]
 * @param ld_dst number of cols in the output (ld_dst >= n_cols)
 * @param stream
 */
template <typename T, typename S, typename IdxT>
void copy_selected(uint64_t n_rows,
                   uint64_t n_cols,
                   const S* src,
                   const IdxT* row_ids,
                   uint64_t ld_src,
                   T* dst,
                   uint64_t ld_dst,
                   rmm::cuda_stream_view stream)
{
  switch (check_pointer_residency(src, dst, row_ids)) {
    case pointer_residency::host_and_device:
    case pointer_residency::device_only: {
      uint64_t block_dim = 128;
      uint64_t grid_dim  = ceildiv(n_rows * n_cols, block_dim);
      copy_selected_kernel<T, S>
        <<<grid_dim, block_dim, 0, stream>>>(n_rows, n_cols, src, row_ids, ld_src, dst, ld_dst);
    } break;
    case pointer_residency::host_only: {
      stream.synchronize();
      for (uint64_t i_dst = 0; i_dst < n_rows; i_dst++) {
        auto i_src = static_cast<uint64_t>(row_ids[i_dst]);
        for (uint64_t j = 0; j < n_cols; j++) {
          dst[ld_dst * i_dst + j] = mapping<T>{}(src[ld_src * i_src + j]);
        }
      }
      stream.synchronize();
    } break;
    default: RAFT_FAIL("All pointers must reside on the same side, host or device.");
  }
}
}  // namespace raft::spatial::knn::detail::utils
