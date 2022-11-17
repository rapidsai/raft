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

#include <raft/distance/distance.cuh>
#include <raft/distance/distance_types.hpp>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

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
      case cudaMemoryTypeUnregistered: return std::make_tuple(on_device, on_host + 1);
      case cudaMemoryTypeHost:
        return std::make_tuple(on_device + int(attr.devicePointer == ptr), on_host + 1);
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

/** RAII helper to access the host data from gpu when necessary. */
template <typename PtrT, typename Action>
struct with_mapped_memory_t {
  with_mapped_memory_t(PtrT ptr, size_t size, Action action) : action_(action)
  {
    if (ptr == nullptr) { return; }
    switch (utils::check_pointer_residency(ptr)) {
      case utils::pointer_residency::device_only:
      case utils::pointer_residency::host_and_device: {
        dev_ptr_ = (void*)ptr;  // NOLINT
      } break;
      default: {
        host_ptr_ = (void*)ptr;  // NOLINT
        RAFT_CUDA_TRY(cudaHostRegister(host_ptr_, size, choose_flags(ptr)));
        RAFT_CUDA_TRY(cudaHostGetDevicePointer(&dev_ptr_, host_ptr_, 0));
      } break;
    }
  }

  ~with_mapped_memory_t()
  {
    if (host_ptr_ != nullptr) { cudaHostUnregister(host_ptr_); }
  }

  auto operator()() { return action_((PtrT)dev_ptr_); }  // NOLINT

 private:
  Action action_;
  void* host_ptr_ = nullptr;
  void* dev_ptr_  = nullptr;

  template <typename T>
  static auto choose_flags(const T*) -> unsigned int
  {
    int dev_id, readonly_supported;
    RAFT_CUDA_TRY(cudaGetDevice(&dev_id));
    RAFT_CUDA_TRY(cudaDeviceGetAttribute(
      &readonly_supported, cudaDevAttrHostRegisterReadOnlySupported, dev_id));
    if (readonly_supported) {
      return cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    } else {
      return cudaHostRegisterMapped;
    }
  }

  template <typename T>
  static auto choose_flags(T*) -> unsigned int
  {
    return cudaHostRegisterMapped;
  }
};

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
  HDI auto operator()(const S& x) const -> std::enable_if_t<std::is_same_v<S, T>, T>
  {
    return x;
  };

  template <typename S>
  HDI auto operator()(const S& x) const -> std::enable_if_t<!std::is_same_v<S, T>, T>
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
template <typename T, typename IdxT>
inline void memzero(T* ptr, IdxT n_elems, rmm::cuda_stream_view stream)
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

template <typename IdxT>
__global__ void dots_along_rows_kernel(IdxT n_rows, IdxT n_cols, const float* a, float* out)
{
  IdxT i = threadIdx.y + (blockDim.y * static_cast<IdxT>(blockIdx.x));
  if (i >= n_rows) return;

  float sqsum = 0.0;
  for (IdxT j = threadIdx.x; j < n_cols; j += blockDim.x) {
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
 * @tparam IdxT index type
 *
 * @param n_rows
 * @param n_cols
 * @param[in] a device pointer to the row-major matrix [n_rows, n_cols]
 * @param[out] out device pointer to the vector of dot-products [n_rows]
 * @param stream
 */
template <typename IdxT>
inline void dots_along_rows(
  IdxT n_rows, IdxT n_cols, const float* a, float* out, rmm::cuda_stream_view stream)
{
  dim3 threads(32, 4, 1);
  dim3 blocks(ceildiv<IdxT>(n_rows, threads.y), 1, 1);
  dots_along_rows_kernel<IdxT><<<blocks, threads, 0, stream>>>(n_rows, n_cols, a, out);
  /**
   * TODO: this can be replaced with the rowNorm helper as shown below.
   * However, the rowNorm helper seems to incur a significant performance penalty
   * (example case ann-search slowed down from 150ms to 186ms).
   *
   * raft::linalg::rowNorm(out, a, n_cols, n_rows, raft::linalg::L2Norm, true, stream);
   */
}

template <typename T, typename IdxT>
__global__ void outer_add_kernel(const T* a, IdxT len_a, const T* b, IdxT len_b, T* c)
{
  IdxT gid = threadIdx.x + blockDim.x * static_cast<IdxT>(blockIdx.x);
  IdxT i   = gid / len_b;
  IdxT j   = gid % len_b;
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
  IdxT i = static_cast<IdxT>(blockDim.x) * static_cast<IdxT>(blockIdx.x) + threadIdx.x;
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
 * @tparam T    element type
 * @tparam IdxT index type
 *
 * @param[in] a device pointer to a vector [len_a]
 * @param len_a number of elements in `a`
 * @param[in] b device pointer to a vector [len_b]
 * @param len_b number of elements in `b`
 * @param[out] c row-major matrix [len_a, len_b]
 * @param stream
 */
template <typename T, typename IdxT>
void outer_add(const T* a, IdxT len_a, const T* b, IdxT len_b, T* c, rmm::cuda_stream_view stream)
{
  dim3 threads(128, 1, 1);
  dim3 blocks(ceildiv<IdxT>(len_a * len_b, threads.x), 1, 1);
  outer_add_kernel<<<blocks, threads, 0, stream>>>(a, len_a, b, len_b, c);
}

template <typename T, typename S, typename IdxT, typename LabelT>
__global__ void copy_selected_kernel(
  IdxT n_rows, IdxT n_cols, const S* src, const LabelT* row_ids, IdxT ld_src, T* dst, IdxT ld_dst)
{
  IdxT gid   = threadIdx.x + blockDim.x * static_cast<IdxT>(blockIdx.x);
  IdxT j     = gid % n_cols;
  IdxT i_dst = gid / n_cols;
  if (i_dst >= n_rows) return;
  auto i_src              = static_cast<IdxT>(row_ids[i_dst]);
  dst[ld_dst * i_dst + j] = mapping<T>{}(src[ld_src * i_src + j]);
}

/**
 * @brief Copy selected rows of a matrix while mapping the data from the source to the target
 * type.
 *
 * @tparam T      target type
 * @tparam S      source type
 * @tparam IdxT   index type
 * @tparam LabelT label type
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
template <typename T, typename S, typename IdxT, typename LabelT>
void copy_selected(IdxT n_rows,
                   IdxT n_cols,
                   const S* src,
                   const LabelT* row_ids,
                   IdxT ld_src,
                   T* dst,
                   IdxT ld_dst,
                   rmm::cuda_stream_view stream)
{
  switch (check_pointer_residency(src, dst, row_ids)) {
    case pointer_residency::host_and_device:
    case pointer_residency::device_only: {
      IdxT block_dim = 128;
      IdxT grid_dim  = ceildiv(n_rows * n_cols, block_dim);
      copy_selected_kernel<T, S>
        <<<grid_dim, block_dim, 0, stream>>>(n_rows, n_cols, src, row_ids, ld_src, dst, ld_dst);
    } break;
    case pointer_residency::host_only: {
      stream.synchronize();
      for (IdxT i_dst = 0; i_dst < n_rows; i_dst++) {
        auto i_src = static_cast<IdxT>(row_ids[i_dst]);
        for (IdxT j = 0; j < n_cols; j++) {
          dst[ld_dst * i_dst + j] = mapping<T>{}(src[ld_src * i_src + j]);
        }
      }
      stream.synchronize();
    } break;
    default: RAFT_FAIL("All pointers must reside on the same side, host or device.");
  }
}

}  // namespace raft::spatial::knn::detail::utils
