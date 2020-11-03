/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <raft/error.hpp>

#include <cuda_runtime.h>

#include <execinfo.h>
#include <chrono>
#include <cstdio>
#include <iomanip>
#include <iostream>

///@todo: enable once logging has been enabled in raft
//#include "logger.hpp"

namespace raft {

/**
 * @brief Exception thrown when a CUDA error is encountered.
 */
struct cuda_error : public raft::exception {
  explicit cuda_error(char const* const message) : raft::exception(message) {}
  explicit cuda_error(std::string const& message) : raft::exception(message) {}
};

}  // namespace raft

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 *
 */
#define CUDA_TRY(call)                                                        \
  do {                                                                        \
    cudaError_t const status = call;                                          \
    if (status != cudaSuccess) {                                              \
      cudaGetLastError();                                                     \
      std::string msg{};                                                      \
      SET_ERROR_MSG(                                                          \
        msg, "CUDA error encountered at: ", "call='%s', Reason=%s:%s", #call, \
        cudaGetErrorName(status), cudaGetErrorString(status));                \
      throw raft::cuda_error(msg);                                            \
    }                                                                         \
  } while (0)

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 */
#ifndef NDEBUG
#define CHECK_CUDA(stream) CUDA_TRY(cudaStreamSynchronize(stream));
#else
#define CHECK_CUDA(stream) CUDA_TRY(cudaPeekAtLastError());
#endif

/** FIXME: temporary alias for cuML compatibility */
#define CUDA_CHECK(call) CUDA_TRY(call)

///@todo: enable this only after we have added logging support in raft
// /**
//  * @brief check for cuda runtime API errors but log error instead of raising
//  *        exception.
//  */
#define CUDA_CHECK_NO_THROW(call)                                         \
  do {                                                                    \
    cudaError_t const status = call;                                      \
    if (cudaSuccess != status) {                                          \
      printf("CUDA call='%s' at file=%s line=%d failed with %s\n", #call, \
             __FILE__, __LINE__, cudaGetErrorString(status));             \
    }                                                                     \
  } while (0)

namespace raft {

/** Helper method to get to know warp size in device code */
__host__ __device__ constexpr inline int warp_size() { return 32; }

__host__ __device__ constexpr inline unsigned int warp_full_mask() {
  return 0xffffffff;
}

/**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to threads.
 */
class grid_1d_thread_t {
 public:
  int const block_size{0};
  int const num_blocks{0};

  /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   * @param elements_per_thread Typically, a single kernel thread processes more than a single
   * element; this affects the number of threads the grid must contain
   */
  grid_1d_thread_t(size_t overall_num_elements, size_t num_threads_per_block,
                   size_t max_num_blocks_1d, size_t elements_per_thread = 1)
    : block_size(num_threads_per_block),
      num_blocks(std::min((overall_num_elements +
                           (elements_per_thread * num_threads_per_block) - 1) /
                            (elements_per_thread * num_threads_per_block),
                          max_num_blocks_1d)) {
    RAFT_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
    RAFT_EXPECTS(num_threads_per_block / warp_size() > 0,
                 "num_threads_per_block / warp_size() must be > 0");
    RAFT_EXPECTS(elements_per_thread > 0, "elements_per_thread must be > 0");
  }
};

/**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to warps.
 */
class grid_1d_warp_t {
 public:
  int const block_size{0};
  int const num_blocks{0};

  /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   */
  grid_1d_warp_t(size_t overall_num_elements, size_t num_threads_per_block,
                 size_t max_num_blocks_1d)
    : block_size(num_threads_per_block),
      num_blocks(std::min(
        (overall_num_elements + (num_threads_per_block / warp_size()) - 1) /
          (num_threads_per_block / warp_size()),
        max_num_blocks_1d)) {
    RAFT_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
    RAFT_EXPECTS(num_threads_per_block / warp_size() > 0,
                 "num_threads_per_block / warp_size() must be > 0");
  }
};

/**
 * @brief A kernel grid configuration construction gadget for simple one-dimensional mapping
 * elements to blocks.
 */
class grid_1d_block_t {
 public:
  int const block_size{0};
  int const num_blocks{0};

  /**
   * @param overall_num_elements The number of elements the kernel needs to handle/process
   * @param num_threads_per_block The grid block size, determined according to the kernel's
   * specific features (amount of shared memory necessary, SM functional units use pattern etc.);
   * this can't be determined generically/automatically (as opposed to the number of blocks)
   */
  grid_1d_block_t(size_t overall_num_elements, size_t num_threads_per_block,
                  size_t max_num_blocks_1d)
    : block_size(num_threads_per_block),
      num_blocks(std::min(overall_num_elements, max_num_blocks_1d)) {
    RAFT_EXPECTS(overall_num_elements > 0, "overall_num_elements must be > 0");
    RAFT_EXPECTS(num_threads_per_block / warp_size() > 0,
                 "num_threads_per_block / warp_size() must be > 0");
  }
};

/**
 * @brief Generic copy method for all kinds of transfers
 * @tparam Type data type
 * @param dst destination pointer
 * @param src source pointer
 * @param len lenth of the src/dst buffers in terms of number of elements
 * @param stream cuda stream
 */
template <typename Type>
void copy(Type* dst, const Type* src, size_t len, cudaStream_t stream) {
  CUDA_CHECK(
    cudaMemcpyAsync(dst, src, len * sizeof(Type), cudaMemcpyDefault, stream));
}

/**
 * @defgroup Copy Copy methods
 * These are here along with the generic 'copy' method in order to improve
 * code readability using explicitly specified function names
 * @{
 */
/** performs a host to device copy */
template <typename Type>
void update_device(Type* d_ptr, const Type* h_ptr, size_t len,
                   cudaStream_t stream) {
  copy(d_ptr, h_ptr, len, stream);
}

/** performs a device to host copy */
template <typename Type>
void update_host(Type* h_ptr, const Type* d_ptr, size_t len,
                 cudaStream_t stream) {
  copy(h_ptr, d_ptr, len, stream);
}

template <typename Type>
void copy_async(Type* d_ptr1, const Type* d_ptr2, size_t len,
                cudaStream_t stream) {
  CUDA_CHECK(cudaMemcpyAsync(d_ptr1, d_ptr2, len * sizeof(Type),
                             cudaMemcpyDeviceToDevice, stream));
}
/** @} */

/**
 * @defgroup Debug Utils for debugging host/device buffers
 * @{
 */
template <class T, class OutStream>
void print_host_vector(const char* variable_name, const T* host_mem,
                       size_t componentsCount, OutStream& out) {
  out << variable_name << "=[";
  for (size_t i = 0; i < componentsCount; ++i) {
    if (i != 0) out << ",";
    out << host_mem[i];
  }
  out << "];\n";
}

template <class T, class OutStream>
void print_device_vector(const char* variable_name, const T* devMem,
                         size_t componentsCount, OutStream& out) {
  T* host_mem = new T[componentsCount];
  CUDA_CHECK(cudaMemcpy(host_mem, devMem, componentsCount * sizeof(T),
                        cudaMemcpyDeviceToHost));
  print_host_vector(variable_name, host_mem, componentsCount, out);
  delete[] host_mem;
}
/** @} */

/** cuda malloc */
template <typename Type>
void allocate(Type*& ptr, size_t len, bool setZero = false) {
  CUDA_CHECK(cudaMalloc((void**)&ptr, sizeof(Type) * len));
  if (setZero) CUDA_CHECK(cudaMemset(ptr, 0, sizeof(Type) * len));
}

/** helper method to get max usable shared mem per block parameter */
inline int get_shared_mem_per_block() {
  int dev_id;
  CUDA_CHECK(cudaGetDevice(&dev_id));
  int smem_per_blk;
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_per_blk,
                                    cudaDevAttrMaxSharedMemoryPerBlock, dev_id));
  return smem_per_blk;
}

/** helper method to get multi-processor count parameter */
inline int get_multi_processor_count() {
  int dev_id;
  CUDA_CHECK(cudaGetDevice(&dev_id));
  int mp_count;
  CUDA_CHECK(
    cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, dev_id));
  return mp_count;
}

/** helper method to convert an array on device to a string on host */
template <typename T>
std::string arr2str(const T* arr, int size, std::string name,
                    cudaStream_t stream, int width = 4) {
  std::stringstream ss;
  std::vector<T> arr_h(size);
  update_host(arr_h.data(), arr, size, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  ss << name << " = [ ";
  for (int i = 0; i < size; i++) {
    ss << std::setw(width) << arr_h[i];
    if (i < size - 1) ss << ", ";
  }
  ss << " ]" << std::endl;
  return ss.str();
}

/** this seems to be unused, but may be useful in the future */
template <typename T>
void assert_device_mem(T* ptr, std::string name) {
  cudaPointerAttributes s_att;
  cudaError_t s_err = cudaPointerGetAttributes(&s_att, ptr);
  if (s_err != 0 || s_att.device == -1) {
    std::cout << "Invalid device pointer encountered in " << name
              << ". device=" << s_att.device << ", err=" << s_err << std::endl;
  }
}

inline uint32_t cur_time_millis() {
  auto now = std::chrono::high_resolution_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::milliseconds>(duration)
    .count();
}

/** Helper function to calculate need memory for allocate to store dense matrix.
    * @param rows number of rows in matrix
    * @param columns number of columns in matrix
    * @return need number of items to allocate via allocate()
    * @sa allocate()
    */
inline size_t alloc_length_for_matrix(size_t rows, size_t columns) {
  return rows * columns;
}

/** Helper function to check alignment of pointer.
    * @param ptr the pointer to check
    * @param alignment to be checked for
    * @return true if address in bytes is a multiple of alignment
    */
template <typename Type>
bool is_aligned(Type* ptr, size_t alignment) {
  return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

/** calculate greatest common divisor of two numbers
* @a integer
* @b integer
* @ return gcd of a and b
*/
template <typename IntType>
IntType gcd(IntType a, IntType b) {
  while (b != 0) {
    IntType tmp = b;
    b = a % b;
    a = tmp;
  }
  return a;
}

}  // namespace raft
