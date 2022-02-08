/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <utility>

namespace raft {
namespace mr {

/**
 * @brief Base for all RAII-based owning of temporary memory allocations. This
 *        class should ideally not be used by users directly, but instead via
 *        the child classes `device_buffer` and `host_buffer`.
 *
 * @tparam T          data type
 * @tparam AllocatorT The underly allocator object
 */
template <typename T, typename AllocatorT>
class buffer_base {
 public:
  using size_type       = std::size_t;
  using value_type      = T;
  using iterator        = value_type*;
  using const_iterator  = const value_type*;
  using reference       = T&;
  using const_reference = const T&;

  buffer_base() = delete;

  buffer_base(const buffer_base& other) = delete;

  buffer_base& operator=(const buffer_base& other) = delete;

  /**
   * @brief Main ctor
   *
   * @param[in] allocator asynchronous allocator used for managing buffer life
   * @param[in] stream    cuda stream where this allocation operations are async
   * @param[in] n         size of the buffer (in number of elements)
   */
  buffer_base(std::shared_ptr<AllocatorT> allocator, cudaStream_t stream, size_type n = 0)
    : data_(nullptr), size_(n), capacity_(n), stream_(stream), allocator_(std::move(allocator))
  {
    if (capacity_ > 0) {
      data_ =
        static_cast<value_type*>(allocator_->allocate(capacity_ * sizeof(value_type), stream_));
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream_));
    }
  }

  ~buffer_base() { release(); }

  value_type* data() { return data_; }

  const value_type* data() const { return data_; }

  size_type size() const { return size_; }

  void clear() { size_ = 0; }

  iterator begin() { return data_; }

  const_iterator begin() const { return data_; }

  iterator end() { return data_ + size_; }

  const_iterator end() const { return data_ + size_; }

  /**
   * @brief Reserve new memory size for this buffer.
   *
   * It re-allocates a fresh buffer if the new requested capacity is more than
   * the current one, copies the old buffer contents to this new buffer and
   * removes the old one.
   *
   * @param[in] new_capacity new capacity (in number of elements)
   * @{
   */
  void reserve(size_type new_capacity)
  {
    if (new_capacity > capacity_) {
      auto* new_data =
        static_cast<value_type*>(allocator_->allocate(new_capacity * sizeof(value_type), stream_));
      if (size_ > 0) { raft::copy(new_data, data_, size_, stream_); }
      // Only deallocate if we have allocated a pointer
      if (nullptr != data_) {
        allocator_->deallocate(data_, capacity_ * sizeof(value_type), stream_);
      }
      data_     = new_data;
      capacity_ = new_capacity;
    }
  }

  void reserve(size_type new_capacity, cudaStream_t stream)
  {
    set_stream(stream);
    reserve(new_capacity);
  }
  /** @} */

  /**
   * @brief Resize the underlying buffer (uses `reserve` method internally)
   *
   * @param[in] new_size new buffer size
   * @{
   */
  void resize(const size_type new_size)
  {
    reserve(new_size);
    size_ = new_size;
  }

  void resize(const size_type new_size, cudaStream_t stream)
  {
    set_stream(stream);
    resize(new_size);
  }
  /** @} */

  /**
   * @brief Deletes the underlying buffer
   *
   * If this method is not explicitly called, it will be during the destructor
   * @{
   */
  void release()
  {
    if (nullptr != data_) {
      allocator_->deallocate(data_, capacity_ * sizeof(value_type), stream_);
    }
    data_     = nullptr;
    capacity_ = 0;
    size_     = 0;
  }

  void release(cudaStream_t stream)
  {
    set_stream(stream);
    release();
  }
  /** @} */

  /**
   * @brief returns the underlying allocator used
   *
   * @return the allocator pointer
   */
  std::shared_ptr<AllocatorT> get_allocator() const { return allocator_; }

  /**
   * @brief returns the underlying stream used
   *
   * @return the cuda stream
   */
  cudaStream_t get_stream() const { return stream_; }

 protected:
  value_type* data_;

 private:
  size_type size_;
  size_type capacity_;
  cudaStream_t stream_;
  std::shared_ptr<AllocatorT> allocator_;

  /**
   * @brief Sets a new cuda stream where the future operations will be queued
   *
   * This method makes sure that the inter-stream dependencies are met and taken
   * care of, before setting the input stream as a new stream for this buffer.
   * Ideally, the same cuda stream passed during constructor is expected to be
   * used throughout this buffer's lifetime, for performance.
   *
   * @param[in] stream new cuda stream to be set. If it is the same as the
   *                   current one, then this method will be a no-op.
   */
  void set_stream(cudaStream_t stream)
  {
    if (stream_ != stream) {
      cudaEvent_t event;
      RAFT_CUDA_TRY(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      RAFT_CUDA_TRY(cudaEventRecord(event, stream_));
      RAFT_CUDA_TRY(cudaStreamWaitEvent(stream, event, 0));
      stream_ = stream;
      RAFT_CUDA_TRY(cudaEventDestroy(event));
    }
  }
};  // class buffer_base

};  // namespace mr
};  // namespace raft
