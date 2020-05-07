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

#include <cuda_runtime.h>
#include <memory>
#include <utility>
#include "cudart_utils.h"

namespace raft {

/**
 * @brief Base for all RAII-based owning of temporary memory allocations. This
 *        class should ideally not be used by users directly, but instead via
 *        the child classes `device_buffer` and `host_buffer`.
 */
template <typename T, typename Allocator>
class buffer_base {
 public:
  using size_type = std::size_t;
  using value_type = T;
  using iterator = value_type*;
  using const_iterator = const value_type*;
  using reference = T&;
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
  buffer_base(std::shared_ptr<Allocator> allocator, cudaStream_t stream,
              size_type n = 0)
    : size_(n),
      capacity_(n),
      data_(nullptr),
      stream_(stream),
      allocator_(std::move(allocator)) {
    if (capacity_ > 0) {
      data_ = static_cast<value_type*>(
        allocator_->allocate(capacity_ * sizeof(value_type), stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
    }
  }

  ~buffer_base() {
    if (nullptr != data_) {
      allocator_->deallocate(data_, capacity_ * sizeof(value_type), stream_);
    }
  }

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
   * @param[in] stream       cuda stream where allocation operations are queued
   */
  void reserve(const size_type new_capacity, cudaStream_t stream) {
    set_stream(stream);
    if (new_capacity > capacity_) {
      auto* new_data = static_cast<value_type*>(
        allocator_->allocate(new_capacity * sizeof(value_type), stream_));
      if (size_ > 0) {
        CUDA_CHECK(cudaMemcpyAsync(new_data, data_, size_ * sizeof(value_type),
                                   cudaMemcpyDefault, stream_));
      }
      if (nullptr != data_) {
        allocator_->deallocate(data_, capacity_ * sizeof(value_type), stream_);
      }
      data_ = new_data;
      capacity_ = new_capacity;
    }
  }

  /**
   * @brief Resize the underlying buffer (uses `reserve` method internally)
   *
   * @param[in] new_size new buffer size
   * @param[in] stream   cuda stream where the work will be queued
   */
  void resize(const size_type new_size, cudaStream_t stream) {
    reserve(new_size, stream);
    size_ = new_size;
  }

  /**
   * @brief Deletes the underlying buffer
   *
   * If this method is not explicitly called, it will be during the destructor
   *
   * @param[in] stream   cuda stream where the work will be queued
   */
  void release(cudaStream_t stream) {
    set_stream(stream);
    if (nullptr != data_) {
      allocator_->deallocate(data_, capacity_ * sizeof(value_type), stream_);
    }
    data_ = nullptr;
    capacity_ = 0;
    size_ = 0;
  }

  /**
   * @brief returns the underlying allocator used
   *
   * @return the allocator pointer
   */
  std::shared_ptr<Allocator> get_allocator() const { return allocator_; }

 protected:
  value_type* data_;

 private:
  size_type size_;
  size_type capacity_;
  cudaStream_t stream_;
  std::shared_ptr<Allocator> allocator_;

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
  void set_stream(cudaStream_t stream) {
    if (stream_ != stream) {
      cudaEvent_t event;
      CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(event, stream_));
      CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
      stream_ = stream;
      CUDA_CHECK(cudaEventDestroy(event));
    }
  }
};  // class buffer_base

}  // namespace raft
