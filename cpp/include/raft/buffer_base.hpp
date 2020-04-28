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

//TODO: include utils.h
#include "../utils.h"

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
    : _size(n),
      _capacity(n),
      _data(nullptr),
      _stream(stream),
      _allocator(allocator) {
    if (_capacity > 0) {
      _data = static_cast<value_type*>(
        _allocator->allocate(_capacity * sizeof(value_type), _stream));
      CUDA_CHECK(cudaStreamSynchronize(_stream));
    }
  }

  ~buffer_base() {
    if (nullptr != _data) {
      _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
    }
  }

  value_type* data() { return _data; }

  const value_type* data() const { return _data; }

  size_type size() const { return _size; }

  void clear() { _size = 0; }

  iterator begin() { return _data; }

  const_iterator begin() const { return _data; }

  iterator end() { return _data + _size; }

  const_iterator end() const { return _data + _size; }

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
    if (new_capacity > _capacity) {
      value_type* new_data = static_cast<value_type*>(
        _allocator->allocate(new_capacity * sizeof(value_type), _stream));
      if (_size > 0) {
        CUDA_CHECK(cudaMemcpyAsync(new_data, _data, _size * sizeof(value_type),
                                   cudaMemcpyDefault, _stream));
      }
      if (nullptr != _data) {
        _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
      }
      _data = new_data;
      _capacity = new_capacity;
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
    _size = new_size;
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
    if (nullptr != _data) {
      _allocator->deallocate(_data, _capacity * sizeof(value_type), _stream);
    }
    _data = nullptr;
    _capacity = 0;
    _size = 0;
  }

  /**
   * @brief returns the underlying allocator used
   *
   * @return the allocator pointer
   */
  std::shared_ptr<Allocator> get_allocator() const { return _allocator; }

 protected:
  value_type* _data;

 private:
  size_type _size;
  size_type _capacity;
  cudaStream_t _stream;
  std::shared_ptr<Allocator> _allocator;

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
    if (_stream != stream) {
      cudaEvent_t event;
      CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(event, _stream));
      CUDA_CHECK(cudaStreamWaitEvent(stream, event, 0));
      _stream = stream;
      CUDA_CHECK(cudaEventDestroy(event));
    }
  }
};  // class buffer_base

}  // namespace raft
