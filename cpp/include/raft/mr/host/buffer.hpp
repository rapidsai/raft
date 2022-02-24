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

#include "allocator.hpp"
#include <memory>
#include <raft/mr/buffer_base.hpp>
#include <raft/mr/device/buffer.hpp>

namespace raft {
namespace mr {
namespace host {

/**
 * @brief RAII object owning a contigous typed host buffer (aka pinned memory).
 *        The passed in allocator supports asynchronus allocation and
 *        deallocation so this can also be used for temporary memory
 *
 * @code{.cpp}
 * template<typename T>
 * void foo(const T* in_d , T* out_d, ..., cudaStream_t stream) {
 *   ...
 *   raft::mr::host::buffer<T> temp(stream, 0);
 *   ...
 *   temp.resize(n);
 *   raft::copy(temp.data(), in_d, temp.size());
 *   ...
 *   raft::copy(out_d, temp.data(), temp.size());
 *   temp.release(stream);
 *   ...
 * }
 * @endcode
 */
template <typename T>
class buffer : public buffer_base<T, allocator> {
 public:
  using size_type       = typename buffer_base<T, allocator>::size_type;
  using value_type      = typename buffer_base<T, allocator>::value_type;
  using iterator        = typename buffer_base<T, allocator>::iterator;
  using const_iterator  = typename buffer_base<T, allocator>::const_iterator;
  using reference       = typename buffer_base<T, allocator>::reference;
  using const_reference = typename buffer_base<T, allocator>::const_reference;

  buffer() = delete;

  buffer(const buffer& other) = delete;

  buffer& operator=(const buffer& other) = delete;

  buffer(std::shared_ptr<allocator> alloc, const device::buffer<T>& other)
    : buffer_base<T, allocator>(alloc, other.get_stream(), other.size())
  {
    if (other.size() > 0) { raft::copy(data_, other.data(), other.size(), other.get_stream()); }
  }

  buffer(std::shared_ptr<allocator> alloc, cudaStream_t stream, size_type n = 0)
    : buffer_base<T, allocator>(alloc, stream, n)
  {
  }

  reference operator[](size_type pos) { return data_[pos]; }

  const_reference operator[](size_type pos) const { return data_[pos]; }

 private:
  using buffer_base<T, allocator>::data_;
};

};  // namespace host
};  // namespace mr
};  // namespace raft