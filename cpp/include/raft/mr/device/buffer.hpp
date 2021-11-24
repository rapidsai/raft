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

#include <memory>
#include <raft/mr/buffer_base.hpp>
#include "allocator.hpp"

namespace raft {
namespace mr {
namespace device {

/**
 * @brief RAII object owning a contiguous typed device buffer. The passed in
 *        allocator supports asynchronous allocation and deallocation so this
 *        can also be used for temporary memory
 *
 * @code{.cpp}
 * template<typename T>
 * void foo(..., cudaStream_t stream) {
 *   ...
 *   raft::mr::device::buffer<T> temp(stream, 0);
 *   ...
 *   temp.resize(n);
 *   kernelA<<<grid,block,0,stream>>>(...,temp.data(),...);
 *   kernelB<<<grid,block,0,stream>>>(...,temp.data(),...);
 *   temp.release();
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

  buffer(std::shared_ptr<allocator> alloc, cudaStream_t stream, size_type n = 0)
    : buffer_base<T, device::allocator>(alloc, stream, n)
  {
  }
};  // class buffer

};  // namespace device
};  // namespace mr
};  // namespace raft
