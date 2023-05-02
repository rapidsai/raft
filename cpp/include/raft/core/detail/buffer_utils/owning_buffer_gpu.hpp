/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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
#include "owning_buffer_base.hpp"
#include <cuda_runtime_api.h>
#include <raft/core/device_type.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_buffer.hpp>

namespace raft {
namespace detail {
template <typename T>
struct owning_buffer<device_type::gpu, T> {
  using value_type = std::remove_const_t<T>;
  owning_buffer() : data_{} {}

  owning_buffer(raft::resources const& handle,
                std::size_t size) noexcept(false)
    : data_{[&size, handle]() {
        return rmm::device_buffer{size * sizeof(value_type), raft::resource::get_cuda_stream(handle)};
      }()}
  {
  }

  auto* get() const { return reinterpret_cast<T*>(data_.data()); }

 private:
  mutable rmm::device_buffer data_;
};
}  // namespace detail
}  // namespace raft