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
#include <raft/core/device_type.hpp>
#include <raft/core/execution_stream.hpp>
#include <raft/core/detail/buffer_utils/copy_cpu.hpp>
#ifdef CUML_ENABLE_GPU
#include <raft/core/detail/buffer_utils/copy_gpu.hpp>
#endif
#include <raft/core/device_type.hpp>

namespace raft {

template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, uint32_t size, uint32_t dst_offset, uint32_t src_offset) {
  buffer::detail::copy<dst_type, src_type, T>(dst + dst_offset, src + src_offset, size, execution_stream{});
}

template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, uint32_t size, uint32_t dst_offset, uint32_t src_offset, execution_stream stream) {
  buffer::detail::copy<dst_type, src_type, T>(dst + dst_offset, src + src_offset, size, stream);
}

template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, uint32_t size) {
  buffer::detail::copy<dst_type, src_type, T>(dst, src, size, execution_stream{});
}

template<device_type dst_type, device_type src_type, typename T>
void copy(T* dst, T const* src, uint32_t size, execution_stream stream) {
  buffer::detail::copy<dst_type, src_type, T>(dst, src, size, stream);
}

template<typename T>
void copy(T* dst, T const* src, uint32_t size, device_type dst_type, device_type src_type, uint32_t dst_offset, uint32_t src_offset, execution_stream stream) {
  if (dst_type == device_type::gpu && src_type == device_type::gpu) {
    buffer::detail::copy<device_type::gpu, device_type::gpu, T>(dst + dst_offset, src + src_offset, size, stream);
  } else if (dst_type == device_type::cpu && src_type == device_type::cpu) {
    buffer::detail::copy<device_type::cpu, device_type::cpu, T>(dst + dst_offset, src + src_offset, size, stream);
  } else if (dst_type == device_type::gpu && src_type == device_type::cpu) {
    buffer::detail::copy<device_type::gpu, device_type::cpu, T>(dst + dst_offset, src + src_offset, size, stream);
  } else if (dst_type == device_type::cpu && src_type == device_type::gpu) {
    buffer::detail::copy<device_type::cpu, device_type::gpu, T>(dst + dst_offset, src + src_offset, size, stream);
  }
}

template<typename T>
void copy(T* dst, T const* src, uint32_t size, device_type dst_type, device_type src_type) {
  copy<T>(dst, src, size, dst_type, src_type, 0, 0, execution_stream{});
}

template<typename T>
void copy(T* dst, T const* src, uint32_t size, device_type dst_type, device_type src_type, execution_stream stream) {
  copy<T>(dst, src, size, dst_type, src_type, 0, 0, stream);
}

} // namespace raft