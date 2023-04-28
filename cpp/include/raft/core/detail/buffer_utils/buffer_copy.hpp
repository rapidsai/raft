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
#include <raft/util/cudart_utils.hpp>
#include <raft/core/detail/buffer_utils/copy_cpu.hpp>
#include <raft/core/device_type.hpp>
#ifndef RAFT_DISABLE_GPU
#include <raft/core/detail/buffer_utils/copy_gpu.hpp>
#endif
#include <raft/core/device_type.hpp>
#include <raft/core/resources.hpp>
namespace raft {
namespace detail {
template <device_type dst_type, device_type src_type, typename T>
void buffer_copy(raft::resources const& handle,
                 T* dst,
                 T const* src,
                 uint32_t size,
                 uint32_t dst_offset,
                 uint32_t src_offset)
{
  copy<dst_type, src_type, T>(handle, dst + dst_offset, src + src_offset, size);
}

template <device_type dst_type, device_type src_type, typename T>
void buffer_copy(raft::resources const& handle, T* dst, T const* src, uint32_t size)
{
  copy<dst_type, src_type, T>(handle, dst, src, size);
}

template <typename T>
void buffer_copy(raft::resources const& handle,
                 T* dst,
                 T const* src,
                 uint32_t size,
                 device_type dst_type,
                 device_type src_type,
                 uint32_t dst_offset,
                 uint32_t src_offset)
{
  if (dst_type == device_type::gpu && src_type == device_type::gpu) {
    copy<device_type::gpu, device_type::gpu, T>(
      handle, dst + dst_offset, src + src_offset, size);
  } else if (dst_type == device_type::cpu && src_type == device_type::cpu) {
    copy<device_type::cpu, device_type::cpu, T>(
      handle, dst + dst_offset, src + src_offset, size);
  } else if (dst_type == device_type::gpu && src_type == device_type::cpu) {
    raft::print_device_vector("dst_1", dst + dst_offset, size, std::cout);
    copy<device_type::gpu, device_type::cpu, T>(
      handle, dst + dst_offset, src + src_offset, size);
      raft::print_device_vector("dst_2", dst + dst_offset, size, std::cout);
  } else if (dst_type == device_type::cpu && src_type == device_type::gpu) {
    copy<device_type::cpu, device_type::gpu, T>(
      handle, dst + dst_offset, src + src_offset, size);
  }
}

template <typename T>
void buffer_copy(raft::resources const& handle,
                 T* dst,
                 T const* src,
                 uint32_t size,
                 device_type dst_type,
                 device_type src_type)
{
  buffer_copy<T>(handle, dst, src, size, dst_type, src_type, 0, 0);
}
}  // namespace detail
}  // namespace raft