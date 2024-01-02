/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <raft/core/memory_type.hpp>

namespace raft {
TEST(MemoryType, IsDeviceAccessible)
{
  static_assert(!is_device_accessible(memory_type::host));
  static_assert(is_device_accessible(memory_type::device));
  static_assert(is_device_accessible(memory_type::managed));
  static_assert(is_device_accessible(memory_type::pinned));
}

TEST(MemoryType, IsHostAccessible)
{
  static_assert(is_host_accessible(memory_type::host));
  static_assert(!is_host_accessible(memory_type::device));
  static_assert(is_host_accessible(memory_type::managed));
  static_assert(is_host_accessible(memory_type::pinned));
}

TEST(MemoryType, IsHostDeviceAccessible)
{
  static_assert(!is_host_device_accessible(memory_type::host));
  static_assert(!is_host_device_accessible(memory_type::device));
  static_assert(is_host_device_accessible(memory_type::managed));
  static_assert(is_host_device_accessible(memory_type::pinned));
}

TEST(MemoryTypeFromPointer, Host)
{
  auto ptr1 = static_cast<void*>(nullptr);
  cudaMallocHost(&ptr1, 1);
  EXPECT_EQ(memory_type_from_pointer(ptr1), memory_type::host);
  cudaFree(ptr1);
  auto ptr2 = static_cast<void*>(nullptr);
  EXPECT_EQ(memory_type_from_pointer(ptr2), memory_type::host);
}

#ifndef RAFT_DISABLE_CUDA
TEST(MemoryTypeFromPointer, Device)
{
  auto ptr = static_cast<void*>(nullptr);
  cudaMalloc(&ptr, 1);
  EXPECT_EQ(memory_type_from_pointer(ptr), memory_type::device);
  cudaFree(ptr);
}
TEST(MemoryTypeFromPointer, Managed)
{
  auto ptr = static_cast<void*>(nullptr);
  cudaMallocManaged(&ptr, 1);
  EXPECT_EQ(memory_type_from_pointer(ptr), memory_type::managed);
  cudaFree(ptr);
}
#endif
}  // namespace raft
