/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <raft/core/memory_type.hpp>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

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
  cudaFreeHost(ptr1);
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
