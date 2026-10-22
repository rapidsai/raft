/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  auto ptr = static_cast<void*>(nullptr);
  EXPECT_EQ(memory_type_from_pointer(ptr), memory_type::host);
}

#ifndef RAFT_DISABLE_CUDA
TEST(MemoryTypeFromPointer, Pinned)
{
  auto ptr = static_cast<void*>(nullptr);
  cudaMallocHost(&ptr, 1);
  EXPECT_EQ(memory_type_from_pointer(ptr), memory_type::pinned);
  EXPECT_TRUE(is_device_accessible(memory_type_from_pointer(ptr)));
  cudaFreeHost(ptr);
}
TEST(MemoryTypeFromPointer, RegisteredHost)
{
  constexpr auto size = std::size_t{1} << 16;
  auto* buf           = new char[size];
  ASSERT_EQ(cudaHostRegister(buf, size, cudaHostRegisterMapped), cudaSuccess);
  EXPECT_EQ(memory_type_from_pointer(buf), memory_type::pinned);
  cudaHostUnregister(buf);
  delete[] buf;
}
TEST(MemoryTypeFromPointer, Pageable)
{
  // Pageable memory is only device accessible when the device uses the host page tables (ATS).
  // On HMM systems cudaPointerGetAttributes may report a non-null devicePointer for it, but that
  // must not make it device accessible.
  auto* buf = new char[1 << 16];
  int ats   = 0;
  cudaDeviceGetAttribute(&ats, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
  auto expected = ats != 0 ? memory_type::managed : memory_type::host;
  EXPECT_EQ(memory_type_from_pointer(buf), expected);
  EXPECT_EQ(is_device_accessible(memory_type_from_pointer(buf)), ats != 0);
  delete[] buf;
}
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
