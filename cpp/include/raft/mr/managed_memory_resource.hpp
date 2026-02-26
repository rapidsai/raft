/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda/memory_resource>
#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace raft::mr {

/**
 * @brief A cuda::mr::synchronous_resource backed by cudaMallocManaged / cudaFree.
 *
 * Provides CUDA unified (managed) memory accessible from both host and device.
 * Uses synchronous allocation (no stream). Binds to rmm::host_device_resource_ref.
 */
class managed_memory_resource {
 public:
  managed_memory_resource() noexcept                                 = default;
  ~managed_memory_resource() noexcept                                = default;
  managed_memory_resource(managed_memory_resource const&)            = default;
  managed_memory_resource& operator=(managed_memory_resource const&) = default;

  void* allocate_sync(std::size_t bytes, std::size_t /*alignment*/ = alignof(std::max_align_t))
  {
    if (bytes == 0) { return nullptr; }
    void* ptr = nullptr;
    RAFT_CUDA_TRY(cudaMallocManaged(&ptr, bytes));
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t /*bytes*/,
                       std::size_t /*alignment*/ = alignof(std::max_align_t)) noexcept
  {
    if (ptr == nullptr) { return; }
    RAFT_CUDA_TRY_NO_THROW(cudaFree(ptr));
  }

  [[nodiscard]] bool operator==(managed_memory_resource const&) const noexcept { return true; }

  [[nodiscard]] bool operator!=(managed_memory_resource const& other) const noexcept
  {
    return !(*this == other);
  }

  friend void get_property(managed_memory_resource const&, cuda::mr::host_accessible) noexcept {}
  friend void get_property(managed_memory_resource const&, cuda::mr::device_accessible) noexcept {}
};

}  // namespace raft::mr
