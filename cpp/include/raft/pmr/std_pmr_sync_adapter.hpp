/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#ifndef RAFT_DISABLE_CUDA
#include <cuda/memory_resource>
#endif

#include <cstddef>
#include <memory_resource>

namespace raft::pmr {

/**
 * @brief Adapter wrapping std::pmr::memory_resource to satisfy the cuda::mr::synchronous_resource
 *        concept (allocate_sync / deallocate_sync).
 *
 * In CUDA builds this also advertises cuda::mr::host_accessible so that it can bind to
 * rmm::host_resource_ref.  In RAFT_DISABLE_CUDA builds the property tag is omitted but the
 * allocate_sync / deallocate_sync interface is still provided for host_container.
 */
class std_pmr_sync_adapter {
 public:
  explicit std_pmr_sync_adapter(std::pmr::memory_resource* upstream) noexcept : upstream_(upstream)
  {
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    if (bytes == 0) { return nullptr; }
    return upstream_->allocate(bytes, alignment);
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    if (ptr == nullptr) { return; }
    upstream_->deallocate(ptr, bytes, alignment);
  }

  [[nodiscard]] bool operator==(std_pmr_sync_adapter const& other) const noexcept
  {
    return upstream_ == other.upstream_;
  }

  [[nodiscard]] bool operator!=(std_pmr_sync_adapter const& other) const noexcept
  {
    return !(*this == other);
  }

#ifndef RAFT_DISABLE_CUDA
  friend void get_property(std_pmr_sync_adapter const&, cuda::mr::host_accessible) noexcept {}
#endif

 private:
  std::pmr::memory_resource* upstream_;
};

}  // namespace raft::pmr
