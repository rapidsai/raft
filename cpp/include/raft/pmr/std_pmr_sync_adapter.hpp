/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/memory_resource>

#include <cstddef>
#include <memory_resource>

namespace raft::pmr {

/**
 * @brief Adapter wrapping std::pmr::memory_resource to satisfy the cuda::mr::synchronous_resource
 *        concept (allocate_sync / deallocate_sync).
 *
 * Also advertises cuda::mr::host_accessible so that it can bind to rmm::host_resource_ref.
 */
class std_pmr_sync_adapter {
 public:
  std_pmr_sync_adapter() noexcept : upstream_(std::pmr::get_default_resource()) {}

  explicit std_pmr_sync_adapter(std::pmr::memory_resource* upstream) noexcept : upstream_(upstream)
  {
  }

  std_pmr_sync_adapter(std::pmr::memory_resource& upstream) noexcept  // NOLINT
    : upstream_(&upstream)
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

  friend void get_property(std_pmr_sync_adapter const&, cuda::mr::host_accessible) noexcept {}

  [[nodiscard]] std::pmr::memory_resource* pmr_resource() const noexcept { return upstream_; }

 private:
  std::pmr::memory_resource* upstream_;
};

}  // namespace raft::pmr
