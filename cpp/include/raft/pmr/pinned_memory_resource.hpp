/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <memory_resource>

namespace raft::pmr {

/**
 * @brief A std::pmr::memory_resource backed by cudaMallocHost / cudaFreeHost.
 *
 * This provides CUDA-pinned (page-locked) host memory through the polymorphic
 * memory resource interface. It can be used with host_container_policy to
 * create pinned mdarrays that are compatible with dry-run tracking.
 */
class pinned_memory_resource : public std::pmr::memory_resource {
 public:
  pinned_memory_resource() noexcept           = default;
  ~pinned_memory_resource() noexcept override = default;

 protected:
  void* do_allocate(std::size_t bytes, std::size_t /*alignment*/) override
  {
    if (bytes == 0) { return nullptr; }
    void* ptr = nullptr;
    RAFT_CUDA_TRY(cudaMallocHost(&ptr, bytes));
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/) noexcept override
  {
    if (ptr == nullptr) { return; }
    RAFT_CUDA_TRY_NO_THROW(cudaFreeHost(ptr));
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
  {
    return dynamic_cast<const pinned_memory_resource*>(&other) != nullptr;
  }
};

}  // namespace raft::pmr
