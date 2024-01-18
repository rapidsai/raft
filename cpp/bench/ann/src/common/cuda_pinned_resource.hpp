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

#include <rmm/mr/device/device_memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>

#include <cstddef>

namespace raft::mr {
/**
 * @brief `device_memory_resource` derived class that uses cudaMallocHost/Free for
 * allocation/deallocation.
 *
 * This is almost the same as rmm::mr::host::pinned_memory_resource, but it has
 * device_memory_resource as base class. Pinned memory can be accessed from device,
 * and using this allocator we can create device_mdarray backed by pinned allocator.
 *
 * TODO(tfeher): it would be preferred to just rely on the existing allocator from rmm
 * (pinned_memory_resource), but that is incompatible with the container_policy class
 * for device matrix, because the latter expects a device_memory_resource. We shall
 * revise this once we progress with Issue https://github.com/rapidsai/raft/issues/1819
 */
class cuda_pinned_resource final : public rmm::mr::device_memory_resource {
 public:
  cuda_pinned_resource()                                       = default;
  ~cuda_pinned_resource() override                             = default;
  cuda_pinned_resource(cuda_pinned_resource const&)            = default;
  cuda_pinned_resource(cuda_pinned_resource&&)                 = default;
  cuda_pinned_resource& operator=(cuda_pinned_resource const&) = default;
  cuda_pinned_resource& operator=(cuda_pinned_resource&&)      = default;

  /**
   * @brief Query whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation. `cuda_pinned_resource` does not support streams.
   *
   * @returns bool false
   */
  [[nodiscard]] bool supports_streams() const noexcept override { return false; }

  /**
   * @brief Query whether the resource supports the get_mem_info API.
   *
   * @return true
   */
  [[nodiscard]] bool supports_get_mem_info() const noexcept override { return true; }

 private:
  /**
   * @brief Allocates memory of size at least `bytes` using cudaMalloc.
   *
   * The returned pointer has at least 256B alignment.
   *
   * @note Stream argument is ignored
   *
   * @throws `rmm::bad_alloc` if the requested allocation could not be fulfilled
   *
   * @param bytes The size, in bytes, of the allocation
   * @return void* Pointer to the newly allocated memory
   */
  void* do_allocate(std::size_t bytes, rmm::cuda_stream_view) override
  {
    void* ptr{nullptr};
    RMM_CUDA_TRY_ALLOC(cudaMallocHost(&ptr, bytes));
    return ptr;
  }

  /**
   * @brief Deallocate memory pointed to by \p p.
   *
   * @note Stream argument is ignored.
   *
   * @throws Nothing.
   *
   * @param p Pointer to be deallocated
   */
  void do_deallocate(void* ptr, std::size_t, rmm::cuda_stream_view) override
  {
    RMM_ASSERT_CUDA_SUCCESS(cudaFreeHost(ptr));
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two cuda_pinned_resources always compare equal, because they can each
   * deallocate memory allocated by the other.
   *
   * @throws Nothing.
   *
   * @param other The other resource to compare to
   * @return true If the two resources are equivalent
   * @return false If the two resources are not equal
   */
  [[nodiscard]] bool do_is_equal(device_memory_resource const& other) const noexcept override
  {
    return dynamic_cast<cuda_pinned_resource const*>(&other) != nullptr;
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @return std::pair contaiing free_size and total_size of memory
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(
    rmm::cuda_stream_view) const override
  {
    std::size_t free_size{};
    std::size_t total_size{};
    RMM_CUDA_TRY(cudaMemGetInfo(&free_size, &total_size));
    return std::make_pair(free_size, total_size);
  }
};
}  // namespace raft::mr