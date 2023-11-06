/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

// #include <raft/core/logger.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <sys/mman.h>

#include <cstddef>

namespace rmm::mr {
/**
 * @brief `device_memory_resource` derived class that uses mmap to allocate memory.
 * This class enables memory allocation using huge pages.
 * It is assumed that the allocated memory is directly accessible on device. This currently only
 * works on GH systems.
 */
class cuda_huge_page_resource final : public device_memory_resource {
 public:
  cuda_huge_page_resource()                                          = default;
  ~cuda_huge_page_resource() override                                = default;
  cuda_huge_page_resource(cuda_huge_page_resource const&)            = default;
  cuda_huge_page_resource(cuda_huge_page_resource&&)                 = default;
  cuda_huge_page_resource& operator=(cuda_huge_page_resource const&) = default;
  cuda_huge_page_resource& operator=(cuda_huge_page_resource&&)      = default;

  /**
   * @brief Query whether the resource supports use of non-null CUDA streams for
   * allocation/deallocation. `cuda_huge_page_resource` does not support streams.
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
  void* do_allocate(std::size_t bytes, cuda_stream_view) override
  {
    void* _addr{nullptr};
    _addr = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (_addr == MAP_FAILED) {
      // RAFT_LOG_ERROR("mmap failed");

      exit(-1);
    }
    if (madvise(_addr, bytes, MADV_HUGEPAGE) == -1) {
      // RAFT_LOG_ERROR("madvise");
      munmap(_addr, bytes);
      exit(-1);
    }
    memset(_addr, 0, bytes);
    return _addr;
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
  void do_deallocate(void* ptr, std::size_t size, cuda_stream_view) override
  {
    if (munmap(ptr, size) == -1) {
      // RAFT_LOG_ERROR("munmap");
      exit(-1);
    }
  }

  /**
   * @brief Compare this resource to another.
   *
   * Two cuda_huge_page_resources always compare equal, because they can each
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
    return dynamic_cast<cuda_huge_page_resource const*>(&other) != nullptr;
  }

  /**
   * @brief Get free and available memory for memory resource
   *
   * @throws `rmm::cuda_error` if unable to retrieve memory info.
   *
   * @return std::pair contaiing free_size and total_size of memory
   */
  [[nodiscard]] std::pair<std::size_t, std::size_t> do_get_mem_info(cuda_stream_view) const override
  {
    std::size_t free_size{};
    std::size_t total_size{};
    RMM_CUDA_TRY(cudaMemGetInfo(&free_size, &total_size));
    return std::make_pair(free_size, total_size);
  }
};
}  // namespace rmm::mr