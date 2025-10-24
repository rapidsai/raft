/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/integer_utils.hpp>

#include <linux/mman.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace raft::pmr {

namespace detail {

/** RAII wrapper for a file descriptor. */
struct tmpfile_descriptor {
  explicit tmpfile_descriptor(size_t file_size_bytes) : fd_{std::tmpfile()}
  {
    if (fd_ == nullptr) {
      auto e = errno;
      RAFT_FAIL(
        "raft::pmr::tmpfile_descriptor: failed to open a temporary file (std::tmpfile): errno = "
        "%d, %s",
        e,
        strerror(e));
    }
    if (ftruncate(fileno(fd_), file_size_bytes) == -1) {
      auto e = errno;
      RAFT_FAIL(
        "raft::pmr::tmpfile_descriptor: failed to call `ftruncate` to allocate memory for a "
        "temporary file. errno = %d, %s",
        e,
        strerror(e));
    }
  }

  // No copies for owning struct
  tmpfile_descriptor(const tmpfile_descriptor& res)                      = delete;
  auto operator=(const tmpfile_descriptor& other) -> tmpfile_descriptor& = delete;
  // Moving is fine
  tmpfile_descriptor(tmpfile_descriptor&& other) : fd_{std::exchange(other.fd_, nullptr)} {}
  auto operator=(tmpfile_descriptor&& other) -> tmpfile_descriptor&
  {
    std::swap(this->fd_, other.fd_);
    return *this;
  }

  ~tmpfile_descriptor() noexcept
  {
    if (fd_ != nullptr) { std::fclose(fd_); }
  }

  [[nodiscard]] auto value() const -> FILE* { return fd_; }

 private:
  FILE* fd_ = nullptr;
};

}  // namespace detail

/** Default flags for `mmap_memory_resource`. */
constexpr int kMmapDefault = 0x0;
/** Request 2MB huge pages support through `madvise`. */
constexpr int kMmapRequestHugePages = 0x1;
/** Request memory to be backed by a temporary file. */
constexpr int kMmapFileBacked = 0x2;

class mmap_memory_resource : public std::pmr::memory_resource {
 public:
  explicit mmap_memory_resource(int flags = kMmapDefault) noexcept : flags_{flags} {}
  ~mmap_memory_resource() noexcept override = default;

 protected:
  void* do_allocate(std::size_t bytes, std::size_t alignment) override
  {
    // allocating zero bytes is a no-op
    if (bytes == 0) { return nullptr; }
    auto prot  = PROT_READ | PROT_WRITE;
    auto flags = MAP_ANONYMOUS | MAP_PRIVATE;
    void* ptr  = nullptr;
    if (flags_ & kMmapFileBacked) {
      // Note, we don't need the file descriptor to live beyond the call to mmap:
      //       according to the POSIX specification, mmap retains its own descriptor.
      detail::tmpfile_descriptor fd{bytes};
      ptr = mmap_verbose(bytes, prot, flags, fileno(fd.value()), 0);
    } else {
      ptr = mmap_verbose(bytes, prot, flags, -1, 0);
    }
    if (flags_ & kMmapRequestHugePages) {
      // Find a page-aligned subrange of the allocated memory to madvise
      auto madvize_start = raft::round_up_safe(reinterpret_cast<uintptr_t>(ptr), kHugePageSize);
      auto madvize_end =
        raft::round_down_safe(reinterpret_cast<uintptr_t>(ptr) + bytes, kHugePageSize);
      auto madvize_size = madvize_end - madvize_start;
      madvise(reinterpret_cast<void*>(madvize_start), madvize_size, MADV_HUGEPAGE);
    }
    return ptr;
  }

  void do_deallocate(void* ptr, std::size_t bytes, std::size_t alignment) noexcept override
  {
    if (ptr == nullptr) { return; }
    if (munmap(ptr, bytes) != 0) {
      RAFT_LOG_ERROR("Failed call to raft::host_container_policy:munmap(%p, %zu), error: %s",
                     ptr,
                     bytes,
                     strerror(errno));
    }
  }

  bool do_is_equal(const std::pmr::memory_resource& other) const noexcept override
  {
    auto* other_mmap = dynamic_cast<const mmap_memory_resource*>(&other);
    if (other_mmap == nullptr) { return false; }
    return flags_ == other_mmap->flags_;
  }

 private:
  static inline constexpr size_t kHugePageSize = 2ull * 1024ull * 1024ull;  // 2MB
  int flags_{kMmapDefault};

  static inline auto mmap_verbose(size_t length, int prot, int flags, int fd, off_t offset) -> void*
  {
    if (length == 0) {
      // Empty container is allowed
      return nullptr;
    }
    auto ptr = mmap(nullptr, length, prot, flags, fd, offset);
    if (ptr == MAP_FAILED) {
      RAFT_FAIL(
        "Failed call to raft::host_container_policy:mmap(nullptr, %zu, 0x%08x, 0x%08x, %d, %zd), "
        "error: %s",
        length,
        prot,
        flags,
        fd,
        offset,
        strerror(errno));
    }
    return ptr;
  }
};

}  // namespace raft::pmr
