/*
 * Copyright (2019) Sandia Corporation
 *
 * The source code is licensed under the 3-clause BSD license found in the LICENSE file
 * thirdparty/LICENSES/mdarray.license
 */

/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/integer_utils.hpp>

#include <sys/mman.h>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace raft {

/**
 * @brief A thin wrapper over mmap for implementing the host mdarray
 * container policy.
 *
 */
template <typename T>
struct host_container {
  using value_type                                  = std::remove_cv_t<T>;
  static inline constexpr size_t kHugePageThreshold = 1024ull * 1024ull * 1024ull;  // 1GB
  static inline constexpr size_t kHugePageSize      = 2ull * 1024ull * 1024ull;     // 2MB

 private:
  std::size_t bytesize_ = 0;
  value_type* data_     = nullptr;

  static inline auto calculate_size(std::size_t count) -> std::size_t
  {
    auto size = count * sizeof(value_type);
    if (size < kHugePageThreshold) { return size; }
    return raft::round_up_safe(size, kHugePageSize);
  }

  static inline auto mmap_verbose(size_t length, int prot, int flags, int fd, off_t offset)
    -> value_type*
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
    if (length >= kHugePageThreshold && reinterpret_cast<uintptr_t>(ptr) % kHugePageSize == 0) {
      madvise(ptr, length, MADV_HUGEPAGE);
    }
    return reinterpret_cast<value_type*>(ptr);
  }

 public:
  using size_type = std::size_t;

  using reference       = value_type&;
  using const_reference = value_type const&;

  using pointer       = value_type*;
  using const_pointer = value_type const*;

  using iterator       = pointer;
  using const_iterator = const_pointer;

  explicit host_container(std::size_t count)
    : bytesize_(calculate_size(count)),
      data_(mmap_verbose(bytesize_, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0))
  {
  }
  ~host_container() noexcept
  {
    if (data_ != nullptr) { munmap(data_, bytesize_); }
  }

  host_container(host_container&& other)
    : bytesize_{std::exchange(other.bytesize_, 0)}, data_{std::exchange(other.data_, nullptr)}
  {
  }
  host_container& operator=(host_container&& other)
  {
    std::swap(this->bytesize_, other.bytesize_);
    std::swap(this->data_, other.data_);
    return *this;
  }
  host_container(host_container const&) = delete;  // Copying disallowed: one array one owner
  host_container& operator=(host_container const&) = delete;

  /**
   * @brief Index operator that returns a reference to the actual data.
   */
  template <typename Index>
  auto operator[](Index i) noexcept -> reference
  {
    return data_[i];
  }
  /**
   * @brief Index operator that returns a reference to the actual data.
   */
  template <typename Index>
  auto operator[](Index i) const noexcept -> const_reference
  {
    return data_[i];
  }

  [[nodiscard]] auto data() noexcept -> pointer { return data_; }
  [[nodiscard]] auto data() const noexcept -> const_pointer { return data_; }
};

/**
 * @brief A container policy for host mdarray.
 */
template <typename ElementType>
class host_vector_policy {
 public:
  using element_type          = ElementType;
  using container_type        = host_container<element_type>;
  using pointer               = typename container_type::pointer;
  using const_pointer         = typename container_type::const_pointer;
  using reference             = typename container_type::reference;
  using const_reference       = typename container_type::const_reference;
  using accessor_policy       = std::experimental::default_accessor<element_type>;
  using const_accessor_policy = std::experimental::default_accessor<element_type const>;

 public:
  auto create(raft::resources const&, size_t n) -> container_type { return container_type(n); }

  constexpr host_vector_policy() noexcept(std::is_nothrow_default_constructible_v<ElementType>) =
    default;

  [[nodiscard]] constexpr auto access(container_type& c, size_t n) const noexcept -> reference
  {
    return c[n];
  }
  [[nodiscard]] constexpr auto access(container_type const& c, size_t n) const noexcept
    -> const_reference
  {
    return c[n];
  }

  [[nodiscard]] auto make_accessor_policy() noexcept { return accessor_policy{}; }
  [[nodiscard]] auto make_accessor_policy() const noexcept { return const_accessor_policy{}; }
};

}  // namespace raft
