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

#include <memory_resource>

namespace raft {

/**
 * @brief A thin wrapper over mmap for implementing the host mdarray
 * container policy.
 *
 */
template <typename T>
struct host_container {
  using value_type = std::remove_cv_t<T>;

 private:
  std::pmr::memory_resource* mr_;
  std::size_t bytesize_ = 0;
  value_type* data_     = nullptr;

 public:
  using size_type = std::size_t;

  using reference       = value_type&;
  using const_reference = value_type const&;

  using pointer       = value_type*;
  using const_pointer = value_type const*;

  using iterator       = pointer;
  using const_iterator = const_pointer;

  host_container(size_type count, std::pmr::memory_resource* mr = nullptr)
    : mr_(mr == nullptr ? std::pmr::get_default_resource() : mr),
      bytesize_(sizeof(value_type) * count),
      data_(static_cast<pointer>(mr_->allocate(bytesize_)))
  {
  }

  ~host_container() noexcept
  {
    if (data_ != nullptr) { mr_->deallocate(data_, bytesize_); }
  }

  host_container(host_container&& other) noexcept
    : mr_{std::exchange(other.mr_, nullptr)},
      bytesize_{std::exchange(other.bytesize_, 0)},
      data_{std::exchange(other.data_, nullptr)}
  {
  }
  host_container& operator=(host_container&& other) noexcept
  {
    std::swap(this->mr_, other.mr_);
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

  /** Recreate the container using the same allocator but possibly with a different size. */
  void recreate(size_type count)
  {
    // NB: we don't preserve the data, so can deallocate first
    //     resizing is not a part of mdarray api anyway.
    if (data_ != nullptr) { mr_->deallocate(data_, bytesize_); }
    bytesize_ = sizeof(value_type) * count;
    data_     = static_cast<pointer>(mr_->allocate(bytesize_));
  }

  [[nodiscard]] auto data() noexcept -> pointer { return data_; }
  [[nodiscard]] auto data() const noexcept -> const_pointer { return data_; }
};

/**
 * @brief A container policy for host mdarray.
 */
template <typename ElementType>
class host_container_policy {
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
  auto create(raft::resources const&, size_t n) -> container_type { return container_type(n, mr_); }

  constexpr host_container_policy() noexcept = default;
  explicit host_container_policy(std::pmr::memory_resource* mr) noexcept : mr_(mr) {}

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

 private:
  std::pmr::memory_resource* mr_{std::pmr::get_default_resource()};
};

}  // namespace raft
