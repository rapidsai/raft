/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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

#include <cstddef>
#ifndef RAFT_DISABLE_CUDA
#include <cuda_runtime.h>

#include <type_traits>
#else
#include <raft/core/detail/fail_container_policy.hpp>
#endif

namespace raft {
#ifndef RAFT_DISABLE_CUDA

/**
 * @brief A thin wrapper over cudaMallocHost/cudaFreeHost for implementing the pinned mdarray
 * container policy.
 *
 */
template <typename T>
struct pinned_container {
  using value_type = std::remove_cv_t<T>;

 private:
  value_type* data_ = nullptr;

 public:
  using size_type = std::size_t;

  using reference       = value_type&;
  using const_reference = value_type const&;

  using pointer       = value_type*;
  using const_pointer = value_type const*;

  using iterator       = pointer;
  using const_iterator = const_pointer;

  explicit pinned_container(std::size_t size)
  {
    RAFT_CUDA_TRY(cudaMallocHost(&data_, size * sizeof(value_type)));
  }
  ~pinned_container() noexcept
  {
    if (data_ != nullptr) { RAFT_CUDA_TRY_NO_THROW(cudaFreeHost(data_)); }
  }

  pinned_container(pinned_container&& other) { std::swap(this->data_, other.data_); }
  pinned_container& operator=(pinned_container&& other)
  {
    std::swap(this->data_, other.data_);
    return *this;
  }
  pinned_container(pinned_container const&) = delete;  // Copying disallowed: one array one owner
  pinned_container& operator=(pinned_container const&) = delete;

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
 * @brief A container policy for pinned mdarray.
 */
template <typename ElementType>
struct pinned_vector_policy {
  using element_type          = ElementType;
  using container_type        = pinned_container<element_type>;
  using pointer               = typename container_type::pointer;
  using const_pointer         = typename container_type::const_pointer;
  using reference             = typename container_type::reference;
  using const_reference       = typename container_type::const_reference;
  using accessor_policy       = std::experimental::default_accessor<element_type>;
  using const_accessor_policy = std::experimental::default_accessor<element_type const>;

  auto create(raft::resources const&, size_t n) -> container_type { return container_type(n); }

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
#else
template <typename ElementType>
using pinned_vector_policy = detail::fail_container_policy<ElementType>;
#endif
}  // namespace raft
