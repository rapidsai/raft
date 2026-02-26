/*
 * SPDX-FileCopyrightText: Copyright (c) 2019 Sandia Corporation
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
 */
/*
 * Copyright (2019) Sandia Corporation
 *
 * The source code is licensed under the 3-clause BSD license found in the LICENSE file
 * thirdparty/LICENSES/mdarray.license
 */

#pragma once

#include <raft/core/mdspan_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_memory_resource.hpp>

namespace raft {

/**
 * @brief A container using a synchronous memory resource for allocations.
 *
 * Accepts any ref type with allocate_sync / deallocate_sync
 * (e.g. rmm::host_resource_ref, rmm::host_device_resource_ref, std_pmr_sync_adapter).
 *
 * @tparam T element type
 * @tparam SyncMRRef resource ref with allocate_sync/deallocate_sync
 */
template <typename T, typename SyncMRRef>
struct host_container {
  using value_type = std::remove_cv_t<T>;
  using size_type  = std::size_t;

  using reference       = value_type&;
  using const_reference = value_type const&;

  using pointer       = value_type*;
  using const_pointer = value_type const*;

  using iterator       = pointer;
  using const_iterator = const_pointer;

 private:
  SyncMRRef mr_;
  size_type bytesize_ = 0;
  value_type* data_   = nullptr;

 public:
  host_container(size_type count, SyncMRRef mr)
    : mr_(std::move(mr)),
      bytesize_(sizeof(value_type) * count),
      data_(bytesize_ > 0 ? static_cast<pointer>(mr_.allocate_sync(bytesize_)) : nullptr)
  {
  }

  ~host_container() noexcept
  {
    if (bytesize_ > 0 && data_ != nullptr) { mr_.deallocate_sync(data_, bytesize_); }
  }

  host_container(host_container&& other) noexcept
    : mr_{std::move(other.mr_)},
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

  void resize(size_type count)
  {
    auto cur_count = bytesize_ / sizeof(value_type);
    if (count <= cur_count) { return; }
    host_container new_container{count, mr_};
    std::copy(data_, data_ + cur_count, new_container.data_);
    *this = std::move(new_container);
  }

  [[nodiscard]] auto data() noexcept -> pointer { return data_; }
  [[nodiscard]] auto data() const noexcept -> const_pointer { return data_; }
};

/**
 * @brief Container policy for host mdarray.
 *
 * Defaults to raft::mr::get_host_memory_resource().
 */
template <typename ElementType>
class host_container_policy {
 public:
  using element_type          = ElementType;
  using container_type        = host_container<element_type, raft::mr::host_resource_ref>;
  using pointer               = typename container_type::pointer;
  using const_pointer         = typename container_type::const_pointer;
  using reference             = typename container_type::reference;
  using const_reference       = typename container_type::const_reference;
  using accessor_policy       = cuda::std::default_accessor<element_type>;
  using const_accessor_policy = cuda::std::default_accessor<element_type const>;

  host_container_policy() = default;
  explicit host_container_policy(raft::mr::host_resource_ref ref) noexcept : ref_(ref) {}

  auto create(raft::resources const&, size_t n) -> container_type
  {
    return container_type(n, ref_);
  }

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
  raft::mr::host_resource_ref ref_ = raft::mr::get_host_memory_resource();
};

}  // namespace raft
