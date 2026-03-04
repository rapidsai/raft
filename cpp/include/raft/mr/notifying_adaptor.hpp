/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/memory_resource>
#include <cuda/std/atomic>
#include <cuda/stream_ref>

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>

namespace raft::mr {

/**
 * @brief A simple notifier that can be used to signal that one or more allocations/deallocations
 * have occurred.
 */
class notifier {
 private:
  // NB: using `cuda::std` in place of `std`,
  // because this may happen to be included in pre-C++20 code downstream.
  cuda::std::atomic_flag flag_;  // Note, meaning of the flag is inverted

 public:
  notifier() noexcept { flag_.test_and_set(cuda::std::memory_order_relaxed); }

  /**
   * @brief Change the state to "one or more allocations/deallocations have occurred".
   *        Then wake up one waiting thread.
   */
  void notify() noexcept
  {
    flag_.clear(cuda::std::memory_order_release);
    flag_.notify_one();
  }

  /**
   * @brief Wait for the state to be "one or more allocations/deallocations have occurred" then
   * reset it "no allocations/deallocations have occurred".
   *
   */
  void wait() noexcept
  {
    flag_.wait(true, cuda::std::memory_order_acquire);
    flag_.test_and_set(cuda::std::memory_order_relaxed);
  }
};

/**
 * @brief Resource adaptor that signals an external notifier on every allocation
 *        or deallocation.
 *
 * Forwards all calls to the upstream resource, then calls notifier::notify().
 * A separate consumer (e.g. resource_monitor) can call notifier::wait() to
 * block until activity occurs.
 *
 * @tparam Upstream  Stored by value.  Use a concrete resource type for owning
 *                   semantics, or a ref type for non-owning semantics.
 */
template <typename Upstream>
class notifying_adaptor : public cuda::forward_property<notifying_adaptor<Upstream>, Upstream> {
  Upstream upstream_;
  std::shared_ptr<notifier> notifier_;

 public:
  // Prevent recursive concept satisfaction when Upstream is a __basic_any type (GCC C++20).
  template <typename U, std::enable_if_t<std::is_same_v<std::decay_t<U>, Upstream>, int> = 0>
  explicit notifying_adaptor(U&& upstream,
                             std::shared_ptr<notifier> n = std::make_shared<notifier>())
    : upstream_(std::forward<U>(upstream)), notifier_(std::move(n))
  {
  }

  /**
   * @brief Get the shared notifier object.
   *
   * @return shared pointer to the notifier object
   */
  [[nodiscard]] auto get_notifier() const noexcept -> std::shared_ptr<notifier>
  {
    return notifier_;
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate_sync(bytes, alignment);
    notifier_->notify();
    return ptr;
  }

  void deallocate_sync(void* ptr,
                       std::size_t bytes,
                       std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate_sync(ptr, bytes, alignment);
    notifier_->notify();
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    void* ptr = upstream_.allocate(stream, bytes, alignment);
    notifier_->notify();
    return ptr;
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void deallocate(cuda::stream_ref stream,
                  void* ptr,
                  std::size_t bytes,
                  std::size_t alignment = alignof(std::max_align_t)) noexcept
  {
    upstream_.deallocate(stream, ptr, bytes, alignment);
    notifier_->notify();
  }

  [[nodiscard]] bool operator==(notifying_adaptor const& other) const noexcept
  {
    return upstream_ == other.upstream_;
  }

  [[nodiscard]] auto upstream_resource() noexcept -> Upstream& { return upstream_; }
  [[nodiscard]] auto upstream_resource() const noexcept -> Upstream const& { return upstream_; }
};

template <typename Upstream>
notifying_adaptor(Upstream, std::shared_ptr<notifier>) -> notifying_adaptor<Upstream>;

}  // namespace raft::mr
