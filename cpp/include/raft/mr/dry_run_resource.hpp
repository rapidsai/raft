/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>

namespace raft::mr {

/**
 * @brief Tracks peak allocation usage without holding real memory.
 *
 * On first allocation, invokes a user-supplied probe callable to obtain a real
 * pointer in the appropriate address space.  That pointer is retained (not freed)
 * for the lifetime of the allocator so every subsequent allocation returns the
 * same valid pointer.  A cleanup callback frees the probed pointer on destruction.
 *
 * All allocations alias the same memory: the pointers are always valid in the
 * proper address space but identical across allocations.
 *
 * Stats (current and peak bytes) are maintained atomically and are co-owned via
 * shared_ptr so they survive type-erasure of the enclosing dry_run_resource.
 */
struct dry_run_allocator {
  static constexpr std::size_t kProbeSize = 256;

  /**
   * @brief Probe the upstream resource exactly once.
   *
   * Thread-safe: uses std::call_once.  The @p probe_fn is invoked at most once;
   * it must return a pointer that remains valid until the cleanup callback runs.
   *
   * @tparam ProbeFn  Callable returning void* (allocates kProbeSize bytes from upstream).
   * @param probe_fn  Called once to allocate a small chunk from the real upstream.
   */
  template <typename ProbeFn>
  void probe_once(ProbeFn&& probe_fn)
  {
    std::call_once(probe_flag_, [&] { probe_ptr_ = probe_fn(); });
  }

  /**
   * @brief Register a cleanup callback that frees the probed pointer.
   *
   * Called by dry_run_resource right after probe_once succeeds.
   * The callback is invoked in the destructor of the last shared_ptr holder.
   */
  void set_cleanup(std::function<void()> fn) { cleanup_ = std::move(fn); }

  ~dry_run_allocator()
  {
    if (cleanup_) cleanup_();
  }

  dry_run_allocator()                                    = default;
  dry_run_allocator(dry_run_allocator const&)            = delete;
  dry_run_allocator& operator=(dry_run_allocator const&) = delete;
  dry_run_allocator(dry_run_allocator&&)                 = delete;
  dry_run_allocator& operator=(dry_run_allocator&&)      = delete;

  [[nodiscard]] auto get_probe_ptr() const noexcept -> void* { return probe_ptr_; }

  void record_allocate(std::size_t bytes) noexcept
  {
    auto new_total    = allocated_bytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto current_peak = peak_bytes_.load(std::memory_order_relaxed);
    while (new_total > current_peak &&
           !peak_bytes_.compare_exchange_weak(
             current_peak, new_total, std::memory_order_relaxed, std::memory_order_relaxed)) {}
  }

  void record_deallocate(std::size_t bytes) noexcept
  {
    allocated_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
  }

  [[nodiscard]] auto get_allocated_bytes() const noexcept -> std::size_t
  {
    return allocated_bytes_.load(std::memory_order_relaxed);
  }

  [[nodiscard]] auto get_peak_bytes() const noexcept -> std::size_t
  {
    return peak_bytes_.load(std::memory_order_relaxed);
  }

 private:
  std::once_flag probe_flag_;
  void* probe_ptr_{nullptr};
  std::function<void()> cleanup_;
  std::atomic<std::size_t> allocated_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
};

/**
 * @brief Resource adaptor that returns a single probed pointer for every allocation
 *        and tracks peak usage without holding real memory.
 *
 * Modeled after raft::mr::statistics_adaptor: a single template handles host,
 * device, pinned, and managed resources depending on the Upstream type.
 *
 * Properties are forwarded from Upstream via cuda::forward_property, so
 * dry_run_resource<host_resource_ref> satisfies host_accessible,
 * dry_run_resource<host_device_resource_ref> satisfies host + device accessible,
 * and dry_run_resource<rmm::device_async_resource_ref> satisfies device_accessible.
 *
 * @tparam Upstream  Stored by value.  Use a ref type for non-owning semantics.
 */
template <typename Upstream>
class dry_run_resource : public cuda::forward_property<dry_run_resource<Upstream>, Upstream> {
  Upstream upstream_;
  std::shared_ptr<dry_run_allocator> alloc_;

  void ensure_probed_sync(std::size_t alignment)
  {
    alloc_->probe_once([&] {
      void* p = upstream_.allocate_sync(dry_run_allocator::kProbeSize, alignment);
      alloc_->set_cleanup([upstream = upstream_, p, alignment] {
        upstream.deallocate_sync(p, dry_run_allocator::kProbeSize, alignment);
      });
      return p;
    });
  }

 public:
  template <typename U, std::enable_if_t<std::is_same_v<std::decay_t<U>, Upstream>, int> = 0>
  explicit dry_run_resource(U&& upstream)
    : upstream_(std::forward<U>(upstream)), alloc_(std::make_shared<dry_run_allocator>())
  {
  }

  [[nodiscard]] auto get_allocator() const noexcept -> std::shared_ptr<dry_run_allocator>
  {
    return alloc_;
  }

  void* allocate_sync(std::size_t bytes, std::size_t alignment = alignof(std::max_align_t))
  {
    ensure_probed_sync(alignment);
    alloc_->record_allocate(bytes);
    return alloc_->get_probe_ptr();
  }

  void deallocate_sync(void*, std::size_t bytes, std::size_t = alignof(std::max_align_t)) noexcept
  {
    alloc_->record_deallocate(bytes);
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void* allocate(cuda::stream_ref stream,
                 std::size_t bytes,
                 std::size_t alignment = alignof(std::max_align_t))
  {
    alloc_->probe_once([&] {
      void* p = upstream_.allocate(stream, dry_run_allocator::kProbeSize, alignment);
      alloc_->set_cleanup([upstream = upstream_, p, alignment] {
        upstream.deallocate(
          cuda::stream_ref{cudaStreamLegacy}, p, dry_run_allocator::kProbeSize, alignment);
      });
      return p;
    });
    alloc_->record_allocate(bytes);
    return alloc_->get_probe_ptr();
  }

  template <typename U = Upstream, std::enable_if_t<cuda::mr::resource<U>, int> = 0>
  void deallocate(cuda::stream_ref,
                  void*,
                  std::size_t bytes,
                  std::size_t = alignof(std::max_align_t)) noexcept
  {
    alloc_->record_deallocate(bytes);
  }

  [[nodiscard]] bool operator==(dry_run_resource const& other) const noexcept
  {
    return upstream_ == other.upstream_;
  }

  [[nodiscard]] auto upstream_resource() noexcept -> Upstream& { return upstream_; }
  [[nodiscard]] auto upstream_resource() const noexcept -> Upstream const& { return upstream_; }
};

template <typename Upstream>
dry_run_resource(Upstream) -> dry_run_resource<Upstream>;

}  // namespace raft::mr
