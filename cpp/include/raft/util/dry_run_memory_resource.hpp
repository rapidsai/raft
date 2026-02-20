/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/operators.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <memory_resource>
#include <optional>
#include <utility>

namespace raft::util {

/**
 * @defgroup dry_run_memory Dry-run memory resources
 * @{
 */

/**
 * @brief Statistics collected during a dry-run execution.
 */
struct dry_run_stats {
  std::size_t device_workspace_peak;        ///< Peak device workspace bytes
  std::size_t device_large_workspace_peak;  ///< Peak device large workspace bytes
  std::size_t device_global_peak;           ///< Peak device global allocation bytes
  std::size_t device_managed_peak;          ///< Peak device managed allocation bytes
  std::size_t host_peak;                    ///< Peak host (default pmr) allocation bytes
  std::size_t host_pinned_peak;             ///< Peak host pinned allocation bytes
};

/**
 * @brief Lock-free bump allocator that tracks peak usage without holding real memory.
 *
 * On first allocation, invokes a user-supplied probe callable to obtain a base address
 * (typically by briefly allocating and freeing a small chunk from a real upstream).
 * After that, every allocation bumps an atomic address by kProbeSize bytes to produce
 * unique fake pointers. No real memory is held.
 *
 * Tracks total allocated bytes and peak usage for reporting.
 */
struct dry_run_allocator {
  static constexpr std::size_t kProbeSize = 256;

  /**
   * @brief Record an allocation of @p bytes and return a fake pointer.
   * @tparam ProbeFn Callable with signature `void*()` that allocates and immediately frees
   *         a small chunk from a real upstream, returning the probed pointer.
   * @param bytes The number of bytes to record as allocated.
   * @param probe_fn Called exactly once (on the first allocation) to obtain a base address.
   * @return A fake pointer (must not be dereferenced for data access).
   */
  template <typename ProbeFn>
  auto allocate(std::size_t bytes, ProbeFn&& probe_fn) -> void*
  {
    // Ensure the base address is probed exactly once.
    if (address_.load(std::memory_order_relaxed) <= kAddressLocked) {
      auto addr = kAddressUnset;
      while (!address_.compare_exchange_weak(
        addr, kAddressLocked, std::memory_order_relaxed, std::memory_order_relaxed)) {
        if (addr > kAddressLocked) {
          break;  // The address is already set, so we can use it.
        }
        addr = kAddressUnset;  // Otherwise, wait for the lock to be released
      }
      if (addr == kAddressUnset) {  // We acquired the lock
        try {
          void* probe = probe_fn();
          address_.store(reinterpret_cast<std::uintptr_t>(probe), std::memory_order_relaxed);
        } catch (...) {
          address_.store(kAddressUnset, std::memory_order_relaxed);  // release the lock
          throw;
        }
      }
    }

    // Bump the address atomically to produce a fake pointer.
    void* ptr = reinterpret_cast<void*>(address_.fetch_add(kProbeSize, std::memory_order_relaxed));

    // Track allocated bytes and update peak (lock-free).
    auto new_total    = allocated_bytes_.fetch_add(bytes, std::memory_order_relaxed) + bytes;
    auto current_peak = peak_bytes_.load(std::memory_order_relaxed);
    while (new_total > current_peak &&
           !peak_bytes_.compare_exchange_weak(
             current_peak, new_total, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    return ptr;
  }

  /**
   * @brief Record a deallocation of @p bytes.
   */
  void deallocate(std::size_t bytes) noexcept
  {
    allocated_bytes_.fetch_sub(bytes, std::memory_order_relaxed);
  }

  /// @brief Get the current number of allocated (tracked) bytes.
  [[nodiscard]] auto get_allocated_bytes() const noexcept -> std::size_t
  {
    return allocated_bytes_.load(std::memory_order_relaxed);
  }

  /// @brief Get the peak number of allocated (tracked) bytes.
  [[nodiscard]] auto get_peak_bytes() const noexcept -> std::size_t
  {
    return peak_bytes_.load(std::memory_order_relaxed);
  }

 private:
  static constexpr std::uintptr_t kAddressUnset  = 0x0;
  static constexpr std::uintptr_t kAddressLocked = 0x1;

  std::atomic<std::uintptr_t> address_{kAddressUnset};
  std::atomic<std::size_t> allocated_bytes_{0};
  std::atomic<std::size_t> peak_bytes_{0};
};

/**
 * @brief A device memory resource that tracks allocations without real memory.
 *
 * Wraps a dry_run_allocator behind the rmm::mr::device_memory_resource interface.
 * On first use, briefly probes the upstream to obtain a plausible device base address.
 * After that, every allocation bumps an atomic offset and returns a fake pointer.
 *
 * The returned pointers must NOT be dereferenced — they exist only to satisfy the
 * allocator interface during a dry run.
 */
class dry_run_device_memory_resource : public rmm::mr::device_memory_resource {
 public:
  explicit dry_run_device_memory_resource(rmm::mr::device_memory_resource* upstream)
    : upstream_(upstream)
  {
  }
  ~dry_run_device_memory_resource() override = default;

  [[nodiscard]] auto get_allocated_bytes() const noexcept -> std::size_t
  {
    return alloc_.get_allocated_bytes();
  }
  [[nodiscard]] auto get_peak_bytes() const noexcept -> std::size_t
  {
    return alloc_.get_peak_bytes();
  }

 private:
  auto do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) -> void* override
  {
    return alloc_.allocate(bytes, [&] {
      void* p = upstream_->allocate(stream, dry_run_allocator::kProbeSize);
      upstream_->deallocate(stream, p, dry_run_allocator::kProbeSize);
      return p;
    });
  }

  void do_deallocate(void* /*ptr*/,
                     std::size_t bytes,
                     rmm::cuda_stream_view /*stream*/) noexcept override
  {
    alloc_.deallocate(bytes);
  }

  [[nodiscard]] auto do_is_equal(rmm::mr::device_memory_resource const& other) const noexcept
    -> bool override
  {
    return reinterpret_cast<const rmm::mr::device_memory_resource*>(this) == &other;
  }

  rmm::mr::device_memory_resource* upstream_;
  dry_run_allocator alloc_;
};

/**
 * @brief A host memory resource (std::pmr) that tracks allocations without real memory.
 *
 * Wraps a dry_run_allocator behind the std::pmr::memory_resource interface.
 * Analogous to dry_run_device_memory_resource but for host memory.
 */
class dry_run_host_memory_resource : public std::pmr::memory_resource {
 public:
  explicit dry_run_host_memory_resource(std::pmr::memory_resource* upstream) : upstream_(upstream)
  {
  }
  ~dry_run_host_memory_resource() override = default;

  [[nodiscard]] auto get_allocated_bytes() const noexcept -> std::size_t
  {
    return alloc_.get_allocated_bytes();
  }
  [[nodiscard]] auto get_peak_bytes() const noexcept -> std::size_t
  {
    return alloc_.get_peak_bytes();
  }

 private:
  auto do_allocate(std::size_t bytes, std::size_t alignment) -> void* override
  {
    return alloc_.allocate(bytes, [&] {
      void* p = upstream_->allocate(dry_run_allocator::kProbeSize, alignment);
      upstream_->deallocate(p, dry_run_allocator::kProbeSize, alignment);
      return p;
    });
  }

  void do_deallocate(void* /*ptr*/, std::size_t bytes, std::size_t /*alignment*/) noexcept override
  {
    alloc_.deallocate(bytes);
  }

  [[nodiscard]] auto do_is_equal(std::pmr::memory_resource const& other) const noexcept
    -> bool override
  {
    return reinterpret_cast<const std::pmr::memory_resource*>(this) == &other;
  }

  std::pmr::memory_resource* upstream_;
  dry_run_allocator alloc_;
};

/**
 * @brief RAII manager that replaces memory resources with dry-run versions.
 *
 * On construction, saves all current memory resource state and replaces it with
 * dry-run resources. On destruction, restores all original resources.
 *
 * Global resources (rmm device, std::pmr host) are replaced globally.
 * Handle-local resources (workspace, pinned, managed) are replaced only on the handle.
 *
 * This class only manages resources; the action to be dry-run is executed
 * separately (see dry_run_execute()).
 */
class dry_run_resource_manager {
 public:
  /**
   * @brief Set up dry-run resources on the given raft::resources handle.
   * @param res The resources handle to modify.
   */
  explicit dry_run_resource_manager(const raft::resources& res) : res_(res)
  {
    // Save original global resource state
    orig_global_device_mr_ = rmm::mr::get_current_device_resource();
    orig_pmr_              = std::pmr::get_default_resource();

    // Save handle-local resources
    orig_pinned_mr_  = resource::get_pinned_memory_resource(res);
    orig_managed_mr_ = resource::get_managed_memory_resource(res);

    // Save workspace settings (use accessors that handle lazy initialization)
    auto* workspace_mr       = resource::get_workspace_resource(res);
    workspace_limit_         = workspace_mr->get_allocation_limit();
    orig_workspace_upstream_ = orig_global_device_mr_;

    // Save large workspace
    orig_large_workspace_mr_ = resource::get_large_workspace_resource(res);

    // Create dry-run resources
    dry_run_workspace_ = std::make_shared<dry_run_device_memory_resource>(orig_workspace_upstream_);
    dry_run_large_workspace_ =
      std::make_shared<dry_run_device_memory_resource>(orig_large_workspace_mr_);
    dry_run_global_  = std::make_shared<dry_run_device_memory_resource>(orig_global_device_mr_);
    dry_run_managed_ = std::make_shared<dry_run_device_memory_resource>(orig_managed_mr_);
    dry_run_host_    = std::make_unique<dry_run_host_memory_resource>(orig_pmr_);
    dry_run_pinned_  = std::make_shared<dry_run_host_memory_resource>(orig_pinned_mr_);

    // Replace global resources
    rmm::mr::set_current_device_resource(dry_run_global_.get());
    std::pmr::set_default_resource(dry_run_host_.get());

    // Replace handle-local resources
    resource::set_pinned_memory_resource(res, dry_run_pinned_);
    resource::set_managed_memory_resource(res, dry_run_managed_);
    resource::set_workspace_resource(res, dry_run_workspace_, workspace_limit_, std::nullopt);
    resource::set_large_workspace_resource(res, dry_run_large_workspace_);

    // Set dry-run flag
    resource::set_dry_run_flag(res, true);
  }

  ~dry_run_resource_manager() noexcept
  {
    // Restore dry-run flag
    resource::set_dry_run_flag(res_, false);

    // Restore global resources
    rmm::mr::set_current_device_resource(orig_global_device_mr_);
    std::pmr::set_default_resource(orig_pmr_);

    // Restore handle-local resources
    resource::set_pinned_memory_resource(
      res_, std::shared_ptr<std::pmr::memory_resource>(orig_pinned_mr_, void_op{}));
    resource::set_managed_memory_resource(
      res_, std::shared_ptr<rmm::mr::device_memory_resource>(orig_managed_mr_, void_op{}));

    // Restore workspace resources with original settings.
    // Use non-owning shared_ptrs (void_op deleter) since lifetime is managed externally.
    resource::set_workspace_resource(
      res_,
      std::shared_ptr<rmm::mr::device_memory_resource>(orig_workspace_upstream_, void_op{}),
      workspace_limit_,
      std::nullopt);
    resource::set_large_workspace_resource(
      res_, std::shared_ptr<rmm::mr::device_memory_resource>(orig_large_workspace_mr_, void_op{}));
  }

  // Non-copyable, non-movable
  dry_run_resource_manager(dry_run_resource_manager const&)            = delete;
  dry_run_resource_manager& operator=(dry_run_resource_manager const&) = delete;
  dry_run_resource_manager(dry_run_resource_manager&&)                 = delete;
  dry_run_resource_manager& operator=(dry_run_resource_manager&&)      = delete;

  /**
   * @brief Get the collected dry-run statistics.
   * @return dry_run_stats with peak usage information.
   */
  [[nodiscard]] auto get_stats() const -> dry_run_stats
  {
    return {
      .device_workspace_peak       = dry_run_workspace_->get_peak_bytes(),
      .device_large_workspace_peak = dry_run_large_workspace_->get_peak_bytes(),
      .device_global_peak          = dry_run_global_->get_peak_bytes(),
      .device_managed_peak         = dry_run_managed_->get_peak_bytes(),
      .host_peak                   = dry_run_host_->get_peak_bytes(),
      .host_pinned_peak            = dry_run_pinned_->get_peak_bytes(),
    };
  }

 private:
  const raft::resources& res_;

  // Original global resources
  rmm::mr::device_memory_resource* orig_global_device_mr_{nullptr};
  std::pmr::memory_resource* orig_pmr_{nullptr};

  // Original handle-local resources
  std::pmr::memory_resource* orig_pinned_mr_{nullptr};
  rmm::mr::device_memory_resource* orig_managed_mr_{nullptr};
  std::optional<std::size_t> workspace_limit_;
  rmm::mr::device_memory_resource* orig_workspace_upstream_{nullptr};
  rmm::mr::device_memory_resource* orig_large_workspace_mr_{nullptr};

  // Dry-run resources
  std::shared_ptr<dry_run_device_memory_resource> dry_run_workspace_;
  std::shared_ptr<dry_run_device_memory_resource> dry_run_large_workspace_;
  std::shared_ptr<dry_run_device_memory_resource> dry_run_global_;
  std::shared_ptr<dry_run_device_memory_resource> dry_run_managed_;
  std::unique_ptr<dry_run_host_memory_resource> dry_run_host_;
  std::shared_ptr<dry_run_host_memory_resource> dry_run_pinned_;
};

/**
 * @brief Execute an action in dry-run mode and return memory usage statistics.
 *
 * This function:
 * 1. Replaces all memory resources with dry-run versions (RAII).
 * 2. Executes the provided action.
 * 3. Restores all original resources (RAII destructor).
 * 4. Returns statistics about peak memory usage.
 *
 * The action receives the resources handle and can check the dry-run flag via
 * `raft::resource::get_dry_run_flag(res)` to skip kernel execution.
 *
 * @tparam Action A callable with signature `void(const raft::resources&, Args...)`.
 * @tparam Args Additional argument types to forward to the action.
 * @param res The raft resources handle.
 * @param action The action to execute in dry-run mode.
 * @param args Additional arguments to forward to the action.
 * @return dry_run_stats with peak memory usage from the dry run.
 *
 * @code{.cpp}
 * raft::resources res;
 * auto stats = raft::util::dry_run_execute(res, [](const raft::resources& r) {
 *   my_algorithm(r);
 * });
 * std::cout << "Peak workspace: " << stats.device_workspace_peak << " bytes\n";
 * @endcode
 */
template <typename Action, typename... Args>
auto dry_run_execute(const raft::resources& res, Action&& action, Args&&... args) -> dry_run_stats
{
  dry_run_resource_manager manager(res);
  std::forward<Action>(action)(res, std::forward<Args>(args)...);
  return manager.get_stats();
}

/** @} */

}  // namespace raft::util
