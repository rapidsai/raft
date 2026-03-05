/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/dry_run_resource.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/host_memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <memory>
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
  std::size_t host_peak;                    ///< Peak host allocation bytes
  std::size_t host_pinned_peak;             ///< Peak host pinned allocation bytes
};

/**
 * @brief Resources handle that wraps all reachable memory resources with
 *        dry-run adaptors and tracks peak allocation usage.
 *
 * Inherits from raft::resources, so it can be passed anywhere a
 * raft::resources& is expected.  On construction the handle:
 *   - If dry-run mode is already active, does nothing (no-op).
 *   - Materializes all tracked resource types (host, device, pinned,
 *     managed, workspace, large_workspace).
 *   - Takes a snapshot of the original resources to keep them alive.
 *   - Wraps each with dry_run_resource.
 *   - Replaces global host and device resources with dry-run versions.
 *   - Sets the dry-run flag.
 *
 * On destruction the handle resets the flag and restores global resources.
 * Composable with memory_tracking_resources in either order.
 */
class dry_run_resources : public raft::resources {
 public:
  explicit dry_run_resources(const raft::resources& existing)
    : raft::resources(existing),
      active_(!resource::get_dry_run_flag(existing)),
      old_host_ref_(raft::mr::get_default_host_resource()),
      old_device_mr_(rmm::mr::get_current_device_resource()),
      old_device_ref_(rmm::mr::get_current_device_resource_ref())
  {
    if (active_) init();
  }

  ~dry_run_resources() override
  {
    if (!active_) return;
    resource::set_dry_run_flag(*this, false);
    raft::mr::set_default_host_resource(old_host_ref_);
    rmm::mr::set_current_device_resource(old_device_mr_);
    rmm::mr::set_current_device_resource_ref(old_device_ref_);

    // Drop all base-class entries so that probe container RAII cleanup runs
    // during derived-member destruction, while snapshot_ is still alive.
    resources_.clear();
    factories_.clear();
  }

  dry_run_resources(dry_run_resources const&)            = delete;
  dry_run_resources& operator=(dry_run_resources const&) = delete;
  dry_run_resources(dry_run_resources&&)                 = delete;
  dry_run_resources& operator=(dry_run_resources&&)      = delete;

  [[nodiscard]] auto get_stats() const -> dry_run_stats
  {
    if (!active_) return {};
    return {
      .device_workspace_peak       = ws_alloc_->get_peak_bytes(),
      .device_large_workspace_peak = lws_alloc_->get_peak_bytes(),
      .device_global_peak          = device_alloc_->get_peak_bytes(),
      .device_managed_peak         = managed_alloc_->get_peak_bytes(),
      .host_peak                   = host_alloc_->get_peak_bytes(),
      .host_pinned_peak            = pinned_alloc_->get_peak_bytes(),
    };
  }

 private:
  // Declaration order determines destruction order.
  // snapshot_ is destroyed last (keeps original resource shared_ptrs alive
  // while dry-run adaptors hold non-owning refs into them).
  std::vector<pair_resource> snapshot_;

  bool active_;
  raft::mr::host_resource_ref old_host_ref_;
  rmm::mr::device_memory_resource* old_device_mr_;
  rmm::device_async_resource_ref old_device_ref_;

  using host_dry_run_t = raft::mr::dry_run_resource<raft::mr::host_resource_ref>;
  std::unique_ptr<host_dry_run_t> host_adaptor_;

  class device_bridge : public rmm::mr::device_memory_resource {
    raft::mr::dry_run_resource<rmm::device_async_resource_ref> adaptor_;

   protected:
    void* do_allocate(std::size_t bytes, rmm::cuda_stream_view stream) override
    {
      return adaptor_.allocate(cuda::stream_ref{stream.value()}, bytes);
    }
    void do_deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream) noexcept override
    {
      adaptor_.deallocate(cuda::stream_ref{stream.value()}, ptr, bytes);
    }
    [[nodiscard]] bool do_is_equal(
      rmm::mr::device_memory_resource const& other) const noexcept override
    {
      return this == &other;
    }

   public:
    explicit device_bridge(raft::mr::dry_run_resource<rmm::device_async_resource_ref> adaptor)
      : adaptor_(std::move(adaptor))
    {
    }

    [[nodiscard]] auto adaptor_ref() noexcept -> cuda::mr::resource_ref<cuda::mr::device_accessible>
    {
      return adaptor_;
    }
  };

  std::unique_ptr<device_bridge> device_bridge_;

  std::shared_ptr<raft::mr::detail::dry_run_memory_counter> host_alloc_;
  std::shared_ptr<raft::mr::detail::dry_run_memory_counter> pinned_alloc_;
  std::shared_ptr<raft::mr::detail::dry_run_memory_counter> managed_alloc_;
  std::shared_ptr<raft::mr::detail::dry_run_memory_counter> ws_alloc_;
  std::shared_ptr<raft::mr::detail::dry_run_memory_counter> lws_alloc_;
  std::shared_ptr<raft::mr::detail::dry_run_memory_counter> device_alloc_;

  void init()
  {
    // Force-initialize all affected resources (lazy creation).
    auto* ws         = raft::resource::get_workspace_resource(*this);
    auto ws_limit    = ws->get_allocation_limit();
    auto ws_upstream = ws->get_upstream_resource();
    auto lws_ref     = raft::resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref  = raft::resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref = raft::resource::get_managed_memory_resource_ref(*this);

    // Snapshot keeps original resource objects alive while dry-run
    // adaptors hold non-owning refs into them.
    snapshot_ = resources_;

    // --- Host (global) ---
    {
      host_adaptor_ = std::make_unique<host_dry_run_t>(old_host_ref_);
      host_alloc_   = host_adaptor_->get_counter();
      raft::mr::set_default_host_resource(raft::mr::host_resource_ref{*host_adaptor_});
    }

    // --- Pinned ---
    {
      raft::mr::dry_run_resource<raft::mr::host_device_resource_ref> dr{pinned_ref};
      pinned_alloc_ = dr.get_counter();
      raft::resource::set_pinned_memory_resource(*this, std::move(dr));
    }

    // --- Managed ---
    {
      raft::mr::dry_run_resource<raft::mr::host_device_resource_ref> dr{managed_ref};
      managed_alloc_ = dr.get_counter();
      raft::resource::set_managed_memory_resource(*this, std::move(dr));
    }

    // --- Device (global) ---
    {
      rmm::device_async_resource_ref dev_ref{*old_device_mr_};
      raft::mr::dry_run_resource<rmm::device_async_resource_ref> dr{dev_ref};
      device_alloc_  = dr.get_counter();
      device_bridge_ = std::make_unique<device_bridge>(std::move(dr));
      rmm::mr::set_current_device_resource(device_bridge_.get());
      rmm::mr::set_current_device_resource_ref(device_bridge_->adaptor_ref());
    }

    // --- Workspace ---
    {
      raft::mr::dry_run_resource<rmm::device_async_resource_ref> dr{ws_upstream};
      ws_alloc_ = dr.get_counter();
      raft::resource::set_workspace_resource(*this, std::move(dr), ws_limit);
    }

    // --- Large workspace ---
    {
      raft::mr::dry_run_resource<rmm::device_async_resource_ref> dr{lws_ref};
      lws_alloc_ = dr.get_counter();
      raft::resource::set_large_workspace_resource(*this, std::move(dr));
    }

    resource::set_dry_run_flag(*this, true);
  }
};

/**
 * @brief Execute an action in dry-run mode and return memory usage statistics.
 *
 * Creates an independent copy of the resources handle with all memory resources
 * replaced by dry-run versions, executes the action, and returns peak usage stats.
 *
 * The action receives the dry-run resources handle (as const raft::resources&)
 * and can check the dry-run flag via raft::resource::get_dry_run_flag(res) to
 * skip kernel execution.
 *
 * @tparam Action A callable with signature void(const raft::resources&, Args...).
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
  dry_run_resources dry_res(res);
  std::forward<Action>(action)(static_cast<const raft::resources&>(dry_res),
                               std::forward<Args>(args)...);
  return dry_res.get_stats();
}

/** @} */

}  // namespace raft::util
