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
#include <raft/util/memory_stats_resources.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace raft {

/**
 * @defgroup dry_run_memory Dry-run memory resources
 * @{
 */

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
class dry_run_resources : public resources {
 public:
  explicit dry_run_resources(const resources& existing)
    : resources(existing),
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
    mr::set_default_host_resource(old_host_ref_);
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

  [[nodiscard]] auto get_bytes_peak() const -> memory_stats
  {
    if (!active_) return {};
    return {
      .device_workspace       = ws_stats_->get_peak_bytes(),
      .device_large_workspace = lws_stats_->get_peak_bytes(),
      .device_global          = device_stats_->get_peak_bytes(),
      .device_managed         = managed_stats_->get_peak_bytes(),
      .host                   = host_stats_->get_peak_bytes(),
      .host_pinned            = pinned_stats_->get_peak_bytes(),
    };
  }

  [[nodiscard]] auto get_bytes_current() const -> memory_stats
  {
    if (!active_) return {};
    return {
      .device_workspace       = ws_stats_->get_allocated_bytes(),
      .device_large_workspace = lws_stats_->get_allocated_bytes(),
      .device_global          = device_stats_->get_allocated_bytes(),
      .device_managed         = managed_stats_->get_allocated_bytes(),
      .host                   = host_stats_->get_allocated_bytes(),
      .host_pinned            = pinned_stats_->get_allocated_bytes(),
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
    explicit device_bridge(mr::dry_run_resource<rmm::device_async_resource_ref> adaptor)
      : adaptor_(std::move(adaptor))
    {
    }

    [[nodiscard]] auto adaptor_ref() noexcept -> cuda::mr::resource_ref<cuda::mr::device_accessible>
    {
      return adaptor_;
    }
  };

  std::unique_ptr<device_bridge> device_bridge_;

  using counter_t = raft::mr::detail::dry_run_memory_counter;
  std::shared_ptr<counter_t> host_stats_;
  std::shared_ptr<counter_t> pinned_stats_;
  std::shared_ptr<counter_t> managed_stats_;
  std::shared_ptr<counter_t> ws_stats_;
  std::shared_ptr<counter_t> lws_stats_;
  std::shared_ptr<counter_t> device_stats_;

  void init()
  {
    // Force-initialize all affected resources (lazy creation).
    auto* ws         = resource::get_workspace_resource(*this);
    auto ws_free     = resource::get_workspace_free_bytes(*this);
    auto ws_upstream = ws->get_upstream_resource();
    auto lws_ref     = resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref  = resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref = resource::get_managed_memory_resource_ref(*this);

    // Snapshot keeps original resource objects alive while dry-run
    // adaptors hold non-owning refs into them.
    snapshot_ = resources_;

    // --- Host (global) ---
    {
      host_adaptor_ = std::make_unique<host_dry_run_t>(old_host_ref_);
      host_stats_   = host_adaptor_->get_counter();
      mr::set_default_host_resource(mr::host_resource_ref{*host_adaptor_});
    }

    // --- Pinned ---
    {
      mr::dry_run_resource<mr::host_device_resource_ref> dr{pinned_ref};
      pinned_stats_ = dr.get_counter();
      resource::set_pinned_memory_resource(*this, std::move(dr));
    }

    // --- Managed ---
    {
      mr::dry_run_resource<mr::host_device_resource_ref> dr{managed_ref};
      managed_stats_ = dr.get_counter();
      resource::set_managed_memory_resource(*this, std::move(dr));
    }

    // --- Device (global) ---
    {
      rmm::device_async_resource_ref dev_ref{*old_device_mr_};
      mr::dry_run_resource<rmm::device_async_resource_ref> dr{dev_ref};
      device_stats_  = dr.get_counter();
      device_bridge_ = std::make_unique<device_bridge>(std::move(dr));
      rmm::mr::set_current_device_resource(device_bridge_.get());
      rmm::mr::set_current_device_resource_ref(device_bridge_->adaptor_ref());
    }

    // --- Workspace ---
    {
      mr::dry_run_resource<rmm::device_async_resource_ref> dr{ws_upstream};
      ws_stats_ = dr.get_counter();
      resource::set_workspace_resource(*this, std::move(dr), ws_free);
    }

    // --- Large workspace ---
    {
      mr::dry_run_resource<rmm::device_async_resource_ref> dr{lws_ref};
      lws_stats_ = dr.get_counter();
      resource::set_large_workspace_resource(*this, std::move(dr));
    }

    resource::set_dry_run_flag(*this, true);
  }
};

/** @} */

}  // namespace raft

namespace raft::util {

/**
 * @brief Execute an action in dry-run mode and return peak memory usage.
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
 * @return memory_stats with peak memory usage from the dry run.
 *
 * @code{.cpp}
 * raft::resources res;
 * auto stats = raft::util::dry_run_execute(res, [](const raft::resources& r) {
 *   my_algorithm(r);
 * });
 * std::cout << "Peak workspace: " << stats.device_workspace << " bytes\n";
 * @endcode
 */
template <typename Action, typename... Args>
auto dry_run_execute(const raft::resources& res, Action&& action, Args&&... args)
  -> raft::memory_stats
{
  raft::dry_run_resources dry_res(res);
  std::forward<Action>(action)(static_cast<const raft::resources&>(dry_res),
                               std::forward<Args>(args)...);
  return dry_res.get_bytes_peak();
}

}  // namespace raft::util
