/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/mr/host_device_resource.hpp>
#include <raft/mr/host_memory_resource.hpp>
#include <raft/mr/notifying_adaptor.hpp>
#include <raft/mr/resource_monitor.hpp>
#include <raft/mr/statistics_adaptor.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <chrono>
#include <fstream>
#include <memory>
#include <ostream>
#include <string>

namespace raft {

/**
 * @brief A resources handle that wraps all reachable memory resources with
 *        allocation-tracking adaptors and logs CSV statistics from a
 *        background thread.
 *
 * Inherits from raft::resources, so it can be passed anywhere a
 * raft::resources& is expected.  On construction the handle:
 *   - Materializes all tracked resource types (host, device, pinned,
 *     managed, workspace, large_workspace).
 *   - Takes a snapshot of the original resources to keep them alive.
 *   - Wraps each with statistics_adaptor + notifying_adaptor.
 *   - Replaces global host and device resources with tracked versions.
 *   - Starts a background CSV reporter.
 *
 * On destruction the handle stops the reporter and restores the
 * global host and device resources.
 */
class memory_tracking_resources : public resources {
 public:
  using duration = std::chrono::steady_clock::duration;

  /**
   * @brief Construct from an existing resources handle, logging to an ostream.
   *
   * @param existing  Resources to shallow-copy and wrap with tracking.
   * @param out       Output stream for CSV rows (must outlive this object).
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  memory_tracking_resources(const resources& existing,
                            std::ostream& out,
                            duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(&existing, nullptr, &out, sample_interval)
  {
  }

  /**
   * @brief Construct from an existing resources handle, logging to a file.
   *
   * @param existing  Resources to shallow-copy and wrap with tracking.
   * @param file_path Path to the output CSV file (created/truncated).
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  memory_tracking_resources(const resources& existing,
                            const std::string& file_path,
                            duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(
        &existing, std::make_unique<std::ofstream>(file_path), nullptr, sample_interval)
  {
  }

  /**
   * @brief Construct from scratch (default resources), logging to an ostream.
   *
   * @param out       Output stream for CSV rows (must outlive this object).
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  explicit memory_tracking_resources(std::ostream& out,
                                     duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(nullptr, nullptr, &out, sample_interval)
  {
  }

  /**
   * @brief Construct from scratch (default resources), logging to a file.
   *
   * @param file_path Path to the output CSV file (created/truncated).
   * @param sample_interval  Minimum time between successive CSV samples.
   */
  explicit memory_tracking_resources(const std::string& file_path,
                                     duration sample_interval = std::chrono::milliseconds{10})
    : memory_tracking_resources(
        nullptr, std::make_unique<std::ofstream>(file_path), nullptr, sample_interval)
  {
  }

  ~memory_tracking_resources() override
  {
    report_.stop();
    raft::mr::set_default_host_resource(old_host_ref_);
    // Restore pointer map first (also overwrites ref map), then restore the
    // original ref map separately, since the two may have been set independently.
    rmm::mr::set_current_device_resource(old_device_mr_);
    rmm::mr::set_current_device_resource_ref(old_device_ref_);
  }

  memory_tracking_resources(memory_tracking_resources const&)            = delete;
  memory_tracking_resources(memory_tracking_resources&&)                 = delete;
  memory_tracking_resources& operator=(memory_tracking_resources const&) = delete;
  memory_tracking_resources& operator=(memory_tracking_resources&&)      = delete;

  /** @brief Access the underlying CSV reporter (e.g. to read stats). */
  [[nodiscard]] auto report() noexcept -> raft::mr::resource_monitor& { return report_; }

 private:
  memory_tracking_resources(const resources* existing,
                            std::unique_ptr<std::ofstream> owned_stream,
                            std::ostream* out_override,
                            duration sample_interval)
    : resources(existing ? *existing : resources{}),
      owned_stream_(std::move(owned_stream)),
      report_(out_override ? *out_override : *owned_stream_, sample_interval),
      old_host_ref_(raft::mr::get_default_host_resource()),
      old_device_mr_(rmm::mr::get_current_device_resource()),
      old_device_ref_(rmm::mr::get_current_device_resource_ref())
  {
    init();
  }

  // Declaration order determines initialization and destruction order.
  // snapshot_ is destroyed last  (keeps original resource shared_ptrs alive).
  // owned_stream_ outlives report_ (report_ writes to it).
  // report_ is destroyed first of the three (stops background thread).
  std::vector<pair_resource> snapshot_;
  std::unique_ptr<std::ofstream> owned_stream_;
  raft::mr::resource_monitor report_;

  raft::mr::host_resource_ref old_host_ref_;
  rmm::mr::device_memory_resource* old_device_mr_;
  rmm::device_async_resource_ref old_device_ref_;
  std::size_t saved_ws_limit_{};

  using host_stats_t  = raft::mr::statistics_adaptor<raft::mr::host_resource_ref>;
  using host_notify_t = raft::mr::notifying_adaptor<host_stats_t>;
  std::unique_ptr<host_notify_t> host_adaptor_;

  using device_stats_t  = raft::mr::statistics_adaptor<rmm::device_async_resource_ref>;
  using device_notify_t = raft::mr::notifying_adaptor<device_stats_t>;

  // Bridge: exposes device_notify_t as an rmm::mr::device_memory_resource so
  // that set_current_device_resource(ptr) updates both the pointer-based and
  // the ref-based global device resource maps in RMM.
  class device_tracking_bridge : public rmm::mr::device_memory_resource {
    device_notify_t adaptor_;

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
    explicit device_tracking_bridge(device_notify_t adaptor) : adaptor_(std::move(adaptor)) {}

    [[nodiscard]] auto adaptor_ref() noexcept -> rmm::device_async_resource_ref
    {
      return rmm::device_async_resource_ref{adaptor_};
    }
  };

  std::unique_ptr<device_tracking_bridge> device_bridge_;

  void init()
  {
    auto* ws          = raft::resource::get_workspace_resource(*this);
    saved_ws_limit_   = ws->get_allocation_limit();
    auto upstream_ref = ws->get_upstream_resource();
    auto lws_ref      = raft::resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref   = raft::resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref  = raft::resource::get_managed_memory_resource_ref(*this);

    // Keeps original resource objects alive while tracking refs point into them.
    snapshot_ = resources_;

    // --- Host (global) ---
    {
      host_stats_t sa{old_host_ref_};
      report_.register_source("host", sa.get_stats());
      host_adaptor_ = std::make_unique<host_notify_t>(std::move(sa), report_.get_notifier());
      raft::mr::set_default_host_resource(*host_adaptor_);
    }

    // --- Pinned ---
    {
      using stats_t  = raft::mr::statistics_adaptor<raft::mr::host_device_resource_ref>;
      using notify_t = raft::mr::notifying_adaptor<stats_t>;
      stats_t sa{pinned_ref};
      report_.register_source("pinned", sa.get_stats());
      raft::resource::set_pinned_memory_resource(*this,
                                                 notify_t{std::move(sa), report_.get_notifier()});
    }

    // --- Managed ---
    {
      using stats_t  = raft::mr::statistics_adaptor<raft::mr::host_device_resource_ref>;
      using notify_t = raft::mr::notifying_adaptor<stats_t>;
      stats_t sa{managed_ref};
      report_.register_source("managed", sa.get_stats());
      raft::resource::set_managed_memory_resource(*this,
                                                  notify_t{std::move(sa), report_.get_notifier()});
    }

    // --- Device (global) ---
    // Use set_current_device_resource(ptr) to update both the pointer map and the ref map,
    // then overwrite the ref map to point directly at the adaptor (skipping the bridge).
    {
      rmm::device_async_resource_ref dev_ref{*old_device_mr_};
      device_stats_t sa{dev_ref};
      report_.register_source("device", sa.get_stats());
      device_bridge_ = std::make_unique<device_tracking_bridge>(
        device_notify_t{std::move(sa), report_.get_notifier()});
      rmm::mr::set_current_device_resource(device_bridge_.get());
      rmm::mr::set_current_device_resource_ref(device_bridge_->adaptor_ref());
    }

    // --- Workspace (track upstream to preserve limiting_resource_adaptor) ---
    {
      using ws_stats_t  = raft::mr::statistics_adaptor<rmm::device_async_resource_ref>;
      using ws_notify_t = raft::mr::notifying_adaptor<ws_stats_t>;
      ws_stats_t sa{upstream_ref};
      report_.register_source("workspace", sa.get_stats());
      raft::resource::set_workspace_resource(
        *this, ws_notify_t{std::move(sa), report_.get_notifier()}, saved_ws_limit_);
    }

    // --- Large workspace ---
    {
      using lws_stats_t  = raft::mr::statistics_adaptor<rmm::device_async_resource_ref>;
      using lws_notify_t = raft::mr::notifying_adaptor<lws_stats_t>;
      lws_stats_t sa{lws_ref};
      report_.register_source("large_workspace", sa.get_stats());
      raft::resource::set_large_workspace_resource(
        *this, lws_notify_t{std::move(sa), report_.get_notifier()});
    }

    report_.start();
  }
};

}  // namespace raft
