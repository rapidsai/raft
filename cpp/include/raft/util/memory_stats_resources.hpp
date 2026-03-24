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
#include <raft/mr/statistics_adaptor.hpp>

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
#include <vector>

namespace raft {

/**
 * @brief Snapshot of memory usage across the six tracked resource types.
 *
 * Returned by accessor methods on dry_run_resources and
 * memory_stats_resources (e.g. get_bytes_peak(), get_bytes_current()).
 */
struct memory_stats {
  std::size_t device_workspace{0};
  std::size_t device_large_workspace{0};
  std::size_t device_global{0};
  std::size_t device_managed{0};
  std::size_t host{0};
  std::size_t host_pinned{0};

  /**
   * @brief Plain sum of all memory stats.
   *
   * Note, this does not take into account the resource hierarchy.
   * For example, it's common that workspace resources are allocated from the device global
   * resource, so they are effectively counted twice in this function.
   */
  [[nodiscard]] inline constexpr auto total() const -> std::size_t
  {
    return device_workspace + device_large_workspace + device_global + device_managed + host +
           host_pinned;
  }
};

/**
 * @brief Resources handle that wraps all reachable memory resources with
 *        statistics adaptors to track actual allocation usage.
 *
 * Inherits from raft::resources, so it can be passed anywhere a
 * raft::resources& is expected.  On construction the handle:
 *   - Materializes all tracked resource types (host, device, pinned,
 *     managed, workspace, large_workspace).
 *   - Takes a snapshot of the original resources to keep them alive.
 *   - Wraps each with statistics_adaptor.
 *   - Replaces global host and device resources with tracked versions.
 *
 * On destruction the handle restores global resources.
 */
class memory_stats_resources : public resources {
 public:
  explicit memory_stats_resources(const resources& existing)
    : resources(existing),
      old_host_ref_(mr::get_default_host_resource()),
      old_device_mr_(rmm::mr::get_current_device_resource()),
      old_device_ref_(rmm::mr::get_current_device_resource_ref())
  {
    init();
  }

  ~memory_stats_resources() override
  {
    mr::set_default_host_resource(old_host_ref_);
    rmm::mr::set_current_device_resource(old_device_mr_);
    rmm::mr::set_current_device_resource_ref(old_device_ref_);
    resources_.clear();
    factories_.clear();
  }

  memory_stats_resources(memory_stats_resources const&)            = delete;
  memory_stats_resources& operator=(memory_stats_resources const&) = delete;
  memory_stats_resources(memory_stats_resources&&)                 = delete;
  memory_stats_resources& operator=(memory_stats_resources&&)      = delete;

  [[nodiscard]] auto get_bytes_current() const -> memory_stats
  {
    return read_field(&mr::resource_stats::bytes_current);
  }

  [[nodiscard]] auto get_bytes_peak() const -> memory_stats
  {
    return read_field(&mr::resource_stats::bytes_peak);
  }

  [[nodiscard]] auto get_bytes_total_allocated() const -> memory_stats
  {
    return read_field(&mr::resource_stats::bytes_total_allocated);
  }

  [[nodiscard]] auto get_bytes_total_deallocated() const -> memory_stats
  {
    return read_field(&mr::resource_stats::bytes_total_deallocated);
  }

  [[nodiscard]] auto get_num_allocations() const -> memory_stats
  {
    return read_field(&mr::resource_stats::num_allocations);
  }

  [[nodiscard]] auto get_num_deallocations() const -> memory_stats
  {
    return read_field(&mr::resource_stats::num_deallocations);
  }

 private:
  using field_ptr = std::atomic<std::int64_t> mr::resource_stats::*;

  [[nodiscard]] auto read_field(field_ptr field) const -> memory_stats
  {
    auto load = [&](const std::shared_ptr<mr::resource_stats>& s) -> std::size_t {
      return static_cast<std::size_t>((s.get()->*field).load(std::memory_order_relaxed));
    };
    return {
      .device_workspace       = load(ws_stats_),
      .device_large_workspace = load(lws_stats_),
      .device_global          = load(device_stats_),
      .device_managed         = load(managed_stats_),
      .host                   = load(host_stats_),
      .host_pinned            = load(pinned_stats_),
    };
  }

  std::vector<pair_resource> snapshot_;

  mr::host_resource_ref old_host_ref_;
  rmm::mr::device_memory_resource* old_device_mr_;
  rmm::device_async_resource_ref old_device_ref_;

  using host_stats_adaptor_t = mr::statistics_adaptor<mr::host_resource_ref>;
  std::unique_ptr<host_stats_adaptor_t> host_adaptor_;

  class device_stats_bridge : public rmm::mr::device_memory_resource {
    mr::statistics_adaptor<rmm::device_async_resource_ref> adaptor_;

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
    explicit device_stats_bridge(mr::statistics_adaptor<rmm::device_async_resource_ref> adaptor)
      : adaptor_(std::move(adaptor))
    {
    }

    [[nodiscard]] auto adaptor_ref() noexcept -> cuda::mr::resource_ref<cuda::mr::device_accessible>
    {
      return adaptor_;
    }
  };

  std::unique_ptr<device_stats_bridge> device_bridge_;

  std::shared_ptr<mr::resource_stats> host_stats_;
  std::shared_ptr<mr::resource_stats> pinned_stats_;
  std::shared_ptr<mr::resource_stats> managed_stats_;
  std::shared_ptr<mr::resource_stats> ws_stats_;
  std::shared_ptr<mr::resource_stats> lws_stats_;
  std::shared_ptr<mr::resource_stats> device_stats_;

  void init()
  {
    auto* ws         = resource::get_workspace_resource(*this);
    auto ws_free     = resource::get_workspace_free_bytes(*this);
    auto ws_upstream = ws->get_upstream_resource();
    auto lws_ref     = resource::get_large_workspace_resource_ref(*this);
    auto pinned_ref  = resource::get_pinned_memory_resource_ref(*this);
    auto managed_ref = resource::get_managed_memory_resource_ref(*this);

    snapshot_ = resources_;

    // --- Host (global) ---
    {
      host_adaptor_ = std::make_unique<host_stats_adaptor_t>(old_host_ref_);
      host_stats_   = host_adaptor_->get_stats();
      mr::set_default_host_resource(mr::host_resource_ref{*host_adaptor_});
    }

    // --- Pinned ---
    {
      mr::statistics_adaptor<mr::host_device_resource_ref> sa{pinned_ref};
      pinned_stats_ = sa.get_stats();
      resource::set_pinned_memory_resource(*this, std::move(sa));
    }

    // --- Managed ---
    {
      mr::statistics_adaptor<mr::host_device_resource_ref> sa{managed_ref};
      managed_stats_ = sa.get_stats();
      resource::set_managed_memory_resource(*this, std::move(sa));
    }

    // --- Device (global) ---
    {
      rmm::device_async_resource_ref dev_ref{*old_device_mr_};
      mr::statistics_adaptor<rmm::device_async_resource_ref> sa{dev_ref};
      device_stats_  = sa.get_stats();
      device_bridge_ = std::make_unique<device_stats_bridge>(std::move(sa));
      rmm::mr::set_current_device_resource(device_bridge_.get());
      rmm::mr::set_current_device_resource_ref(device_bridge_->adaptor_ref());
    }

    // --- Workspace ---
    {
      mr::statistics_adaptor<rmm::device_async_resource_ref> sa{ws_upstream};
      ws_stats_ = sa.get_stats();
      resource::set_workspace_resource(*this, std::move(sa), ws_free);
    }

    // --- Large workspace ---
    {
      mr::statistics_adaptor<rmm::device_async_resource_ref> sa{lws_ref};
      lws_stats_ = sa.get_stats();
      resource::set_large_workspace_resource(*this, std::move(sa));
    }
  }
};

}  // namespace raft
