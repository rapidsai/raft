/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/core/resource/device_memory_resource.hpp>
#include <raft/core/resource/dry_run_flag.hpp>
#include <raft/core/resource/managed_memory_resource.hpp>
#include <raft/core/resource/pinned_memory_resource.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/dry_run_memory_resource.hpp>

#include <rmm/mr/device_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace raft::util {

// ===== dry_run_device_memory_resource tests =====

TEST(DryRunDeviceMemoryResource, LazyAllocation)
{
  auto* upstream = rmm::mr::get_current_device_resource();
  dry_run_device_memory_resource mr(upstream);

  // Request 1 GiB; actual allocation should be at most 2 MiB
  constexpr std::size_t kOneGiB = 1024UL * 1024UL * 1024UL;
  void* ptr                     = mr.allocate(rmm::cuda_stream_view{}, kOneGiB);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mr.get_allocated_bytes(), kOneGiB);
  EXPECT_EQ(mr.get_peak_bytes(), kOneGiB);

  mr.deallocate(rmm::cuda_stream_view{}, ptr, kOneGiB);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(mr.get_peak_bytes(), kOneGiB);  // peak should not decrease
}

TEST(DryRunDeviceMemoryResource, SmallAllocation)
{
  auto* upstream = rmm::mr::get_current_device_resource();
  dry_run_device_memory_resource mr(upstream);

  // Request less than 2MB - should allocate the actual requested size
  constexpr std::size_t kSmall = 1024;
  void* ptr                    = mr.allocate(rmm::cuda_stream_view{}, kSmall);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mr.get_allocated_bytes(), kSmall);

  mr.deallocate(rmm::cuda_stream_view{}, ptr, kSmall);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

TEST(DryRunDeviceMemoryResource, PeakTracking)
{
  auto* upstream = rmm::mr::get_current_device_resource();
  dry_run_device_memory_resource mr(upstream);

  constexpr std::size_t kSize1 = 100UL * 1024UL * 1024UL;  // 100 MiB
  constexpr std::size_t kSize2 = 200UL * 1024UL * 1024UL;  // 200 MiB

  void* p1 = mr.allocate(rmm::cuda_stream_view{}, kSize1);
  void* p2 = mr.allocate(rmm::cuda_stream_view{}, kSize2);
  EXPECT_EQ(mr.get_peak_bytes(), kSize1 + kSize2);

  mr.deallocate(rmm::cuda_stream_view{}, p1, kSize1);
  EXPECT_EQ(mr.get_allocated_bytes(), kSize2);
  EXPECT_EQ(mr.get_peak_bytes(), kSize1 + kSize2);  // peak unchanged

  void* p3 = mr.allocate(rmm::cuda_stream_view{}, kSize1 / 2);
  EXPECT_EQ(mr.get_peak_bytes(), kSize1 + kSize2);  // still the previous peak

  mr.deallocate(rmm::cuda_stream_view{}, p2, kSize2);
  mr.deallocate(rmm::cuda_stream_view{}, p3, kSize1 / 2);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

TEST(DryRunDeviceMemoryResource, MultipleAllocations)
{
  auto* upstream = rmm::mr::get_current_device_resource();
  dry_run_device_memory_resource mr(upstream);

  constexpr int kNumAllocs        = 10;
  constexpr std::size_t kEachSize = 50UL * 1024UL * 1024UL;  // 50 MiB each
  void* ptrs[kNumAllocs];

  for (int i = 0; i < kNumAllocs; ++i) {
    ptrs[i] = mr.allocate(rmm::cuda_stream_view{}, kEachSize);
    ASSERT_NE(ptrs[i], nullptr);
  }
  EXPECT_EQ(mr.get_allocated_bytes(), kNumAllocs * kEachSize);
  EXPECT_EQ(mr.get_peak_bytes(), kNumAllocs * kEachSize);

  for (int i = 0; i < kNumAllocs; ++i) {
    mr.deallocate(rmm::cuda_stream_view{}, ptrs[i], kEachSize);
  }
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
}

// ===== dry_run_host_memory_resource tests =====

TEST(DryRunHostMemoryResource, LazyAllocation)
{
  auto* upstream = std::pmr::get_default_resource();
  dry_run_host_memory_resource mr(upstream);

  constexpr std::size_t kLarge = 1024UL * 1024UL * 1024UL;  // 1 GiB
  void* ptr                    = mr.allocate(kLarge);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(mr.get_allocated_bytes(), kLarge);
  EXPECT_EQ(mr.get_peak_bytes(), kLarge);

  mr.deallocate(ptr, kLarge);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(mr.get_peak_bytes(), kLarge);
}

TEST(DryRunHostMemoryResource, PeakTracking)
{
  auto* upstream = std::pmr::get_default_resource();
  dry_run_host_memory_resource mr(upstream);

  constexpr std::size_t kSize1 = 100UL * 1024UL * 1024UL;
  constexpr std::size_t kSize2 = 200UL * 1024UL * 1024UL;

  void* p1 = mr.allocate(kSize1);
  void* p2 = mr.allocate(kSize2);
  EXPECT_EQ(mr.get_peak_bytes(), kSize1 + kSize2);

  mr.deallocate(p1, kSize1);
  mr.deallocate(p2, kSize2);
  EXPECT_EQ(mr.get_allocated_bytes(), 0);
  EXPECT_EQ(mr.get_peak_bytes(), kSize1 + kSize2);
}

// ===== dry_run_flag resource tests =====

TEST(DryRunFlag, DefaultIsFalse)
{
  raft::resources res;
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

TEST(DryRunFlag, SetAndGet)
{
  raft::resources res;
  resource::set_dry_run_flag(res, true);
  EXPECT_TRUE(resource::get_dry_run_flag(res));

  resource::set_dry_run_flag(res, false);
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

// ===== dry_run_resource_manager tests =====

TEST(DryRunResourceManager, SetsAndRestoresFlag)
{
  raft::resources res;
  EXPECT_FALSE(resource::get_dry_run_flag(res));
  {
    dry_run_resource_manager manager(res);
    EXPECT_TRUE(resource::get_dry_run_flag(res));
  }
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

TEST(DryRunResourceManager, RestoresGlobalDeviceResource)
{
  auto* original_mr = rmm::mr::get_current_device_resource();
  raft::resources res;
  {
    dry_run_resource_manager manager(res);
    auto* current_mr = rmm::mr::get_current_device_resource();
    EXPECT_NE(current_mr, original_mr);
  }
  EXPECT_EQ(rmm::mr::get_current_device_resource(), original_mr);
}

TEST(DryRunResourceManager, RestoresHostResource)
{
  auto* original_pmr = std::pmr::get_default_resource();
  raft::resources res;
  {
    dry_run_resource_manager manager(res);
    auto* current_pmr = std::pmr::get_default_resource();
    EXPECT_NE(current_pmr, original_pmr);
  }
  EXPECT_EQ(std::pmr::get_default_resource(), original_pmr);
}

TEST(DryRunResourceManager, RestoresPinnedResource)
{
  raft::resources res;
  auto* original_pinned = resource::get_pinned_memory_resource(res);
  {
    dry_run_resource_manager manager(res);
    auto* current_pinned = resource::get_pinned_memory_resource(res);
    EXPECT_NE(current_pinned, original_pinned);
  }
  EXPECT_EQ(resource::get_pinned_memory_resource(res), original_pinned);
}

TEST(DryRunResourceManager, RestoresManagedResource)
{
  raft::resources res;
  auto* original_managed = resource::get_managed_memory_resource(res);
  {
    dry_run_resource_manager manager(res);
    auto* current_managed = resource::get_managed_memory_resource(res);
    EXPECT_NE(current_managed, original_managed);
  }
  EXPECT_EQ(resource::get_managed_memory_resource(res), original_managed);
}

TEST(DryRunResourceManager, StatsAccuracy)
{
  raft::resources res;
  constexpr std::size_t kAllocSize = 64UL * 1024UL * 1024UL;  // 64 MiB

  dry_run_resource_manager manager(res);

  // Allocate from global device resource
  auto* mr  = rmm::mr::get_current_device_resource();
  void* ptr = mr->allocate(rmm::cuda_stream_view{}, kAllocSize);
  mr->deallocate(rmm::cuda_stream_view{}, ptr, kAllocSize);

  auto stats = manager.get_stats();
  EXPECT_EQ(stats.device_global_peak, kAllocSize);
}

TEST(DryRunResourceManager, PinnedStatsAccuracy)
{
  raft::resources res;
  constexpr std::size_t kAllocSize = 64UL * 1024UL * 1024UL;  // 64 MiB

  dry_run_resource_manager manager(res);

  // Allocate from pinned resource in the handle
  auto* mr  = resource::get_pinned_memory_resource(res);
  void* ptr = mr->allocate(kAllocSize);
  mr->deallocate(ptr, kAllocSize);

  auto stats = manager.get_stats();
  EXPECT_EQ(stats.host_pinned_peak, kAllocSize);
}

TEST(DryRunResourceManager, ManagedStatsAccuracy)
{
  raft::resources res;
  constexpr std::size_t kAllocSize = 64UL * 1024UL * 1024UL;  // 64 MiB

  dry_run_resource_manager manager(res);

  // Allocate from managed resource in the handle
  auto* mr  = resource::get_managed_memory_resource(res);
  void* ptr = mr->allocate(rmm::cuda_stream_view{}, kAllocSize);
  mr->deallocate(rmm::cuda_stream_view{}, ptr, kAllocSize);

  auto stats = manager.get_stats();
  EXPECT_EQ(stats.device_managed_peak, kAllocSize);
}

// ===== dry_run_execute tests =====

TEST(DryRunExecute, BasicExecution)
{
  raft::resources res;
  bool action_ran = false;

  auto stats = dry_run_execute(res, [&](raft::resources const& r) {
    action_ran = true;
    EXPECT_TRUE(resource::get_dry_run_flag(r));

    // Allocate via global device resource
    auto* mr                    = rmm::mr::get_current_device_resource();
    constexpr std::size_t kSize = 32UL * 1024UL * 1024UL;
    void* ptr                   = mr->allocate(rmm::cuda_stream_view{}, kSize);
    mr->deallocate(rmm::cuda_stream_view{}, ptr, kSize);
  });

  EXPECT_TRUE(action_ran);
  EXPECT_FALSE(resource::get_dry_run_flag(res));
  EXPECT_EQ(stats.device_global_peak, 32UL * 1024UL * 1024UL);
}

TEST(DryRunExecute, ExceptionSafety)
{
  raft::resources res;
  auto* original_mr      = rmm::mr::get_current_device_resource();
  auto* original_pmr     = std::pmr::get_default_resource();
  auto* original_pinned  = resource::get_pinned_memory_resource(res);
  auto* original_managed = resource::get_managed_memory_resource(res);

  EXPECT_THROW(dry_run_execute(
                 res, [](raft::resources const&) { throw std::runtime_error("test exception"); }),
               std::runtime_error);

  // Global resources should be restored even after exception
  EXPECT_EQ(rmm::mr::get_current_device_resource(), original_mr);
  EXPECT_EQ(std::pmr::get_default_resource(), original_pmr);
  // Handle-local resources should be restored even after exception
  EXPECT_EQ(resource::get_pinned_memory_resource(res), original_pinned);
  EXPECT_EQ(resource::get_managed_memory_resource(res), original_managed);
  EXPECT_FALSE(resource::get_dry_run_flag(res));
}

}  // namespace raft::util
