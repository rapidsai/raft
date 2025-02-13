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
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/logger.hpp>

#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>
#include <omp.h>

#include <array>
#include <mutex>
#include <set>

namespace raft {
auto get_test_device_ids()
{
  auto devices      = std::array<int, 2>{int{}, int{}};
  auto device_count = 0;
  RAFT_CUDA_TRY(cudaGetDeviceCount(&device_count));
  devices[1] = int{device_count > 1};
  return devices;
}

TEST(DeviceResourcesManager, ObeysSetters)
{
  auto devices = get_test_device_ids();

  auto streams_per_device = 3;
  auto pools_per_device   = 3;
  auto streams_per_pool   = 7;
  auto workspace_limit    = 2048;
  auto workspace_init     = 1024;
  device_resources_manager::set_streams_per_device(streams_per_device);
  device_resources_manager::set_stream_pools_per_device(pools_per_device, streams_per_pool);
  device_resources_manager::set_mem_pool();
  device_resources_manager::set_workspace_allocation_limit(workspace_limit);

  auto unique_streams = std::array<std::set<cudaStream_t>, 2>{};
  auto unique_pools   = std::array<std::set<rmm::cuda_stream_pool const*>, 2>{};

  // Provide lock for counting unique objects
  auto mtx = std::mutex{};
  auto workspace_mrs =
    std::array<std::shared_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>, 2>{
      nullptr, nullptr};
  auto alternate_workspace_mrs = std::array<std::shared_ptr<rmm::mr::cuda_memory_resource>, 2>{};
  auto upstream_mrs            = std::array<rmm::mr::cuda_memory_resource*, 2>{
    dynamic_cast<rmm::mr::cuda_memory_resource*>(
      rmm::mr::get_per_device_resource(rmm::cuda_device_id{devices[0]})),
    dynamic_cast<rmm::mr::cuda_memory_resource*>(
      rmm::mr::get_per_device_resource(rmm::cuda_device_id{devices[1]}))};

  for (auto i = std::size_t{}; i < devices.size(); ++i) {
    auto scoped_device = device_setter{devices[i]};
    if (upstream_mrs[i] == nullptr) {
      RAFT_LOG_WARN(
        "RMM memory resource already set. Tests for device_resources_manger will be incomplete.");
    } else {
      workspace_mrs[i] =
        std::make_shared<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
          upstream_mrs[i], workspace_init, workspace_limit);
      alternate_workspace_mrs[i] = std::make_shared<rmm::mr::cuda_memory_resource>();
    }
  }

  device_resources_manager::set_workspace_memory_resource(workspace_mrs[0], devices[0]);
  device_resources_manager::set_workspace_memory_resource(workspace_mrs[1], devices[1]);

  // Suppress the many warnings from testing use of setters after initial
  // get_device_resources call
  auto scoped_log_level = log_level_setter{level_enum::error};

  omp_set_dynamic(0);
#pragma omp parallel for num_threads(5)
  for (auto i = std::size_t{}; i < 101; ++i) {
    thread_local auto prev_streams = std::array<std::optional<cudaStream_t>, 2>{};
    auto device                    = devices[i % devices.size()];
    auto const& res                = device_resources_manager::get_device_resources(device);

    auto primary_stream  = res.get_stream().value();
    prev_streams[device] = prev_streams[device].value_or(primary_stream);
    // Expect to receive the same stream every time for a given thread
    EXPECT_EQ(*prev_streams[device], primary_stream);

    // Using RAII device setter here to avoid changing device in other tests
    // that depend on a specific device to be set
    auto scoped_device = device_setter{device};
    auto const& res2   = device_resources_manager::get_device_resources();
    // Expect device_resources to default to current device
    EXPECT_EQ(primary_stream, res2.get_stream().value());

    auto const& pool = res.get_stream_pool();
    EXPECT_EQ(streams_per_pool, pool.get_pool_size());

    auto* mr = dynamic_cast<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>*>(
      rmm::mr::get_current_device_resource());

    if (upstream_mrs[i % devices.size()] != nullptr) {
      // Expect that the current memory resource is a pool memory resource as requested
      EXPECT_NE(mr, nullptr);
    }

    {
      auto lock = std::unique_lock{mtx};
      unique_streams[device].insert(primary_stream);
      unique_pools[device].insert(&pool);
    }
    // Ensure that setters have no effect after get_device_resources call
    device_resources_manager::set_streams_per_device(streams_per_device + 1);
    device_resources_manager::set_stream_pools_per_device(pools_per_device - 1);
    device_resources_manager::set_mem_pool();
    device_resources_manager::set_workspace_allocation_limit(1024);
    device_resources_manager::set_workspace_memory_resource(
      alternate_workspace_mrs[i % devices.size()], devices[i % devices.size()]);
  }

  EXPECT_EQ(streams_per_device, unique_streams[devices[0]].size());
  EXPECT_EQ(streams_per_device, unique_streams[devices[1]].size());
  EXPECT_EQ(pools_per_device, unique_pools[devices[0]].size());
  EXPECT_EQ(pools_per_device, unique_pools[devices[1]].size());
}

}  // namespace raft
