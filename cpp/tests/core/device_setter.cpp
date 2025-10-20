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
#include <raft/core/device_setter.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

namespace raft {
TEST(DeviceSetter, ScopedDevice)
{
  auto device_a = int{};
  auto device_b = int{device_setter::get_device_count() > 1};
  if (device_b == device_a) {
    RAFT_LOG_WARN("Only 1 CUDA device detected. device_setter test will be trivial");
  }
  auto initial_device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&initial_device));
  auto current_device = initial_device;
  {
    auto scoped_device = device_setter{device_a};
    // Confirm that device is currently device_a
    RAFT_CUDA_TRY(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, device_a);
    // Confirm that get_current_device reports expected device
    EXPECT_EQ(current_device, device_setter::get_current_device());
  }

  // Confirm that device went back to initial value once setter was out of
  // scope
  RAFT_CUDA_TRY(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, initial_device);

  {
    auto scoped_device = device_setter{device_b};
    // Confirm that device is currently device_b
    RAFT_CUDA_TRY(cudaGetDevice(&current_device));
    EXPECT_EQ(current_device, device_b);
    // Confirm that get_current_device reports expected device
    EXPECT_EQ(current_device, device_setter::get_current_device());
  }

  // Confirm that device went back to initial value once setter was out of
  // scope
  RAFT_CUDA_TRY(cudaGetDevice(&current_device));
  EXPECT_EQ(current_device, initial_device);

  {
    auto scoped_device1 = device_setter{device_b};
    auto scoped_device2 = device_setter{device_a};
    RAFT_CUDA_TRY(cudaGetDevice(&current_device));
    // Confirm that multiple setters behave as expected, with the last
    // constructed taking precedence
    EXPECT_EQ(current_device, device_a);
  }
}
}  // namespace raft
