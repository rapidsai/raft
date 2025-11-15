/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
