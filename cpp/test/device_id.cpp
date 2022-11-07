/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <gtest/gtest.h>
#include <raft/core/device_id.hpp>
#include <raft/core/device_type.hpp>
#include <raft/core/device_support.hpp>

namespace raft {
TEST(DeviceID, CPU)
{
  auto dev_id = device_id<device_type::cpu>{};
  ASSERT_EQ(dev_id.value(), 0);
  ASSERT_THROW(dev_id.rmm_id(), bad_device_type);
}

TEST(DeviceID, GPU)
{
  auto dev_id = device_id<device_type::gpu>{};
#ifdef RAFT_DISABLE_CUDA
  ASSERT_THROW(dev_id.rmm_id(), cuda_unsupported);
  ASSERT_THROW(dev_id.value(), cuda_unsupported);
#else
  ASSERT_EQ(dev_id.value(), dev_id.rmm_id().value());
#endif
}
}  // namespace raft
