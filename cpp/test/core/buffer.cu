/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <iostream>
#include <cuda_runtime_api.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <raft/core/buffer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/device_type.hpp>
#include <raft/core/memory_type.hpp>

namespace raft {

__global__ void check_buffer_access(int* buf) {
  if (buf[0] == 1) {
    buf[0] = 4;
  }
  if (buf[1] == 2) {
    buf[1] = 5;
  }
  if (buf[2] == 3) {
    buf[2] = 6;
  }
}

TEST(Buffer, device_buffer_access)
{
  auto data = std::vector<int>{1, 2, 3};
  auto expected = std::vector<int>{4, 5, 6};
  raft::resources handle;
  auto buf = buffer<int>(
    handle,
    buffer<int>(handle, data.data(), data.size(), memory_type::host),
    memory_type::device,
    0);
  // check_buffer_access<<<1,1>>>(buf.data());
  // auto data_out = std::vector<int>(expected.size());
  // auto host_buf = buffer<int>(data_out.data(), data_out.size(), memory_type::host);
  // copy<true>(host_buf, buf);
  // ASSERT_EQ(cudaStreamSynchronize(execution_stream{}), cudaSuccess);
  // EXPECT_THAT(data_out, testing::ElementsAreArray(expected));
}

}