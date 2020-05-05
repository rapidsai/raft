/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <raft/device_buffer.hpp>
#include <iostream>
#include <memory>

namespace raft {

TEST(Raft, DeviceBuffer) {
  auto allocator = std::make_shared<defaultDeviceAllocator>();
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  // no allocation at construction
  device_buffer<char> buff(allocator, stream);
  ASSERT_EQ(0, buff.size());
  // explicit allocation after construction
  buff.resize(20, stream);
  ASSERT_EQ(20, buff.size());
  // resizing to a smaller buffer size
  buff.resize(10, stream);
  ASSERT_EQ(10, buff.size());
  // explicit deallocation
  buff.release(stream);
  ASSERT_EQ(0, buff.size());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

}  // namespace raft
