/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#include <iostream>
#include <memory>
#include <raft/mr/host/buffer.hpp>

namespace raft {
namespace mr {
namespace host {

TEST(Raft, HostBuffer)
{
  auto alloc = std::make_shared<default_allocator>();
  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));
  // no allocation at construction
  buffer<char> buff(alloc, stream);
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
  // use these methods without the explicit stream parameter
  buff.resize(20);
  ASSERT_EQ(20, buff.size());
  buff.resize(10);
  ASSERT_EQ(10, buff.size());
  buff.release();
  ASSERT_EQ(0, buff.size());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

TEST(Raft, DeviceToHostBuffer)
{
  auto d_alloc = std::make_shared<device::default_allocator>();
  auto h_alloc = std::make_shared<default_allocator>();
  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));
  device::buffer<char> d_buff(d_alloc, stream, 32);
  RAFT_CUDA_TRY(cudaMemsetAsync(d_buff.data(), 0, sizeof(char) * d_buff.size(), stream));
  buffer<char> h_buff(h_alloc, d_buff);
  ASSERT_EQ(d_buff.size(), h_buff.size());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

}  // namespace host
}  // namespace mr
}  // namespace raft