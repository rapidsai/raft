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

#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <raft/core/mdbuffer.hpp>
#include <raft/core/resources.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/error.hpp>

#include <raft/core/device_container_policy.hpp>

namespace raft {

TEST(Buffer, default_buffer)
{
  auto exts = raft::make_extents<int>(5);
  auto buf = buffer<int, decltype(exts)>();
  EXPECT_EQ(buf.mem_type(), memory_type::host);
  EXPECT_EQ(buf.size(), 0);
}

TEST(Buffer, device_buffer)
{
  raft::resources handle;
  auto data = std::vector<int>{1, 2, 3};
  auto exts = raft::make_extents<size_t>(data.size());
  auto test_buffers = std::vector<buffer<int, decltype(exts)>>{};
  test_buffers.emplace_back(handle, exts, memory_type::device);
  test_buffers.emplace_back(handle, exts, memory_type::device);
  test_buffers.emplace_back(handle, exts, memory_type::device);

  for (auto& buf : test_buffers) {
    ASSERT_EQ(buf.mem_type(), memory_type::device);
    ASSERT_EQ(buf.size(), data.size());
#ifndef RAFT_DISABLE_GPU
    ASSERT_NE(buf.data_handle(), nullptr);
    auto data_out = std::vector<int>(data.size());
    raft::update_device(buf.data_handle(), data.data(), data.size(), raft::resource::get_cuda_stream(handle));
    raft::update_host(data_out.data(), buf.data_handle(), buf.size(), raft::resource::get_cuda_stream(handle));
    EXPECT_THAT(data_out, testing::ElementsAreArray(data));
#endif
  }
}

// TEST(Buffer, non_owning_device_buffer)
// {
//   raft::resources handle;
//   auto data = std::vector<int>{1, 2, 3};
//   auto* ptr_d = static_cast<int*>(nullptr);
// #ifndef RAFT_DISABLE_GPU
//   cudaMalloc(reinterpret_cast<void**>(&ptr_d), sizeof(int) * data.size());
//   cudaMemcpy(static_cast<void*>(ptr_d),
//              static_cast<void*>(data.data()),
//              sizeof(int) * data.size(),
//              cudaMemcpyHostToDevice);
// #endif
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(handle, ptr_d, data.size(), memory_type::device);
//   test_buffers.emplace_back(handle, ptr_d, data.size(), memory_type::device);
// #ifndef RAFT_DISABLE_GPU

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::device);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_EQ(buf.data_handle(), ptr_d);

//     auto data_out = std::vector<int>(data.size());
//     cudaMemcpy(static_cast<void*>(data_out.data()),
//                static_cast<void*>(buf.data_handle()),
//                sizeof(int) * data.size(),
//                cudaMemcpyDeviceToHost);
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
//   cudaFree(reinterpret_cast<void*>(ptr_d));
// #endif
// }

// TEST(Buffer, host_buffer)
// { 
//   raft::resources handle;
//   auto data   = std::vector<int>{1, 2, 3};
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(handle, data.size(), memory_type::host);
//   test_buffers.emplace_back(handle, data.size(), memory_type::host);
//   test_buffers.emplace_back(handle, data.size(), memory_type::host);
//   test_buffers.emplace_back(handle, data.size());

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_NE(buf.data_handle(), nullptr);

//     std::memcpy(
//       static_cast<void*>(buf.data_handle()), static_cast<void*>(data.data()), data.size() * sizeof(int));

//     auto data_out = std::vector<int>(buf.data_handle(), buf.data_handle() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// }

// TEST(Buffer, non_owning_host_buffer)
// {
//   raft::resources handle;
//   auto data   = std::vector<int>{1, 2, 3};
//   std::vector<buffer<int>> test_buffers;
//   test_buffers.emplace_back(handle, data.data(), data.size(), memory_type::host);
//   test_buffers.emplace_back(handle, data.data(), data.size(), memory_type::host);
//   test_buffers.emplace_back(handle, data.data(), data.size());

//   for (auto& buf : test_buffers) { 
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_EQ(buf.data_handle(), data.data());

//     auto data_out = std::vector<int>(buf.data_handle(), buf.data_handle() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// }

// TEST(Buffer, copy_constructor)
// {
//   raft::resources handle;
//   auto data        = std::vector<int>{1, 2, 3};
//   buffer<int> const orig_buffer = buffer(handle, data.data(), data.size(), memory_type::host);

//   // host to host copy operations
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(handle, orig_buffer);
//   test_buffers.emplace_back(handle, orig_buffer, memory_type::host);
//   test_buffers.emplace_back(handle, orig_buffer, memory_type::host);
//   test_buffers.emplace_back(handle, orig_buffer, memory_type::host);

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_NE(buf.data_handle(), orig_buffer.data_handle());

//     auto data_out = std::vector<int>(buf.data_handle(), buf.data_handle() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));

// #ifndef RAFT_DISABLE_GPU
//     // host to device copy operations
//     auto test_dev_buffers = std::vector<buffer<int>>{};
//     test_dev_buffers.emplace_back(handle, orig_buffer, memory_type::device);
//     test_dev_buffers.emplace_back(handle, orig_buffer, memory_type::device);
//     test_dev_buffers.emplace_back(handle, orig_buffer, memory_type::device);
//     for (auto& dev_buf : test_dev_buffers) {
//       data_out = std::vector<int>(data.size());
//       RAFT_CUDA_TRY(cudaMemcpy(static_cast<void*>(data_out.data()), static_cast<void*>(dev_buf.data_handle()), dev_buf.size() * sizeof(int), cudaMemcpyDefault));
//       EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
      
//       // device to device copy operations
//       auto test_dev_copies = std::vector<buffer<int>>{};
//       test_dev_copies.emplace_back(handle, dev_buf, memory_type::device);
//       test_dev_copies.emplace_back(handle, dev_buf, memory_type::device);
//       test_dev_copies.emplace_back(handle, dev_buf, memory_type::device);
//       // for (auto& copy_buf : test_dev_copies) {
//       //   data_out = std::vector<int>(data.size());
//       //   RAFT_CUDA_TRY(cudaMemcpy(static_cast<void*>(data_out.data()), static_cast<void*>(copy_buf.data_handle()), copy_buf.size() * sizeof(int), cudaMemcpyDefault));
//       //   EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//       // }

//     //   // device to host copy operations
//     //   auto test_host_buffers = std::vector<buffer<int>>{};
//     //   test_host_buffers.emplace_back(handle, dev_buf, memory_type::host);
//     //   test_host_buffers.emplace_back(handle, dev_buf, memory_type::host);
//     //   test_host_buffers.emplace_back(handle, dev_buf, memory_type::host);
//     //   for (auto& host_buf : test_host_buffers) {
//     //     data_out = std::vector<int>(host_buf.data_handle(), host_buf.data_handle() + host_buf.size());
//     //     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//     //   }
//     }
// #endif
//   }
// }

// TEST(Buffer, move_buffer)
// {
//   raft::resources handle;
//   auto data   = std::vector<int>{1, 2, 3};
//   auto test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(buffer<int>(handle, data.data(), data.size(), memory_type::host));
//   test_buffers.emplace_back(handle, buffer<int>(handle, data.data(), data.size(), memory_type::host), memory_type::host);
//   test_buffers.emplace_back(handle, buffer<int>(handle, data.data(), data.size(), memory_type::host), memory_type::host);
//   test_buffers.emplace_back(handle, buffer<int>(handle, data.data(), data.size(), memory_type::host), memory_type::host);

//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::host);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_EQ(buf.data_handle(), data.data());

//     auto data_out = std::vector<int>(buf.data_handle(), buf.data_handle() + buf.size());
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// #ifndef RAFT_DISABLE_GPU
//   test_buffers = std::vector<buffer<int>>{};
//   test_buffers.emplace_back(handle, buffer<int>(handle, data.data(), data.size(), memory_type::host), memory_type::device);
//   test_buffers.emplace_back(handle, buffer<int>(handle, data.data(), data.size(), memory_type::host), memory_type::device);
//   test_buffers.emplace_back(handle, buffer<int>(handle, data.data(), data.size(), memory_type::host), memory_type::device);
//   for (auto& buf : test_buffers) {
//     ASSERT_EQ(buf.mem_type(), memory_type::device);
//     ASSERT_EQ(buf.size(), data.size());
//     ASSERT_NE(buf.data_handle(), data.data());

//     auto data_out = std::vector<int>(buf.size());
//     RAFT_CUDA_TRY(cudaMemcpy(static_cast<void*>(data_out.data()), static_cast<void*>(buf.data_handle()), buf.size() * sizeof(int), cudaMemcpyDefault));
//     EXPECT_THAT(data_out, ::testing::ElementsAreArray(data));
//   }
// #endif
// }

// TEST(Buffer, move_assignment_buffer)
// {
//   raft::resources handle;
//   auto data = std::vector<int>{1, 2, 3};

// #ifndef RAFT_DISABLE_GPU
//   auto buf = buffer<int>{handle, data.data(), data.size() - 1, memory_type::device};
// #else
//   auto buf = buffer<int>{handle, data.data(), data.size() - 1, memory_type::host};
// #endif
//   buf      = buffer<int>{handle, data.size(), memory_type::host};

//   ASSERT_EQ(buf.mem_type(), memory_type::host);
//   ASSERT_EQ(buf.size(), data.size());
// }

// TEST(Buffer, partial_buffer_copy)
// {
//   raft::resources handle;
//   auto data1 = std::vector<int>{1, 2, 3, 4, 5};
//   auto data2 = std::vector<int>{0, 0, 0, 0, 0};
//   auto expected = std::vector<int>{0, 3, 4, 5, 0};
// #ifndef RAFT_DISABLE_GPU
//   auto buf1 = buffer<int>{handle, buffer<int>{handle, data1.data(), data1.size(), memory_type::host}, memory_type::device};
// #else
//   auto buf1 = buffer<int>{handle, data1.data(), data1.size(), memory_type::host};
// #endif
//   auto buf2 = buffer<int>{handle, data2.data(), data2.size(), memory_type::host};
//   copy<true>(handle, buf2, buf1, 1, 2, 3);
//   copy<false>(handle, buf2, buf1, 1, 2, 3);
//   EXPECT_THROW(copy<true>(handle, buf2, buf1, 1, 2, 4), out_of_bounds);
// }

// TEST(Buffer, buffer_copy_overloads)
// {
//   raft::resources handle;
//   auto data        = std::vector<int>{1, 2, 3};
//   auto expected = data;
//   auto orig_host_buffer = buffer<int>(handle, data.data(), data.size(), memory_type::host);
//   auto orig_dev_buffer = buffer<int>(handle, orig_host_buffer, memory_type::device);
//   auto copy_dev_buffer = buffer<int>(handle, data.size(), memory_type::device);
  
//   // copying host to host
//   auto data_out = std::vector<int>(data.size());
//   auto copy_host_buffer = buffer<int>(handle, data_out.data(), data.size(), memory_type::host);
//   copy<true>(handle, copy_host_buffer, orig_host_buffer);
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

//   // copying host to host with stream
//   data_out = std::vector<int>(data.size());
//   copy_host_buffer = buffer<int>(handle, data_out.data(), data.size(), memory_type::host);
//   copy<true>(handle, copy_host_buffer, orig_host_buffer);
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

//   // copying host to host with offset
//   data_out = std::vector<int>(data.size() + 1);
//   copy_host_buffer = buffer<int>(handle, data_out.data(), data.size(), memory_type::host);
//   copy<true>(handle, copy_host_buffer, orig_host_buffer, 2, 1, 1);
//   expected = std::vector<int>{0, 0, 2, 0};
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

// #ifndef RAFT_DISABLE_GPU
//   // copy device to host
//   data_out = std::vector<int>(data.size());
//   copy_host_buffer = buffer<int>(handle, data_out.data(), data.size(), memory_type::host);
//   copy<true>(handle, copy_host_buffer, orig_dev_buffer);
//   expected = data;
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));

//   // copy device to host with stream
//   data_out = std::vector<int>(data.size());
//   copy_host_buffer = buffer<int>(handle, data_out.data(), data.size(), memory_type::host);
//   copy<true>(handle, copy_host_buffer, orig_dev_buffer);
//   expected = data;
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));
  
//   // copy device to host with offset
//   data_out = std::vector<int>(data.size() + 1);
//   copy_host_buffer = buffer<int>(handle, data_out.data(), data.size(), memory_type::host);
//   copy<true>(handle, copy_host_buffer, orig_dev_buffer, 2, 1, 1);
//   expected = std::vector<int>{0, 0, 2, 0};
//   EXPECT_THAT(data_out, ::testing::ElementsAreArray(expected));
// #endif
// }

}