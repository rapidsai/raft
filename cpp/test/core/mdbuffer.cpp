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

#include <gtest/gtest.h>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdbuffer.hpp>
#include <raft/core/mdspan_types.hpp>
#include <raft/core/memory_type.hpp>
#include <raft/core/resources.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/device_mdarray.hpp>
#endif
namespace raft {
TEST(MDBuffer, DefaultConstructor) {
  auto buf = mdbuffer<int, matrix_extent<int>>{};
}

TEST(MDBuffer, FromHost) {
  auto res = raft::resources{};
  auto rows = 3;
  auto features = 5;
  auto matrix = make_host_matrix<float>(res, rows, features);
  auto buf = mdbuffer{matrix};
  ASSERT_EQ(buf.mem_type(), memory_type::host);
  ASSERT_FALSE(buf.is_owning());
  ASSERT_EQ(buf.data_handle(), matrix.data_handle());

  auto* ptr = matrix.data_handle();
  buf = mdbuffer{std::move(matrix)};
  ASSERT_EQ(buf.mem_type(), memory_type::host);
  ASSERT_TRUE(buf.is_owning());
  ASSERT_EQ(buf.data_handle(), ptr);
}

TEST(MDBuffer, FromDevice) {
  auto res = raft::resources{};
  auto rows = 3;
  auto features = 5;
  auto matrix = make_device_matrix<float>(res, rows, features);
  auto buf = mdbuffer{matrix};
  ASSERT_EQ(buf.mem_type(), memory_type::device);
  ASSERT_FALSE(buf.is_owning());
  ASSERT_EQ(buf.data_handle(), matrix.data_handle());

  auto* ptr = matrix.data_handle();
  buf = mdbuffer{std::move(matrix)};
  ASSERT_EQ(buf.mem_type(), memory_type::device);
  ASSERT_TRUE(buf.is_owning());
  ASSERT_EQ(buf.data_handle(), ptr);
}
}  // namespace raft

