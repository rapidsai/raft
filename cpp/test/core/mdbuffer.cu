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

#include "../test_utils.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/mdbuffer.cuh>
#include <utility>

namespace raft {

TEST(MDBuffer, FromHost)
{
  auto res             = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};
  auto data = make_host_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) { return x * 7 + y * 11 + z * 13; };

  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        data(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }

  auto buffer = mdbuffer(data);
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::host);
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(),
            std::as_const(buffer).view<memory_type::host>().data_handle());
  EXPECT_EQ(buffer.view().index(), variant_index_from_memory_type(memory_type::host));

  buffer = mdbuffer(data.view());
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::host);
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::host>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(),
            std::as_const(buffer).view<memory_type::host>().data_handle());

  auto original_data_handle = data.data_handle();
  buffer                    = mdbuffer(std::move(data));
  EXPECT_TRUE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::host);
  EXPECT_EQ(buffer.view<memory_type::host>().data_handle(), original_data_handle);

  auto buffer2 = mdbuffer(res, buffer);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::host);
  EXPECT_EQ(buffer2.view<memory_type::host>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::host);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::host);
  EXPECT_EQ(buffer2.view<memory_type::host>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::device);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::device);
  EXPECT_NE(buffer2.view<memory_type::device>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::managed);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::managed);
  EXPECT_NE(buffer2.view<memory_type::managed>().data_handle(),
            buffer.view<memory_type::host>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::pinned);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::pinned);
  EXPECT_NE(buffer2.view<memory_type::pinned>().data_handle(),
            buffer.view<memory_type::host>().data_handle());
}

TEST(MDBuffer, FromDevice)
{
  auto res             = device_resources{};
  auto constexpr depth = std::uint32_t{5};
  auto constexpr rows  = std::uint32_t{3};
  auto constexpr cols  = std::uint32_t{2};
  auto data = make_device_mdarray<int, std::uint32_t, layout_c_contiguous, depth, rows, cols>(
    res, extents<std::uint32_t, depth, rows, cols>{});
  auto gen_unique_entry = [](auto&& x, auto&& y, auto&& z) { return x * 7 + y * 11 + z * 13; };

  for (auto i = std::uint32_t{}; i < depth; ++i) {
    for (auto j = std::uint32_t{}; j < rows; ++j) {
      for (auto k = std::uint32_t{}; k < cols; ++k) {
        data(i, j, k) = gen_unique_entry(i, j, k);
      }
    }
  }

  auto buffer = mdbuffer(data);
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::device);
  EXPECT_EQ(buffer.view<memory_type::device>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::device>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::device>().data_handle(),
            std::as_const(buffer).view<memory_type::device>().data_handle());
  EXPECT_EQ(buffer.view().index(), variant_index_from_memory_type(memory_type::device));

  buffer = mdbuffer(data.view());
  EXPECT_FALSE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::device);
  EXPECT_EQ(buffer.view<memory_type::device>().data_handle(), data.data_handle());
  EXPECT_EQ(std::as_const(buffer).view<memory_type::device>().data_handle(), data.data_handle());
  EXPECT_EQ(buffer.view<memory_type::device>().data_handle(),
            std::as_const(buffer).view<memory_type::device>().data_handle());

  auto original_data_handle = data.data_handle();
  buffer                    = mdbuffer(std::move(data));
  EXPECT_TRUE(buffer.is_owning());
  EXPECT_EQ(buffer.mem_type(), memory_type::device);
  EXPECT_EQ(buffer.view<memory_type::device>().data_handle(), original_data_handle);

  auto buffer2 = mdbuffer(res, buffer);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::device);
  EXPECT_EQ(buffer2.view<memory_type::device>().data_handle(),
            buffer.view<memory_type::device>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::host);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::host);
  EXPECT_NE(buffer2.view<memory_type::host>().data_handle(),
            buffer.view<memory_type::device>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::device);
  EXPECT_FALSE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::device);
  EXPECT_EQ(buffer2.view<memory_type::device>().data_handle(),
            buffer.view<memory_type::device>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::managed);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::managed);
  EXPECT_NE(buffer2.view<memory_type::managed>().data_handle(),
            buffer.view<memory_type::device>().data_handle());

  buffer2 = mdbuffer(res, buffer, memory_type::pinned);
  EXPECT_TRUE(buffer2.is_owning());
  EXPECT_EQ(buffer2.mem_type(), memory_type::pinned);
  EXPECT_NE(buffer2.view<memory_type::pinned>().data_handle(),
            buffer.view<memory_type::device>().data_handle());
}

}  // namespace raft