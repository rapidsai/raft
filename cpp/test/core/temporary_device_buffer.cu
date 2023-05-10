/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include "../test_utils.cuh"

#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/temporary_device_buffer.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace raft {

TEST(TemporaryDeviceBuffer, DevicePointer)
{
  {
    raft::device_resources handle;
    auto exts  = raft::make_extents<int>(5);
    auto array = raft::make_device_mdarray<int, int>(handle, exts);

    auto d_buf = raft::make_temporary_device_buffer(handle, array.data_handle(), exts);

    ASSERT_EQ(array.data_handle(), d_buf.view().data_handle());
    static_assert(!std::is_const_v<typename decltype(d_buf.view())::element_type>,
                  "element_type should not be const");
  }

  {
    raft::device_resources handle;
    auto exts  = raft::make_extents<int>(5);
    auto array = raft::make_device_mdarray<int, int>(handle, exts);

    auto d_buf = raft::make_readonly_temporary_device_buffer(handle, array.data_handle(), exts);

    ASSERT_EQ(array.data_handle(), d_buf.view().data_handle());
    static_assert(std::is_const_v<typename decltype(d_buf.view())::element_type>,
                  "element_type should be const");
  }
}

TEST(TemporaryDeviceBuffer, HostPointerWithWriteBack)
{
  raft::device_resources handle;
  auto exts  = raft::make_extents<int>(5);
  auto array = raft::make_host_mdarray<int, int>(exts);
  thrust::fill(array.data_handle(), array.data_handle() + array.extent(0), 1);
  rmm::device_uvector<int> result(5, handle.get_stream());

  {
    auto d_buf  = raft::make_writeback_temporary_device_buffer(handle, array.data_handle(), exts);
    auto d_view = d_buf.view();

    thrust::fill(rmm::exec_policy(handle.get_stream()),
                 d_view.data_handle(),
                 d_view.data_handle() + d_view.extent(0),
                 10);
    raft::copy(result.data(), d_view.data_handle(), d_view.extent(0), handle.get_stream());

    static_assert(!std::is_const_v<typename decltype(d_buf.view())::element_type>,
                  "element_type should not be const");
  }

  ASSERT_TRUE(raft::devArrMatchHost(array.data_handle(),
                                    result.data(),
                                    array.extent(0),
                                    raft::Compare<int>(),
                                    handle.get_stream()));
}

}  // namespace raft
