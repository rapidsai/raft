/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/limiting_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <iostream>
#include <memory>

namespace raft {
namespace mr {
namespace device {

TEST(Raft, DeviceBufferAlloc)
{
  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));
  // no allocation at construction
  rmm::device_uvector<char> buff(0, stream);
  ASSERT_EQ(0, buff.size());
  // explicit allocation after construction
  buff.resize(20, stream);
  ASSERT_EQ(20, buff.size());
  // resizing to a smaller buffer size
  buff.resize(10, stream);
  ASSERT_EQ(10, buff.size());
  // explicit deallocation
  buff.release();
  ASSERT_EQ(0, buff.size());
  // use these methods without the explicit stream parameter
  buff.resize(20, stream);
  ASSERT_EQ(20, buff.size());
  buff.resize(10, stream);
  ASSERT_EQ(10, buff.size());
  buff.release();
  ASSERT_EQ(0, buff.size());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

TEST(Raft, DeviceBufferZeroResize)
{
  // Create a limiting_resource_adaptor to track allocations
  auto curr_mr =
    dynamic_cast<rmm::mr::cuda_memory_resource*>(rmm::mr::get_current_device_resource());
  auto limit_mr =
    std::make_shared<rmm::mr::limiting_resource_adaptor<rmm::mr::cuda_memory_resource>>(curr_mr,
                                                                                        1000);

  rmm::mr::set_current_device_resource(limit_mr.get());

  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));
  // no allocation at construction
  rmm::device_uvector<char> buff(10, stream);
  ASSERT_EQ(10, buff.size());
  // explicit allocation after construction
  buff.resize(0, stream);
  ASSERT_EQ(0, buff.size());
  // resizing to a smaller buffer size
  buff.resize(20, stream);
  ASSERT_EQ(20, buff.size());
  // explicit deallocation
  buff.release();
  ASSERT_EQ(0, buff.size());

  // Now check that there is no memory left. (Used to not be true)
  ASSERT_EQ(0, limit_mr->get_allocated_bytes());

  rmm::mr::set_current_device_resource(curr_mr);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

}  // namespace device
}  // namespace mr
}  // namespace raft
