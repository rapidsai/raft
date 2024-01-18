/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <raft/core/cuda_support.hpp>
#include <raft/core/stream_view.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <rmm/cuda_stream_view.hpp>
#endif
namespace raft {
TEST(StreamView, Default)
{
  auto stream = stream_view_per_thread;
  ASSERT_EQ(stream.is_per_thread_default(), raft::CUDA_ENABLED);
  ASSERT_FALSE(stream.is_default());
  if (raft::CUDA_ENABLED) {
    EXPECT_NO_THROW(stream.synchronize());
    EXPECT_NO_THROW(stream.interruptible_synchronize());
  } else {
    EXPECT_THROW(stream.synchronize(), raft::non_cuda_build_error);
    EXPECT_THROW(stream.interruptible_synchronize(), raft::non_cuda_build_error);
  }
  EXPECT_NO_THROW(stream.synchronize_no_throw());
  EXPECT_NO_THROW(stream.synchronize_if_cuda_enabled());
#ifndef RAFT_DISABLE_CUDA
  static_assert(std::is_same_v<decltype(stream.underlying()), rmm::cuda_stream_view>,
                "underlying should return rmm::cuda_stream_view");
#endif
}
}  // namespace raft
