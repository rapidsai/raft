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

#include <cstddef>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <raft/core/handle.hpp>

namespace raft {

TEST(Raft, HandleDefault)
{
  handle_t h;
  ASSERT_EQ(0, h.get_device());
  ASSERT_EQ(rmm::cuda_stream_per_thread, h.get_stream());
  ASSERT_NE(nullptr, h.get_cublas_handle());
  ASSERT_NE(nullptr, h.get_cusolver_dn_handle());
  ASSERT_NE(nullptr, h.get_cusolver_sp_handle());
  ASSERT_NE(nullptr, h.get_cusparse_handle());
}

TEST(Raft, Handle)
{
  // test stream pool creation
  constexpr std::size_t n_streams = 4;
  auto stream_pool                = std::make_shared<rmm::cuda_stream_pool>(n_streams);
  handle_t h(rmm::cuda_stream_default, stream_pool);
  ASSERT_EQ(n_streams, h.get_stream_pool_size());

  // test non default stream handle
  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));
  rmm::cuda_stream_view stream_view(stream);
  handle_t handle(stream_view);
  ASSERT_EQ(stream_view, handle.get_stream());
  handle.sync_stream(stream);
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

TEST(Raft, DefaultConstructor)
{
  handle_t handle;

  // Make sure waiting on the default stream pool
  // does not fail.
  handle.wait_stream_pool_on_stream();
  handle.sync_stream_pool();

  auto s1 = handle.get_next_usable_stream();
  auto s2 = handle.get_stream();
  auto s3 = handle.get_next_usable_stream(5);

  ASSERT_EQ(s1, s2);
  ASSERT_EQ(s2, s3);
  ASSERT_EQ(0, handle.get_stream_pool_size());
}

TEST(Raft, GetHandleFromPool)
{
  constexpr std::size_t n_streams = 4;
  auto stream_pool                = std::make_shared<rmm::cuda_stream_pool>(n_streams);
  handle_t parent(rmm::cuda_stream_default, stream_pool);

  for (std::size_t i = 0; i < n_streams; i++) {
    auto worker_stream = parent.get_stream_from_stream_pool(i);
    handle_t child(worker_stream);
    ASSERT_EQ(parent.get_stream_from_stream_pool(i), child.get_stream());
  }
}

}  // namespace raft
