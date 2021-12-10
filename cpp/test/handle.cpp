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
#include <cstddef>
#include <iostream>
#include <memory>
#include <raft/handle.hpp>

namespace raft {

TEST(Raft, HandleDefault)
{
  handle_t h;
  ASSERT_EQ(0, h.get_device());
  ASSERT_EQ(nullptr, h.get_stream());
  ASSERT_NE(nullptr, h.get_cublas_handle());
  ASSERT_NE(nullptr, h.get_cusolver_dn_handle());
  ASSERT_NE(nullptr, h.get_cusolver_sp_handle());
  ASSERT_NE(nullptr, h.get_cusparse_handle());
}

TEST(Raft, Handle)
{
  handle_t h(4);
  ASSERT_EQ(4, h.get_num_internal_streams());
  cudaStream_t stream;
  RAFT_CUDA_TRY(cudaStreamCreate(&stream));
  h.set_stream(stream);
  ASSERT_EQ(stream, h.get_stream());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  RAFT_CUDA_TRY(cudaStreamDestroy(stream));
}

TEST(Raft, GetInternalStreams)
{
  handle_t h(4);
  auto streams = h.get_internal_streams();
  ASSERT_EQ(4U, streams.size());
}

TEST(Raft, GetHandleFromPool)
{
  handle_t parent(4);

  handle_t child(parent, 2);
  ASSERT_EQ(parent.get_internal_stream(2), child.get_stream());

  child.set_stream(parent.get_internal_stream(3));
  ASSERT_EQ(parent.get_internal_stream(3), child.get_stream());
  ASSERT_NE(parent.get_internal_stream(2), child.get_stream());

  ASSERT_EQ(parent.get_device(), child.get_device());
}

TEST(Raft, GetHandleStreamViews)
{
  handle_t parent(4);

  handle_t child(parent, 2);
  ASSERT_EQ(parent.get_internal_stream_view(2), child.get_stream_view());
  ASSERT_EQ(parent.get_internal_stream_view(2).value(), child.get_stream_view().value());
  EXPECT_FALSE(child.get_stream_view().is_default());
}
}  // namespace raft
