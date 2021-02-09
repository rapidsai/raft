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

TEST(Raft, HandleDefault) {
  handle_t h;
  ASSERT_EQ(0, h.get_num_internal_streams());
  ASSERT_EQ(0, h.get_device());
  ASSERT_EQ(nullptr, h.get_stream());
  ASSERT_NE(nullptr, h.get_cublas_handle());
  ASSERT_NE(nullptr, h.get_cusolver_dn_handle());
  ASSERT_NE(nullptr, h.get_cusolver_sp_handle());
  ASSERT_NE(nullptr, h.get_cusparse_handle());
}

TEST(Raft, Handle) {
  handle_t h(4);
  ASSERT_EQ(4, h.get_num_internal_streams());
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  h.set_stream(stream);
  ASSERT_EQ(stream, h.get_stream());
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamDestroy(stream));
}

TEST(Raft, GetInternalStreams) {
  handle_t h(4);
  auto streams = h.get_internal_streams();
  ASSERT_EQ(4U, streams.size());
}

TEST(Raft, GetHandleFromPool) {
  handle_t parent(4);
  int sid = 2;
  auto child = parent.get_handle_from_internal_pool(sid);
  std::cout << "done" << std::endl;

  ASSERT_EQ(parent.get_internal_stream(sid), child.get_stream());
  ASSERT_EQ(0, child.get_num_internal_streams());
  ASSERT_EQ(parent.get_device(), child.get_device());
}
}  // namespace raft
