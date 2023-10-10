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
#pragma once

#include <raft/core/resources.hpp>
#include <raft/core/stream_view.hpp>
#ifndef RAFT_DISABLE_CUDA
#include <raft/core/resource/cuda_stream.hpp>
#endif

namespace raft::resource {
struct stream_view_resource : public resource {
  stream_view_resource(raft::stream_view view = raft::stream_view_per_thread) : stream(view) {}
  void* get_resource() override { return &stream; }

  ~stream_view_resource() override {}

 private:
  raft::stream_view stream;
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
struct stream_view_resource_factory : public resource_factory {
 public:
  stream_view_resource_factory(raft::stream_view view = raft::stream_view_per_thread) : stream(view)
  {
  }
  resource_type get_resource_type() override { return resource_type::STREAM_VIEW; }
  resource* make_resource() override { return new stream_view_resource(stream); }

 private:
  raft::stream_view stream;
};

/**
 * @defgroup resource_stream_view stream resource functions compatible with
 * non-CUDA builds
 * @{
 */
/**
 * Load a raft::stream_view from a resources instance (and populate it on the res
 * if needed).
 * @param res raft res object for managing resources
 * @return
 */
inline raft::stream_view get_stream_view(resources const& res)
{
  if (!res.has_resource_factory(resource_type::STREAM_VIEW)) {
    res.add_resource_factory(std::make_shared<stream_view_resource_factory>());
  }
  return *res.get_resource<raft::stream_view>(resource_type::STREAM_VIEW);
};

/**
 * Load a raft::stream__view from a resources instance (and populate it on the res
 * if needed).
 * @param[in] res raft resources object for managing resources
 * @param[in] view raft stream view
 */
inline void set_stream_view(resources const& res, raft::stream_view view)
{
  res.add_resource_factory(std::make_shared<stream_view_resource_factory>(view));
};

/**
 * @brief synchronize a specific stream
 *
 * @param[in] res the raft resources object
 * @param[in] stream stream to synchronize
 */
inline void sync_stream_view(const resources& res, raft::stream_view stream)
{
  stream.interruptible_synchronize();
}

/**
 * @brief synchronize main stream on the resources instance
 */
inline void sync_stream_view(const resources& res) { sync_stream_view(res, get_stream_view(res)); }

/**
 * @}
 */

}  // namespace raft::resource
