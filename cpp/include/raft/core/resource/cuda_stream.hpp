/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <raft/core/interruptible.hpp>
#include <raft/core/resource/resource_types.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda_runtime.h>

namespace raft::resource {
class cuda_stream_resource : public resource {
 public:
  cuda_stream_resource(rmm::cuda_stream_view stream_view = rmm::cuda_stream_per_thread)
    : stream(stream_view)
  {
  }
  void* get_resource() override { return &stream; }

  ~cuda_stream_resource() override {}

 private:
  rmm::cuda_stream_view stream;
};

/**
 * Factory that knows how to construct a specific raft::resource to populate
 * the resources instance.
 */
class cuda_stream_resource_factory : public resource_factory {
 public:
  cuda_stream_resource_factory(rmm::cuda_stream_view stream_view = rmm::cuda_stream_per_thread)
    : stream(stream_view)
  {
  }
  resource_type get_resource_type() override { return resource_type::CUDA_STREAM_VIEW; }
  resource* make_resource() override { return new cuda_stream_resource(stream); }

 private:
  rmm::cuda_stream_view stream;
};

/**
 * @defgroup resource_cuda_stream CUDA stream resource functions
 * @{
 */
/**
 * Load a rmm::cuda_stream_view from a resources instance (and populate it on the res
 * if needed).
 * @param res raft res object for managing resources
 * @return
 */
inline rmm::cuda_stream_view get_cuda_stream(resources const& res)
{
  if (!res.has_resource_factory(resource_type::CUDA_STREAM_VIEW)) {
    res.add_resource_factory(std::make_shared<cuda_stream_resource_factory>());
  }
  return *res.get_resource<rmm::cuda_stream_view>(resource_type::CUDA_STREAM_VIEW);
};

/**
 * Load a rmm::cuda_stream_view from a resources instance (and populate it on the res
 * if needed).
 * @param[in] res raft resources object for managing resources
 * @param[in] stream_view cuda stream view
 */
inline void set_cuda_stream(resources const& res, rmm::cuda_stream_view stream_view)
{
  res.add_resource_factory(std::make_shared<cuda_stream_resource_factory>(stream_view));
};

/**
 * @brief synchronize a specific stream
 *
 * @param[in] res the raft resources object
 * @param[in] stream stream to synchronize
 */
inline void sync_stream(const resources& res, rmm::cuda_stream_view stream)
{
  interruptible::synchronize(stream);
}

/**
 * @brief synchronize main stream on the resources instance
 */
inline void sync_stream(const resources& res) { sync_stream(res, get_cuda_stream(res)); }

/**
 * @}
 */

}  // namespace raft::resource
