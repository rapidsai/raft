/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <raft/core/device_resources.hpp>

namespace raft {

/**
 * raft::handle_t is being kept around for backwards
 * compatibility and will be removed in a future version.
 *
 * Extending the `raft::handle_t` instead of `using` to
 * minimize needed changes downstream
 * (e.g. existing forward declarations, etc...)
 *
 * Use of `raft::resources` or `raft::handle_t` is preferred.
 */
class handle_t : public raft::device_resources {
 public:
  handle_t(const handle_t& handle,
           std::shared_ptr<rmm::mr::device_memory_resource> workspace_resource)
    : device_resources(handle, workspace_resource)
  {
  }

  handle_t(const handle_t& handle) : device_resources{handle} {}

  handle_t(handle_t&&)            = delete;
  handle_t& operator=(handle_t&&) = delete;

  /**
   * @brief Construct a resources instance with a stream view and stream pool
   *
   * @param[in] stream_view the default stream (which has the default per-thread stream if
   * unspecified)
   * @param[in] stream_pool the stream pool used (which has default of nullptr if unspecified)
   * @param[in] workspace_resource an optional resource used by some functions for allocating
   *            temporary workspaces.
   */
  handle_t(rmm::cuda_stream_view stream_view                  = rmm::cuda_stream_per_thread,
           std::shared_ptr<rmm::cuda_stream_pool> stream_pool = {nullptr},
           std::shared_ptr<rmm::mr::device_memory_resource> workspace_resource = {nullptr})
    : device_resources{stream_view, stream_pool, workspace_resource}
  {
  }

  /** Destroys all held-up resources */
  ~handle_t() override {}
};

}  // end NAMESPACE raft
